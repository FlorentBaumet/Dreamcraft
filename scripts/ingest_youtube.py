"""
Ingestion YouTube -> episodes npz pseudo-etiquetes par l'IDM.

Pipeline : video (mp4) -> reechantillonnage 20 Hz -> frames 64x64 RGB
        -> IDM (touches devinees + flag menu) -> decoupage en episodes de
           6000 frames (5 min, comme VPT) -> data/processed_youtube/<nom>_XXX.npz

Format de sortie IDENTIQUE aux npz VPT (frames NHWC uint8, actions (N-1,10),
gui (N-1,)) -> tout le pipeline existant (train, rollout, chooseur) les avale.

Lance : .venv\\Scripts\\python.exe scripts\\ingest_youtube.py <video.mp4> [prefixe]
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dreamcraft.models.idm import IDM  # noqa: E402

OUT_DIR = ROOT / "data" / "processed_youtube"
IDM_CKPT = ROOT / "outputs" / "idm" / "idm_best.pt"

RES = 64
TARGET_FPS = 20.0
EP_LEN = 6001          # frames par episode (comme VPT : 5 min a 20 Hz)
MAX_GUI_SHARE = 0.5    # episodes majoritairement "menu" -> jetes


@torch.no_grad()
def idm_label(frames_u8_nchw: np.ndarray, model, device) -> tuple[np.ndarray, np.ndarray]:
    """frames (N,3,64,64) -> actions (N-1,10), gui (N-1,). Fenetre t-2..t+2 -> action[t]."""
    n = frames_u8_nchw.shape[0]
    actions = np.zeros((n - 1, 10), dtype=np.float32)
    gui = np.zeros(n - 1, dtype=np.uint8)
    centers = np.arange(2, n - 2)
    for i in range(0, len(centers), 512):
        b = centers[i:i + 512]
        cols = [frames_u8_nchw[b + d] for d in (-2, -1, 0, 1, 2)]
        x = torch.from_numpy(np.concatenate(cols, axis=1)).to(device).float() / 255.0
        lb, cam, lg = model(x)
        btn = (torch.sigmoid(lb) > 0.5).float().cpu().numpy()
        actions[b, :8] = btn
        actions[b, 8:10] = np.clip(cam.cpu().numpy(), -1, 1)
        gui[b] = (torch.sigmoid(lg[:, 0]) > 0.5).cpu().numpy().astype(np.uint8)
    # bords : recopie du plus proche
    actions[:2] = actions[2]
    actions[-1] = actions[-2]
    gui[:2] = gui[2]
    gui[-1] = gui[-2]
    return actions, gui


def main():
    video = Path(sys.argv[1])
    prefix = sys.argv[2] if len(sys.argv) > 2 else video.stem
    max_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9   # plafond d'episodes
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = IDM().to(device)
    model.load_state_dict(torch.load(IDM_CKPT, weights_only=True))
    model.eval()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = src_fps / TARGET_FPS
    print(f"{video.name}: {n_src} frames @ {src_fps:.1f} fps -> 20 Hz (pas {step:.2f})")

    buf, ep_idx, next_pick, src_i = [], 0, 0.0, 0
    saved, dropped = 0, 0

    def flush(buf, ep_idx):
        nonlocal saved, dropped
        if len(buf) < 600:      # < 30 s : pas la peine
            return
        frames_nhwc = np.stack(buf)                       # (N,64,64,3) RGB
        nchw = np.ascontiguousarray(np.transpose(frames_nhwc, (0, 3, 1, 2)))
        actions, gui = idm_label(nchw, model, device)
        if gui.mean() > MAX_GUI_SHARE:
            dropped += 1
            return
        out = OUT_DIR / f"{prefix}_{ep_idx:03d}.npz"
        tmp = out.with_suffix(".npz.part")
        with open(tmp, "wb") as f:
            np.savez_compressed(f, frames=frames_nhwc, actions=actions, gui=gui)
        tmp.replace(out)
        saved += 1
        print(f"  [{out.name}] {len(buf)} frames | taper {actions[:,6].mean():.0%} "
              f"| cam |dx| {np.abs(actions[:,8]).mean():.2f} | gui {gui.mean():.0%}")

    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if src_i >= next_pick:
            small = cv2.resize(fr, (RES, RES), interpolation=cv2.INTER_AREA)
            buf.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            next_pick += step
            if len(buf) >= EP_LEN:
                flush(buf, ep_idx)
                buf, ep_idx = [], ep_idx + 1
                if ep_idx >= max_ep:
                    break
        src_i += 1
    flush(buf, ep_idx)
    cap.release()
    print(f"\nTermine : {saved} episodes sauves, {dropped} jetes (menus) "
          f"-> {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
