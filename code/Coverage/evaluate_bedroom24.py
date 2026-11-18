from pathlib import Path
import csv
import math

import torch
import lpips
import numpy as np
from PIL import Image
from pytorch_msssim import ms_ssim


ROOT = Path(
    "outputs/bedroom_video_24views_sparsesurf_7000/test/ours_7000"
)

GT_DIR = ROOT / "gt"
RENDER_DIR = ROOT / "renders"

VIEWS = [
    "bedroom_002.png",
    "bedroom_014.png",
    "bedroom_022.png",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

lpips_model = lpips.LPIPS(net="alex").to(DEVICE)
lpips_model.eval()


def load_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0

    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    return x.to(DEVICE)


def calc_psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2).item()

    if mse == 0:
        return float("inf")

    return -10.0 * math.log10(mse)


rows = []

print()
print("============================================================")
print("SparseSurf Bedroom — 24 views — 7000 iterations")
print("============================================================")

for name in VIEWS:

    gt_path = GT_DIR / name
    render_path = RENDER_DIR / name

    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    if not render_path.exists():
        raise FileNotFoundError(render_path)

    gt = load_image(gt_path)
    pred = load_image(render_path)

    if gt.shape != pred.shape:
        raise RuntimeError(
            f"Shape mismatch for {name}: "
            f"GT={tuple(gt.shape)}, Render={tuple(pred.shape)}"
        )

    with torch.no_grad():

        psnr = calc_psnr(pred, gt)

        msssim = ms_ssim(
            pred,
            gt,
            data_range=1.0,
            size_average=True
        ).item()

        # LPIPS expects [-1, 1]
        lp = lpips_model(
            pred * 2.0 - 1.0,
            gt * 2.0 - 1.0
        ).item()

    rows.append({
        "view": name,
        "PSNR": psnr,
        "MS_SSIM": msssim,
        "LPIPS": lp,
    })

    print(
        f"{name:20s} "
        f"PSNR={psnr:8.4f}  "
        f"MS-SSIM={msssim:.6f}  "
        f"LPIPS={lp:.6f}"
    )


mean_psnr = np.mean([r["PSNR"] for r in rows])
mean_msssim = np.mean([r["MS_SSIM"] for r in rows])
mean_lpips = np.mean([r["LPIPS"] for r in rows])

print()
print("------------------------------------------------------------")
print(
    f"MEAN                 "
    f"PSNR={mean_psnr:8.4f}  "
    f"MS-SSIM={mean_msssim:.6f}  "
    f"LPIPS={mean_lpips:.6f}"
)
print("------------------------------------------------------------")


csv_path = ROOT / "bedroom24_metrics_7000.csv"

with open(csv_path, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "view",
            "PSNR",
            "MS_SSIM",
            "LPIPS",
        ],
    )

    writer.writeheader()

    for row in rows:
        writer.writerow(row)

    writer.writerow({
        "view": "MEAN",
        "PSNR": mean_psnr,
        "MS_SSIM": mean_msssim,
        "LPIPS": mean_lpips,
    })


print()
print("Saved:")
print(csv_path)
