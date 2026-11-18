import os
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args


def save_split(name, views, gaussians, pipeline, background, out_root):
    split_dir = os.path.join(out_root, name)
    os.makedirs(split_dir, exist_ok=True)

    rows = [
        "view\tH\tW\talpha_min\talpha_median\talpha_mean\talpha_max"
        "\tdepth_valid_pct\tdepth_median\tdepth_min_valid\tdepth_max_valid"
    ]

    for view in tqdm(views, desc=f"Raw {name} render"):
        pkg = render(view, gaussians, pipeline, background)

        required = ["render", "rendered_alpha", "plane_depth"]
        missing = [k for k in required if k not in pkg]
        if missing:
            raise RuntimeError(
                f"{view.image_name}: renderer missing required outputs: {missing}"
            )

        rgb = (
            pkg["render"]
            .detach()
            .float()
            .cpu()
            .permute(1, 2, 0)
            .contiguous()
            .numpy()
            .astype(np.float32)
        )

        alpha = (
            pkg["rendered_alpha"]
            .detach()
            .float()
            .cpu()
            .squeeze()
            .numpy()
            .astype(np.float32)
        )

        depth = (
            pkg["plane_depth"]
            .detach()
            .float()
            .cpu()
            .squeeze()
            .numpy()
            .astype(np.float32)
        )

        if alpha.ndim != 2:
            raise RuntimeError(
                f"{view.image_name}: unexpected alpha shape {alpha.shape}"
            )

        if depth.ndim != 2:
            raise RuntimeError(
                f"{view.image_name}: unexpected depth shape {depth.shape}"
            )

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise RuntimeError(
                f"{view.image_name}: unexpected RGB shape {rgb.shape}"
            )

        if alpha.shape != depth.shape or rgb.shape[:2] != depth.shape:
            raise RuntimeError(
                f"{view.image_name}: shape mismatch "
                f"rgb={rgb.shape}, alpha={alpha.shape}, depth={depth.shape}"
            )

        stem = view.image_name

        np.save(
            os.path.join(split_dir, stem + "_rgb.npy"),
            rgb,
            allow_pickle=False,
        )

        np.save(
            os.path.join(split_dir, stem + "_alpha.npy"),
            alpha,
            allow_pickle=False,
        )

        np.save(
            os.path.join(split_dir, stem + "_depth.npy"),
            depth,
            allow_pickle=False,
        )

        valid_depth = np.isfinite(depth) & (depth > 0)

        if np.any(valid_depth):
            dv = depth[valid_depth]
            depth_pct = 100.0 * float(valid_depth.mean())
            depth_med = float(np.median(dv))
            depth_min = float(dv.min())
            depth_max = float(dv.max())
        else:
            depth_pct = 0.0
            depth_med = np.nan
            depth_min = np.nan
            depth_max = np.nan

        av = alpha[np.isfinite(alpha)]

        rows.append(
            f"{stem}\t"
            f"{depth.shape[0]}\t{depth.shape[1]}\t"
            f"{float(av.min()):.8f}\t"
            f"{float(np.median(av)):.8f}\t"
            f"{float(av.mean()):.8f}\t"
            f"{float(av.max()):.8f}\t"
            f"{depth_pct:.4f}\t"
            f"{depth_med:.8f}\t"
            f"{depth_min:.8f}\t"
            f"{depth_max:.8f}"
        )

        print(
            f"{stem}: "
            f"alpha median={np.median(av):.4f}, "
            f"mean={av.mean():.4f}, "
            f"valid depth={depth_pct:.2f}%"
        )

    with open(os.path.join(split_dir, "stats.tsv"), "w") as f:
        f.write("\n".join(rows) + "\n")


def run(dataset, pipeline, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)

        scene = Scene(
            dataset,
            gaussians,
            load_iteration=args.iteration,
            shuffle=False,
        )

        print("Loaded iteration:", scene.loaded_iter)
        print("Gaussian count:", gaussians.get_xyz.shape[0])

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]

        background = torch.tensor(
            bg_color,
            dtype=torch.float32,
            device="cuda",
        )

        os.makedirs(args.raw_out, exist_ok=True)

        if not args.skip_train:
            save_split(
                "train",
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                args.raw_out,
            )

        if not args.skip_test:
            save_split(
                "test",
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                args.raw_out,
            )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Export renderer-native RGB, alpha and depth"
    )

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument(
        "--raw_out",
        required=True,
        type=str,
    )

    args = get_combined_args(parser)

    print("Model:", args.model_path)
    print("Raw output:", args.raw_out)

    safe_state(args.quiet)

    run(
        model.extract(args),
        pipeline.extract(args),
        args,
    )
