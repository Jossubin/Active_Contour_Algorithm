"""
main.py

Command-line runner for the Chan-Vese active contour implementation.

Example:
    python main.py --input images/test.png --output results --max-iter 500

This saves outputs under a subdirectory named after the input image:
    results/test/

You can also name a run folder under results:
    python main.py --input images/test.png --output test_1000
    results/test_1000/

If --input is omitted, the program creates and uses a synthetic image:
    python main.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chan_vese import ChanVeseParams, chan_vese
from utils import (
    read_grayscale_image,
    save_energy_plot,
    save_mask,
    save_overlay,
    save_phi,
    save_synthetic_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chan-Vese Active Contours Without Edges segmentation"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Path to input image. If omitted, a synthetic test image is generated.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Result folder name under results. If omitted, the input image name is used.",
    )
    parser.add_argument(
        "--init",
        type=str,
        default="checkerboard",
        choices=["checkerboard", "circle"],
        help="Initial level set method.",
    )
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--mu", type=float, default=0.25)
    parser.add_argument("--nu", type=float, default=0.0)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--timestep", type=float, default=0.1)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument(
        "--reinit-every",
        type=int,
        default=0,
        help="Run approximate reinitialization every N iterations. 0 disables it.",
    )
    parser.add_argument("--reinit-iters", type=int, default=5)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input:
        input_path = Path(args.input)
    else:
        input_path = Path("images/synthetic_test.png")
        save_synthetic_image(input_path)
        print(f"No input image supplied. Generated synthetic image at: {input_path}")

    output_name = args.output if args.output else input_path.stem
    output_dir = Path("results") / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    image = read_grayscale_image(input_path)

    params = ChanVeseParams(
        mu=args.mu,
        nu=args.nu,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        epsilon=args.epsilon,
        timestep=args.timestep,
        max_iter=args.max_iter,
        tol=args.tol,
        reinit_every=args.reinit_every,
        reinit_iters=args.reinit_iters,
    )

    result = chan_vese(
        image=image,
        params=params,
        init_level_set=args.init,
    )

    save_mask(result.mask, output_dir / "mask.png")
    save_phi(result.phi, output_dir / "phi.png")
    save_overlay(
        image,
        result.phi,
        output_dir / "overlay.png",
        title=f"Chan-Vese contour after {result.iterations} iterations",
    )
    save_energy_plot(result.energies, output_dir / "energy.png")

    print("Chan-Vese segmentation finished.")
    print(f"Input image: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Iterations: {result.iterations}")
    print(f"c1 inside average: {result.c1:.6f}")
    print(f"c2 outside average: {result.c2:.6f}")
    print("Saved files:")
    print(f"  - {output_dir / 'mask.png'}")
    print(f"  - {output_dir / 'phi.png'}")
    print(f"  - {output_dir / 'overlay.png'}")
    print(f"  - {output_dir / 'energy.png'}")


if __name__ == "__main__":
    main()
