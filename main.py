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

from chan_vese import ChanVeseParams, chan_vese, multiphase_chan_vese
from utils import (
    read_grayscale_image,
    save_color_label_mask,
    save_energy_plot,
    save_label_mask,
    save_mask,
    save_multiphase_overlay,
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
        choices=["checkerboard", "circle", "intensity"],
        help="Initial level set method.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="two-phase",
        choices=["two-phase", "multi-phase"],
        help="Segmentation model. multi-phase uses two level sets and four regions.",
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
        "--solver",
        type=str,
        default="semi-implicit",
        choices=["semi-implicit", "explicit"],
        help="Numerical update scheme. semi-implicit follows the paper's finite-difference scheme.",
    )
    parser.add_argument(
        "--inner-iter",
        type=int,
        default=5,
        help="Fixed-point iterations used by the semi-implicit solver.",
    )
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
        solver=args.solver,
        inner_iter=args.inner_iter,
        reinit_every=args.reinit_every,
        reinit_iters=args.reinit_iters,
    )

    if args.model == "multi-phase":
        result = multiphase_chan_vese(
            image=image,
            params=params,
            init_level_set=args.init,
        )
        phi1, phi2 = result.phis if result.phis is not None else (result.phi, result.phi)
        save_label_mask(result.mask, output_dir / "label_mask.png")
        save_color_label_mask(result.mask, output_dir / "label_mask_color.png")
        save_phi(phi1, output_dir / "phi1.png")
        save_phi(phi2, output_dir / "phi2.png")
        save_multiphase_overlay(
            image,
            phi1,
            phi2,
            output_dir / "overlay.png",
            title=f"Multiphase Chan-Vese contours after {result.iterations} iterations",
        )
    else:
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
    print(f"Model: {args.model}")
    print(f"Iterations: {result.iterations}")
    if args.model == "multi-phase" and result.c_values is not None:
        print("Region averages:")
        for index, value in enumerate(result.c_values):
            print(f"  - c{index}: {value:.6f}")
    else:
        print(f"c1 inside average: {result.c1:.6f}")
        print(f"c2 outside average: {result.c2:.6f}")
    print("Saved files:")
    if args.model == "multi-phase":
        print(f"  - {output_dir / 'label_mask.png'}")
        print(f"  - {output_dir / 'label_mask_color.png'}")
        print(f"  - {output_dir / 'phi1.png'}")
        print(f"  - {output_dir / 'phi2.png'}")
    else:
        print(f"  - {output_dir / 'mask.png'}")
        print(f"  - {output_dir / 'phi.png'}")
    print(f"  - {output_dir / 'overlay.png'}")
    print(f"  - {output_dir / 'energy.png'}")


if __name__ == "__main__":
    main()
