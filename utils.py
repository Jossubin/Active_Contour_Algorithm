"""
utils.py

Input/output and visualization helpers for the Chan-Vese project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_grayscale_image(path: str | Path) -> np.ndarray:
    """
    Read an image file as grayscale float64 in [0, 1].
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image file: {path}")

    return image.astype(np.float64) / 255.0


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    """
    Save a binary mask image.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = (mask.astype(np.uint8) * 255)
    cv2.imwrite(str(path), out)


def save_phi(phi: np.ndarray, path: str | Path) -> None:
    """
    Save normalized phi as an image for inspection.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    phi_min = float(phi.min())
    phi_max = float(phi.max())

    if abs(phi_max - phi_min) < 1e-12:
        normalized = np.zeros_like(phi)
    else:
        normalized = (phi - phi_min) / (phi_max - phi_min)

    cv2.imwrite(str(path), (normalized * 255).astype(np.uint8))


def save_overlay(
    image: np.ndarray,
    phi: np.ndarray,
    path: str | Path,
    title: Optional[str] = None,
) -> None:
    """
    Save original image with the final zero-level contour overlaid.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))
    plt.imshow(image, cmap="gray")
    plt.contour(phi, levels=[0], linewidths=2)
    plt.axis("off")

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_energy_plot(energies: list[float], path: str | Path) -> None:
    """
    Save a plot of energy values over iterations.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot(energies)
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("Chan-Vese Energy")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def create_synthetic_image(size: int = 160) -> np.ndarray:
    """
    Create a simple synthetic test image with multiple objects.
    This allows the project to run even if the user has not prepared an image.
    """
    image = np.ones((size, size), dtype=np.float64) * 0.25

    yy, xx = np.indices(image.shape)

    circle = (xx - size * 0.35) ** 2 + (yy - size * 0.35) ** 2 < (size * 0.18) ** 2
    image[circle] = 0.75

    rectangle = (
        (xx > size * 0.58)
        & (xx < size * 0.85)
        & (yy > size * 0.55)
        & (yy < size * 0.80)
    )
    image[rectangle] = 0.65

    hole = (xx - size * 0.35) ** 2 + (yy - size * 0.35) ** 2 < (size * 0.07) ** 2
    image[hole] = 0.25

    rng = np.random.default_rng(7)
    image = image + rng.normal(0.0, 0.05, image.shape)

    return np.clip(image, 0.0, 1.0)


def save_synthetic_image(path: str | Path) -> None:
    """
    Save a generated synthetic test image to disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image = create_synthetic_image()
    cv2.imwrite(str(path), (image * 255).astype(np.uint8))
