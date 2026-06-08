"""
chan_vese.py

A compact Python implementation of the Chan-Vese "Active Contours Without Edges"
model using a level-set formulation.

Paper:
Tony F. Chan and Luminita A. Vese,
"Active Contours Without Edges", IEEE Transactions on Image Processing, 2001.

This implementation follows the paper's main idea:
- Represent the evolving contour C as the zero level set of phi.
- Estimate c1 = average intensity inside C, c2 = average intensity outside C.
- Evolve phi by minimizing the Chan-Vese energy:
    F = mu * Length(C)
        + nu * Area(inside(C))
        + lambda1 * ∫inside(C) |u0 - c1|^2 dxdy
        + lambda2 * ∫outside(C) |u0 - c2|^2 dxdy

The update here uses an explicit finite-difference gradient descent scheme.
It is intentionally written to be readable for coursework submission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ChanVeseParams:
    """Parameters for the Chan-Vese active contour model."""

    mu: float = 0.25          # length regularization weight
    nu: float = 0.0           # area term weight
    lambda1: float = 1.0      # inside fitting term weight
    lambda2: float = 1.0      # outside fitting term weight
    epsilon: float = 1.0      # smoothing value for Heaviside/Dirac delta
    timestep: float = 0.1     # gradient descent time step
    max_iter: int = 500       # maximum number of iterations
    tol: float = 1e-4         # stopping threshold
    reinit_every: int = 0     # 0 disables reinitialization
    reinit_iters: int = 5     # iterations used when reinitializing


@dataclass
class ChanVeseResult:
    """Result container returned by chan_vese."""

    phi: np.ndarray
    mask: np.ndarray
    c1: float
    c2: float
    energies: List[float]
    iterations: int


def regularized_heaviside(phi: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Smooth approximation H_epsilon(phi).

    This is the arctangent approximation used in the Chan-Vese paper:
        H_epsilon(z) = 1/2 * (1 + (2/pi) * arctan(z / epsilon))
    """
    return 0.5 * (1.0 + (2.0 / np.pi) * np.arctan(phi / epsilon))


def regularized_delta(phi: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Smooth approximation delta_epsilon(phi), derivative of H_epsilon(phi).
    """
    return (epsilon / np.pi) / (epsilon * epsilon + phi * phi)


def initialize_phi(
    shape: Tuple[int, int],
    method: str = "checkerboard",
    radius: Optional[float] = None,
) -> np.ndarray:
    """
    Create an initial level-set function.

    Parameters
    ----------
    shape:
        Image shape as (height, width).
    method:
        "checkerboard" or "circle".
        - checkerboard: many initial contours spread across the image.
        - circle: one circular contour near the center.
    radius:
        Radius for circle initialization. If None, a default is used.

    Returns
    -------
    phi:
        Positive values are treated as inside the contour.
        Negative values are treated as outside the contour.
    """
    height, width = shape

    if method == "checkerboard":
        y, x = np.indices((height, width))
        block = max(8, min(height, width) // 8)
        phi = np.sin(np.pi * x / block) * np.sin(np.pi * y / block)
        return phi.astype(np.float64)

    if method == "circle":
        y, x = np.indices((height, width))
        cy = (height - 1) / 2.0
        cx = (width - 1) / 2.0
        if radius is None:
            radius = min(height, width) * 0.35

        # Positive inside the circle, negative outside.
        phi = radius - np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return phi.astype(np.float64)

    raise ValueError("method must be either 'checkerboard' or 'circle'")


def compute_region_averages(
    image: np.ndarray,
    phi: np.ndarray,
    epsilon: float,
) -> Tuple[float, float]:
    """
    Compute c1 and c2:
    c1 = average image value inside contour
    c2 = average image value outside contour
    """
    h = regularized_heaviside(phi, epsilon)

    inside_sum = np.sum(image * h)
    inside_area = np.sum(h)

    outside = 1.0 - h
    outside_sum = np.sum(image * outside)
    outside_area = np.sum(outside)

    tiny = 1e-12
    c1 = inside_sum / (inside_area + tiny)
    c2 = outside_sum / (outside_area + tiny)

    return float(c1), float(c2)


def curvature(phi: np.ndarray) -> np.ndarray:
    """
    Compute curvature div(grad(phi) / |grad(phi)|) using central differences.
    """
    phi_y, phi_x = np.gradient(phi)

    norm = np.sqrt(phi_x * phi_x + phi_y * phi_y + 1e-12)
    nx = phi_x / norm
    ny = phi_y / norm

    nxx = np.gradient(nx, axis=1)
    nyy = np.gradient(ny, axis=0)

    return nxx + nyy


def compute_energy(
    image: np.ndarray,
    phi: np.ndarray,
    params: ChanVeseParams,
    c1: float,
    c2: float,
) -> float:
    """
    Compute the regularized Chan-Vese energy value for monitoring convergence.
    """
    h = regularized_heaviside(phi, params.epsilon)
    delta = regularized_delta(phi, params.epsilon)

    phi_y, phi_x = np.gradient(phi)
    grad_norm = np.sqrt(phi_x * phi_x + phi_y * phi_y + 1e-12)

    length_term = params.mu * np.sum(delta * grad_norm)
    area_term = params.nu * np.sum(h)
    inside_term = params.lambda1 * np.sum(((image - c1) ** 2) * h)
    outside_term = params.lambda2 * np.sum(((image - c2) ** 2) * (1.0 - h))

    return float(length_term + area_term + inside_term + outside_term)


def reinitialize_sdf(phi: np.ndarray, iterations: int = 5, dt: float = 0.3) -> np.ndarray:
    """
    Optional approximate reinitialization of phi toward a signed-distance-like function.

    This is a simplified version of the Sussman-Osher reinitialization idea:
        psi_t = sign(phi) * (1 - |grad psi|)

    It helps avoid extremely flat or steep phi values, but excessive
    reinitialization may suppress interior contours. Therefore it is disabled
    by default.
    """
    psi = phi.copy()
    sign_phi = phi / np.sqrt(phi * phi + 1.0)

    for _ in range(iterations):
        psi_y, psi_x = np.gradient(psi)
        grad_norm = np.sqrt(psi_x * psi_x + psi_y * psi_y + 1e-12)
        psi = psi + dt * sign_phi * (1.0 - grad_norm)

    return psi


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to float64 in the range [0, 1].
    """
    image = image.astype(np.float64)

    if image.max() > 1.0:
        image = image / 255.0

    image = np.clip(image, 0.0, 1.0)
    return image


def chan_vese(
    image: np.ndarray,
    params: Optional[ChanVeseParams] = None,
    init_level_set: str = "checkerboard",
    init_phi: Optional[np.ndarray] = None,
) -> ChanVeseResult:
    """
    Run Chan-Vese segmentation.

    Parameters
    ----------
    image:
        2D grayscale image.
    params:
        ChanVeseParams object.
    init_level_set:
        Initial level set method if init_phi is not supplied.
    init_phi:
        Optional user-supplied initial phi.

    Returns
    -------
    ChanVeseResult
    """
    if params is None:
        params = ChanVeseParams()

    if image.ndim != 2:
        raise ValueError("chan_vese expects a 2D grayscale image.")

    image = normalize_image(image)

    if init_phi is None:
        phi = initialize_phi(image.shape, method=init_level_set)
    else:
        if init_phi.shape != image.shape:
            raise ValueError("init_phi must have the same shape as image.")
        phi = init_phi.astype(np.float64)

    energies: List[float] = []
    previous_phi = phi.copy()
    c1 = 0.0
    c2 = 0.0

    for iteration in range(1, params.max_iter + 1):
        c1, c2 = compute_region_averages(image, phi, params.epsilon)

        delta = regularized_delta(phi, params.epsilon)
        kappa = curvature(phi)

        # Gradient descent for the Euler-Lagrange equation.
        force = (
            params.mu * kappa
            - params.nu
            - params.lambda1 * (image - c1) ** 2
            + params.lambda2 * (image - c2) ** 2
        )

        phi = phi + params.timestep * delta * force

        # Neumann-like boundary condition by copying neighboring values.
        phi[0, :] = phi[1, :]
        phi[-1, :] = phi[-2, :]
        phi[:, 0] = phi[:, 1]
        phi[:, -1] = phi[:, -2]

        if params.reinit_every > 0 and iteration % params.reinit_every == 0:
            phi = reinitialize_sdf(phi, iterations=params.reinit_iters)

        energy = compute_energy(image, phi, params, c1, c2)
        energies.append(energy)

        change = np.mean(np.abs(phi - previous_phi))
        if change < params.tol:
            break

        previous_phi = phi.copy()

    mask = phi >= 0.0

    return ChanVeseResult(
        phi=phi,
        mask=mask,
        c1=c1,
        c2=c2,
        energies=energies,
        iterations=iteration,
    )
