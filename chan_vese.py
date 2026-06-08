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

The default update uses a semi-implicit finite-difference approximation of
the Euler-Lagrange PDE, following the numerical scheme described in Section
III of Chan and Vese (2001). An explicit update is kept for comparison.
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
    solver: str = "semi-implicit"  # "semi-implicit" follows the paper's finite-difference scheme
    inner_iter: int = 5       # fixed-point iterations for the semi-implicit linearized update
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
    phis: Optional[Tuple[np.ndarray, np.ndarray]] = None
    c_values: Optional[List[float]] = None


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
    image: Optional[np.ndarray] = None,
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

    if method == "intensity":
        if image is None:
            raise ValueError("intensity initialization requires an image.")
        return (image - np.median(image)).astype(np.float64)

    raise ValueError("method must be 'checkerboard', 'circle', or 'intensity'")


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


def apply_neumann_boundary(phi: np.ndarray) -> None:
    """
    Apply the zero normal derivative boundary condition from the paper.
    """
    phi[0, :] = phi[1, :]
    phi[-1, :] = phi[-2, :]
    phi[:, 0] = phi[:, 1]
    phi[:, -1] = phi[:, -2]


def explicit_update(
    image: np.ndarray,
    phi: np.ndarray,
    params: ChanVeseParams,
    c1: float,
    c2: float,
) -> np.ndarray:
    """
    Readable explicit gradient-descent update for comparison.
    """
    delta = regularized_delta(phi, params.epsilon)
    kappa = curvature(phi)

    force = (
        params.mu * kappa
        - params.nu
        - params.lambda1 * (image - c1) ** 2
        + params.lambda2 * (image - c2) ** 2
    )

    next_phi = phi + params.timestep * delta * force
    apply_neumann_boundary(next_phi)
    return next_phi


def semi_implicit_update(
    image: np.ndarray,
    phi: np.ndarray,
    params: ChanVeseParams,
    c1: float,
    c2: float,
) -> np.ndarray:
    """
    Semi-implicit finite-difference update for equation (9) in the paper.

    The nonlinear curvature term
        div(grad(phi) / |grad(phi)|)
    is discretized with coefficients computed from phi^n, while the neighbor
    values of phi are iterated toward phi^(n+1). This is the practical form of
    the paper's linearized implicit curvature update.
    """
    old_phi = phi.copy()
    next_phi = phi.copy()
    delta = regularized_delta(old_phi, params.epsilon)
    tiny = 1e-8

    center = old_phi[1:-1, 1:-1]
    east = old_phi[1:-1, 2:]
    west = old_phi[1:-1, :-2]
    south = old_phi[2:, 1:-1]
    north = old_phi[:-2, 1:-1]

    c_e = 1.0 / np.sqrt((east - center) ** 2 + ((south - north) * 0.5) ** 2 + tiny)
    c_w = 1.0 / np.sqrt(
        (center - west) ** 2
        + ((old_phi[2:, :-2] - old_phi[:-2, :-2]) * 0.5) ** 2
        + tiny
    )
    c_s = 1.0 / np.sqrt(((east - west) * 0.5) ** 2 + (south - center) ** 2 + tiny)
    c_n = 1.0 / np.sqrt(
        ((old_phi[:-2, 2:] - old_phi[:-2, :-2]) * 0.5) ** 2
        + (center - north) ** 2
        + tiny
    )

    delta_i = delta[1:-1, 1:-1]
    data_force = (
        -params.nu
        - params.lambda1 * (image[1:-1, 1:-1] - c1) ** 2
        + params.lambda2 * (image[1:-1, 1:-1] - c2) ** 2
    )

    curvature_weight = params.timestep * delta_i * params.mu
    rhs = center + params.timestep * delta_i * data_force
    denominator = 1.0 + curvature_weight * (c_e + c_w + c_s + c_n)

    for _ in range(max(1, params.inner_iter)):
        next_phi[1:-1, 1:-1] = (
            rhs
            + curvature_weight
            * (
                c_e * next_phi[1:-1, 2:]
                + c_w * next_phi[1:-1, :-2]
                + c_s * next_phi[2:, 1:-1]
                + c_n * next_phi[:-2, 1:-1]
            )
        ) / denominator
        apply_neumann_boundary(next_phi)

    return next_phi


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


def initialize_multiphase_phi(
    image: np.ndarray,
    method: str = "checkerboard",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize two level-set functions for a four-region multiphase model.
    """
    shape = image.shape
    height, width = shape
    y, x = np.indices((height, width))

    if method == "checkerboard":
        block = max(8, min(height, width) // 8)
        phi1 = np.sin(np.pi * x / block) * np.sin(np.pi * y / block)
        phi2 = np.sin(np.pi * (x + block * 0.5) / block) * np.sin(
            np.pi * (y + block * 0.25) / block
        )
        return phi1.astype(np.float64), phi2.astype(np.float64)

    if method == "circle":
        cy = (height - 1) / 2.0
        cx = (width - 1) / 2.0
        radius1 = min(height, width) * 0.42
        radius2 = min(height, width) * 0.24
        phi1 = radius1 - np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        phi2 = radius2 - np.sqrt((x - cx - width * 0.12) ** 2 + (y - cy) ** 2)
        return phi1.astype(np.float64), phi2.astype(np.float64)

    if method == "intensity":
        q25, q50, q75 = np.quantile(image, [0.25, 0.5, 0.75])
        phi1 = image - q50
        phi2 = (image - q25) * (image - q75)
        return phi1.astype(np.float64), phi2.astype(np.float64)

    raise ValueError("method must be 'checkerboard', 'circle', or 'intensity'")


def compute_multiphase_memberships(
    phi1: np.ndarray,
    phi2: np.ndarray,
    epsilon: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the four soft region memberships defined by two level sets.
    """
    h1 = regularized_heaviside(phi1, epsilon)
    h2 = regularized_heaviside(phi2, epsilon)
    m11 = h1 * h2
    m10 = h1 * (1.0 - h2)
    m01 = (1.0 - h1) * h2
    m00 = (1.0 - h1) * (1.0 - h2)
    return m11, m10, m01, m00


def compute_multiphase_averages(
    image: np.ndarray,
    phi1: np.ndarray,
    phi2: np.ndarray,
    epsilon: float,
) -> List[float]:
    """
    Compute one average intensity for each of the four regions.
    """
    memberships = compute_multiphase_memberships(phi1, phi2, epsilon)
    tiny = 1e-12
    return [
        float(np.sum(image * membership) / (np.sum(membership) + tiny))
        for membership in memberships
    ]


def multiphase_labels(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    """
    Convert two level sets into hard labels 0, 1, 2, and 3.
    """
    bit1 = phi1 >= 0.0
    bit2 = phi2 >= 0.0
    return (bit1.astype(np.uint8) * 2 + bit2.astype(np.uint8))


def compute_multiphase_energy(
    image: np.ndarray,
    phi1: np.ndarray,
    phi2: np.ndarray,
    params: ChanVeseParams,
    c_values: List[float],
) -> float:
    """
    Compute the four-region Chan-Vese multiphase energy.
    """
    memberships = compute_multiphase_memberships(phi1, phi2, params.epsilon)
    fitting_weight = 0.5 * (params.lambda1 + params.lambda2)
    fitting_term = sum(
        fitting_weight * np.sum(((image - c) ** 2) * membership)
        for c, membership in zip(c_values, memberships)
    )

    length_term = 0.0
    for phi in (phi1, phi2):
        delta = regularized_delta(phi, params.epsilon)
        phi_y, phi_x = np.gradient(phi)
        grad_norm = np.sqrt(phi_x * phi_x + phi_y * phi_y + 1e-12)
        length_term += params.mu * np.sum(delta * grad_norm)

    return float(length_term + fitting_term)


def multiphase_explicit_update(
    image: np.ndarray,
    phi1: np.ndarray,
    phi2: np.ndarray,
    params: ChanVeseParams,
    c_values: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Explicit gradient descent update for the two-level-set, four-region model.
    """
    c11, c10, c01, c00 = c_values
    h1 = regularized_heaviside(phi1, params.epsilon)
    h2 = regularized_heaviside(phi2, params.epsilon)
    d1 = regularized_delta(phi1, params.epsilon)
    d2 = regularized_delta(phi2, params.epsilon)
    fitting_weight = 0.5 * (params.lambda1 + params.lambda2)

    e11 = fitting_weight * (image - c11) ** 2
    e10 = fitting_weight * (image - c10) ** 2
    e01 = fitting_weight * (image - c01) ** 2
    e00 = fitting_weight * (image - c00) ** 2

    data_force1 = -(h2 * (e11 - e01) + (1.0 - h2) * (e10 - e00))
    data_force2 = -(h1 * (e11 - e10) + (1.0 - h1) * (e01 - e00))

    next_phi1 = phi1 + params.timestep * d1 * (
        params.mu * curvature(phi1) + data_force1
    )
    next_phi2 = phi2 + params.timestep * d2 * (
        params.mu * curvature(phi2) + data_force2
    )

    apply_neumann_boundary(next_phi1)
    apply_neumann_boundary(next_phi2)
    return next_phi1, next_phi2


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

    if params.solver not in {"semi-implicit", "explicit"}:
        raise ValueError("solver must be either 'semi-implicit' or 'explicit'.")

    if image.ndim != 2:
        raise ValueError("chan_vese expects a 2D grayscale image.")

    image = normalize_image(image)

    if init_phi is None:
        phi = initialize_phi(image.shape, method=init_level_set, image=image)
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

        if params.solver == "semi-implicit":
            phi = semi_implicit_update(image, phi, params, c1, c2)
        else:
            phi = explicit_update(image, phi, params, c1, c2)

        if params.reinit_every > 0 and iteration % params.reinit_every == 0:
            phi = reinitialize_sdf(phi, iterations=params.reinit_iters)
            apply_neumann_boundary(phi)

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


def multiphase_chan_vese(
    image: np.ndarray,
    params: Optional[ChanVeseParams] = None,
    init_level_set: str = "checkerboard",
    init_phis: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> ChanVeseResult:
    """
    Run a two-level-set, four-region multiphase Chan-Vese segmentation.

    This is useful when the image is not well described by only one inside
    average and one outside average. Two level-set functions split the image
    into four regions:
        H(phi1)H(phi2), H(phi1)(1-H(phi2)),
        (1-H(phi1))H(phi2), and (1-H(phi1))(1-H(phi2)).
    """
    if params is None:
        params = ChanVeseParams()

    if image.ndim != 2:
        raise ValueError("multiphase_chan_vese expects a 2D grayscale image.")

    image = normalize_image(image)

    if init_phis is None:
        phi1, phi2 = initialize_multiphase_phi(image, method=init_level_set)
    else:
        phi1, phi2 = init_phis
        if phi1.shape != image.shape or phi2.shape != image.shape:
            raise ValueError("init_phis must have the same shape as image.")
        phi1 = phi1.astype(np.float64)
        phi2 = phi2.astype(np.float64)

    energies: List[float] = []
    previous_phi1 = phi1.copy()
    previous_phi2 = phi2.copy()
    c_values = [0.0, 0.0, 0.0, 0.0]

    for iteration in range(1, params.max_iter + 1):
        c_values = compute_multiphase_averages(
            image,
            phi1,
            phi2,
            params.epsilon,
        )

        phi1, phi2 = multiphase_explicit_update(
            image,
            phi1,
            phi2,
            params,
            c_values,
        )

        if params.reinit_every > 0 and iteration % params.reinit_every == 0:
            phi1 = reinitialize_sdf(phi1, iterations=params.reinit_iters)
            phi2 = reinitialize_sdf(phi2, iterations=params.reinit_iters)
            apply_neumann_boundary(phi1)
            apply_neumann_boundary(phi2)

        energy = compute_multiphase_energy(image, phi1, phi2, params, c_values)
        energies.append(energy)

        change = 0.5 * (
            np.mean(np.abs(phi1 - previous_phi1))
            + np.mean(np.abs(phi2 - previous_phi2))
        )
        if change < params.tol:
            break

        previous_phi1 = phi1.copy()
        previous_phi2 = phi2.copy()

    labels = multiphase_labels(phi1, phi2)

    return ChanVeseResult(
        phi=phi1,
        mask=labels,
        c1=c_values[0],
        c2=c_values[1],
        energies=energies,
        iterations=iteration,
        phis=(phi1, phi2),
        c_values=c_values,
    )
