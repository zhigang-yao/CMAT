"""Self-Action CMAT, YY-based comparison, and temporal pose correction.

Pass one uninterrupted observed trajectory [frames, joints, 3] to correct().
The result contains float64 positions in the input unit and solver diagnostics.
Only NumPy, SciPy and Python's standard library are required. No clean poses,
training data, learned weights, package installation, or source loader is used.

Configuration, local geometry, observation-only candidate selection, and the
joint trajectory solver are defined below in ordinary readable Python.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy import sparse
from scipy.linalg import cholesky_banded, cho_solve_banded, solveh_banded
from scipy.sparse.linalg import LinearOperator, cg
from scipy.spatial.distance import cdist

__all__ = [
    "CorrectionConfig", "CorrectionResult", "published_config", "correct",
    "VARIANTS", "REBUILD", "normalize_observations", "prepare", "recover_prepared",
]


# Published inference settings, separated from the numerical algorithm.

@dataclass(frozen=True)
class CorrectionConfig:
    """One fixed inference configuration; positions may use any length unit.

    Root indices specify a single root or the mean of multiple roots. The seed
    belongs to the sequence, not the method, so comparators share probes.
    """

    method: str = "cmat"
    fraction: float = 0.8
    eta: float = 30.0
    policy: str = "best"
    root_indices: tuple = (0,)
    yy_supports: tuple = (8, 16, 24, 32)

    def __post_init__(self):
        if self.method not in ("cmat", "yy", "temporal"):
            raise ValueError("method must be cmat, yy, or temporal")
        if not 0 <= self.fraction <= 1:
            raise ValueError("fraction must lie in [0,1]")
        if not 0 <= self.eta < float("inf"):
            raise ValueError("eta must be finite and nonnegative")
        if self.policy not in ("best", "mix"):
            raise ValueError("policy must be best or mix")
        if not self.root_indices or len(set(self.root_indices)) != len(self.root_indices):
            raise ValueError("root_indices must be nonempty and unique")
        if any(not isinstance(i, int) or i < 0 for i in self.root_indices):
            raise ValueError("root_indices must be nonnegative integers")
        if len(self.yy_supports) != 4 or any(
            not isinstance(k, int) or k < 1 for k in self.yy_supports
        ):
            raise ValueError("YY-based needs four positive candidate supports")


def published_config(method="cmat", dataset="amass"):
    """Return the recorded AMASS or 3DPW--EFT configuration.

    The paper gives the validation search space. Exact selected values and
    the comparator's support lists come from its frozen experiment records.
    """
    if dataset not in ("amass", "eft"):
        raise ValueError("dataset must be amass or eft")
    if dataset == "amass":
        return CorrectionConfig(method=method, eta=3.0 if method == "yy" else 30.0)
    return CorrectionConfig(
        method=method,
        eta=0.3,
        policy="mix",
        root_indices=(2, 3),
        yy_supports=(12, 20, 28, 32),
    )


# Explicit interventions used only for the component experiments.
#
# The full estimator is the default. These names describe the frozen controlled
# comparisons in Supplement D.2.1; they are not alternative tuned estimators.

VARIANTS = {
    "full": "Full Self-Action CMAT",
    "no_guide": "No preliminary guide",
    "no_cylinder": "No cylinder boundary",
    "uniform_attention": "Reliability-only weighting",
    "no_reliability": "No reliability weighting",
    "no_balance": "No balance head",
    "no_gls": "No GLS head",
    "no_balance_gls": "No balance or GLS heads",
    "normal_only": "Rank-three normal projection only",
    "full_displacement_only": "Full-center constraint only",
    "no_cap": "No displacement coefficient",
    "no_null": "No null option",
    "no_risk_margin": "No score margin",
    "equal_head_mix": "Equal candidate mixture",
    "no_final_temporal": "No final temporal penalty",
    "no_final_robust_data": "Quadratic final observation loss",
    "no_overlap_normalization": "No overlap normalization",
    "temporal_neighbors": "Temporal neighbors",
    "pose_only_neighbors": "Position-only matching",
    "time_only": "Temporal-only control",
}

# These variants change the guide or local proposals, so the positive/negative
# perturbation passes must repeat the intervention too. The remaining variants
# reuse the full method's candidate states and perturbation scores.
REBUILD = frozenset(
    {
        "no_guide",
        "no_cylinder",
        "uniform_attention",
        "no_reliability",
        "no_cap",
        "temporal_neighbors",
        "pose_only_neighbors",
    }
)


def validate(intervention, method):
    if intervention not in VARIANTS:
        raise ValueError(f"Unknown component intervention: {intervention}")
    if intervention != "full" and method != "cmat":
        raise ValueError("Component interventions apply to CMAT, not its comparators")


# Root/scale normalization and the observation-only temporal guide.

def variation_scales(y):
    """Per-joint scale from third differences, with the paper's positive floor."""
    return np.maximum(
        np.median(np.linalg.norm(np.diff(y, n=3, axis=0), axis=2), axis=0)
        / (1.538 * np.sqrt(20)),
        2e-5,
    )


def trend_solve(y, scale, alpha=12, max_iter=30):
    n = len(y)
    d = sparse.diags(
        [-np.ones(n - 3), 3 * np.ones(n - 3), -3 * np.ones(n - 3), np.ones(n - 3)],
        [0, 1, 2, 3],
        shape=(n - 3, n),
        format="csr",
    )
    z = y.copy()
    for iteration in range(max_iter):
        old = z
        rr = np.linalg.norm(z - y, axis=1)
        w = np.maximum(0.02, np.minimum(1, 2.8 * scale / np.maximum(rr, 1e-12)))
        v = 1 / np.sqrt(np.sum((d @ z) ** 2, axis=1) + (0.05 * scale) ** 2)
        h = (sparse.diags(w) + alpha * scale * (d.T @ sparse.diags(v) @ d)).tocsc()
        band = np.zeros((4, n))
        for k in range(4):
            band[k, : n - k] = h.diagonal(-k)
        z = solveh_banded(band, w[:, None] * y, lower=True, check_finite=False)
        if np.linalg.norm(z - old) / max(np.linalg.norm(old), scale * np.sqrt(n)) < 1e-05:
            break
    return z


# Fixed-fraction working covariance and probe draws (Supplement C.1.4).
#
# This describes the model used for candidate comparison. It does not estimate
# a calibrated true pose-error distribution or fit an AR fraction on test data.

TAU = 15.0
FILTER = np.array([-1.0, 3.0, -3.0, 1.0])


def difference_variance(fraction, rho):
    """Variance of the third-difference filter under unit position variance."""
    variance = 0.0
    for a, ca in enumerate(FILTER):
        for b, cb in enumerate(FILTER):
            distance = abs(a - b)
            covariance = (1 - fraction) * float(distance == 0) + fraction * rho**distance
            variance += ca * cb * covariance
    return variance


def estimate_covariance(y, fraction, root_indices):
    """Robust spatial covariance with the validation-selected temporal fraction."""
    frames, joints, _ = y.shape
    dimension = joints * 3
    difference = np.diff(y.reshape(frames, dimension), n=3, axis=0)
    centered = difference - np.median(difference, axis=0)
    robust_scale = np.median(np.abs(centered), axis=0) / 0.6744897501960817
    rms = np.sqrt(np.mean(centered * centered, axis=0))
    robust_scale = np.where(robust_scale > 0, robust_scale, rms)
    clipped = np.clip(centered, -4 * robust_scale, 4 * robust_scale)
    clipped -= clipped.mean(axis=0, keepdims=True)
    sample = clipped.T @ clipped / max(len(clipped) - 1, 1)
    covariance = 0.5 * sample + 0.5 * np.diag(np.diag(sample))
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    covariance = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    covariance = 0.5 * (covariance + covariance.T)
    rho = float(np.exp(-1 / TAU))
    covariance /= difference_variance(fraction, rho)
    center = np.eye(joints)
    center[:, list(root_indices)] -= 1 / len(root_indices)
    projection = np.kron(center, np.eye(3))
    covariance = projection @ covariance @ projection.T
    return dict(
        sigma_spatial=covariance,
        fraction=fraction,
        rho=rho,
        dimensions=dimension,
        joints=joints,
    )


def retained_variance(y, guide, scales, model):
    """Approximate variance retained by the robust temporal guide."""
    delta = np.diff(guide, n=3, axis=0)
    temporal_weights = 1 / np.sqrt(
        np.sum(delta * delta, axis=2) + (0.05 * scales[None]) ** 2
    )
    observation_weights = np.maximum(
        0.02,
        np.minimum(
            1, 2.8 * scales[None] / np.maximum(np.linalg.norm(guide - y, axis=2), 1e-12)
        ),
    )
    strength = 12 * scales * np.median(temporal_weights, axis=0)
    strength /= np.maximum(observation_weights.mean(axis=0), 1e-12)
    omega = np.linspace(0, np.pi, 1025)
    transfer = 1 / (1 + strength[:, None] * (2 * np.sin(omega[None] / 2)) ** 6)
    fraction, rho = model["fraction"], model["rho"]
    spectrum = (1 - fraction) + fraction * (1 - rho * rho) / (
        1 + rho * rho - 2 * rho * np.cos(omega)
    )
    retention = np.trapezoid(transfer**2 * spectrum[None], omega, axis=1) / np.pi
    raw = np.diag(model["sigma_spatial"]).reshape(-1, 3).mean(axis=1)
    return float(np.mean(raw * retention)), retention


def probes(model, frames, seed, count=2):
    """Draw full-sequence probes with a stationary AR(1) initialization.

    Draw order is retained from the frozen experiment so the same integer
    seed reproduces its Gaussian draws. The caller shares this seed between
    methods; all candidates receive the same probes.
    """
    covariance = model["sigma_spatial"]
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))
    rng = np.random.default_rng(seed)
    shape = (count, frames, model["dimensions"])
    independent = rng.normal(size=shape)
    fraction, rho = model["fraction"], model["rho"]
    if fraction > 0:
        innovations = rng.normal(size=shape)
        autoregressive = np.empty_like(innovations)
        autoregressive[:, 0] = innovations[:, 0]
        for t in range(1, frames):
            autoregressive[:, t] = (
                rho * autoregressive[:, t - 1] + np.sqrt(1 - rho * rho) * innovations[:, t]
            )
        standard = np.sqrt(1 - fraction) * independent + np.sqrt(fraction) * autoregressive
    else:
        standard = independent
    return (standard @ factor.T).reshape(count, frames, model["joints"], 3)


# Matched nine-frame windows within 120-frame blocks (Supplement C.1.2).

def neighbors_xyz(windows, mode="full"):
    n = len(windows)
    if mode == "temporal_neighbors":
        indices = np.arange(n)
        return [np.argsort(abs(indices - i), kind="stable")[1:33] for i in indices]
    features = (
        windows.reshape(n, -1)
        if mode == "pose_only_neighbors"
        else np.concatenate(
            [
                windows.reshape(n, -1),
                0.7 * np.diff(windows, axis=1).reshape(n, -1),
                0.175 * np.diff(windows, n=2, axis=1).reshape(n, -1),
            ],
            axis=1,
        )
    )
    median = np.median(features, axis=0)
    scale = np.maximum(np.median(np.abs(features - median), axis=0), 0.001)
    standardized = (features - median) / scale
    nearest = np.argsort(cdist(standardized, standardized, "sqeuclidean"), axis=1)
    result = []
    for i, candidates in enumerate(nearest):
        chosen = [int(j) for j in candidates if abs(int(j) - i) > 9][:16]
        local = np.flatnonzero(
            (np.abs(np.arange(n) - i) > 0) & (np.abs(np.arange(n) - i) <= 18)
        )
        local = local[np.argsort(np.abs(local - i))]
        for j in list(local) + list(candidates):
            if j != i and j not in chosen:
                chosen.append(int(j))
            if len(chosen) >= 32:
                break
        result.append(np.asarray(chosen[:32]))
    return result


# Preliminary weighted plane and finite cylinder in query-relative coordinates.

def preliminary_fit(x, rank=3):
    """Fit the common pilot geometry; rows of the basis are orthonormal.

    ``sigma`` is a dimensionless residual scale. It is unrelated to the
    temporal/spatial working covariance used to score correction candidates.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < rank + 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite local differences with at least rank+2 rows")
    count, dimension = x.shape
    squared_distance = np.einsum("kd,kd->k", x, x)
    scale_squared = max(float(np.median(squared_distance)), 1e-16)
    weights = np.exp(-np.minimum(squared_distance / scale_squared, 700))
    weights /= weights.sum()
    mean = weights @ x
    centered = x - mean
    _, singular_values, basis = np.linalg.svd(
        centered * np.sqrt(weights[:, None]),
        full_matrices=False,
    )
    basis = basis[:rank]
    normal = centered - (centered @ basis.T) @ basis
    residual_variance = np.sum(weights * np.sum(normal * normal, axis=1))
    scale = np.sqrt(scale_squared)
    sigma = np.clip(
        np.sqrt(residual_variance / max(dimension - rank, 1)) / scale,
        1e-5,
        0.95,
    )
    return dict(
        mean=mean,
        basis=basis,
        sigma=float(sigma),
        scale_squared=scale_squared,
        singular_values=singular_values,
    )


def cylinder_support(x, fit, minimum=24):
    """One cylinder, using joint radius/half-length expansion in Supplement C.2.

    The historical implementation expanded first to 11 and then to 24 points.
    For ordered cylindrical distances q_11 <= q_24, its total multiplier was
    max(1, 1.1*q_11, 1.1*q_24) = max(1, 1.1*q_24), used directly here.
    """
    count = len(x)
    scale = np.sqrt(max(np.median(np.sum(x * x, axis=1)), 1e-16))
    distance = np.linalg.norm(x, axis=1)
    sigma = fit["sigma"]
    radius = 5 * sigma / np.log10(max(count, 2)) * scale
    half_length = 10 * sigma * np.sqrt(np.log(1 / sigma)) / np.log10(max(count, 2)) * scale
    fifth = np.partition(distance, min(4, count - 1))[min(4, count - 1)]
    pilot_radius = max(radius, 1.1 * fifth, 1e-12)
    pilot_weights = np.maximum(1 - (distance / pilot_radius) ** 2, 0) ** 3
    pilot_weights /= max(pilot_weights.sum(), 1e-300)
    pilot_mean = pilot_weights @ x
    length = np.linalg.norm(pilot_mean)
    axis = pilot_mean / length if length > 1e-12 * scale else fit["basis"][0]
    axial = x @ axis
    radial = np.sqrt(np.maximum(np.sum(x * x, axis=1) - axial**2, 0))
    ratios = np.maximum(
        radial / max(radius, 1e-15), np.abs(axial) / max(half_length, 1e-15)
    )
    order = min(minimum, count) - 1
    expansion = max(1.0, 1.1 * np.partition(ratios, order)[order])
    return ratios < expansion


def weighted_basis(x, weights):
    """Refit the center and three directions under one attention rule."""
    mean = weights @ x
    _, _, basis = np.linalg.svd(
        (x - mean) * np.sqrt(weights[:, None]),
        full_matrices=False,
    )
    return mean, basis[:3]


# Entropy-balanced tangent weights (Supplement C.1.3).
#
# The query remains the origin. The prior's second moment, rather than a
# centered variance, rescales tangent coordinates. Newton's method solves the
# strictly convex three-dimensional dual; no fitting or training data are used.

SPEC = {"standardization": "prior-weighted second-moment RMS, no centering"}


def _state(beta, coordinates, log_prior, kappa):
    logits = log_prior - coordinates @ beta
    maximum = float(logits.max())
    unnormalized = np.exp(logits - maximum)
    mass = float(unnormalized.sum())
    posterior = unnormalized / mass
    log_partition = maximum + np.log(mass)
    mean = posterior @ coordinates
    centered = coordinates - mean
    gradient = beta / kappa - mean
    hessian = centered.T @ (posterior[:, None] * centered) + np.eye(len(beta)) / kappa
    objective = float(log_partition + 0.5 * (beta @ beta) / kappa)
    return (objective, posterior, mean, gradient, hessian, logits - log_partition)


def balanced_weights(p, u):
    """Return balanced weights[K], JSON-serializable numerical diagnostics.

    u is standardized internally using p. p may be unnormalized; zero prior
    entries always produce exactly zero returned entries. u outside that mask
    is ignored. The fixed preliminary basis is supplied indirectly through u.
    """
    p = np.asarray(p, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if p.ndim != 1 or u.ndim != 2 or len(p) != len(u) or (not len(p)) or (not u.shape[1]):
        raise ValueError("Expected p[K] and u[K,R] with K,R positive")
    if not np.isfinite(p).all() or np.any(p < 0) or (not np.any(p > 0)):
        raise ValueError("Prior attention must be finite, nonnegative, and nonempty")
    active = p > 0
    if not np.isfinite(u[active]).all():
        raise ValueError("Active tangent coordinates must be finite")
    prior = p[active] / p[active].sum()
    coordinates = u[active]
    rms = np.sqrt(prior @ (coordinates * coordinates))
    floor = max(float(rms.max(initial=0.0)) * 1e-08, 1e-14)
    scales = np.maximum(rms, floor)
    standardized = coordinates / scales
    kappa = float(1 / (prior @ prior))
    log_prior = np.log(prior)
    beta = np.zeros(u.shape[1])
    evaluations = 0
    tolerance = 1e-09
    converged = False
    for iteration in range(100):
        objective, posterior, mean, gradient, hessian, log_posterior = _state(
            beta, standardized, log_prior, kappa
        )
        evaluations += 1
        residual = float(np.linalg.norm(gradient, ord=np.inf))
        if residual <= tolerance * (1 + np.linalg.norm(mean, ord=np.inf)):
            converged = True
            break
        direction = -np.linalg.solve(hessian, gradient)
        slope = float(gradient @ direction)
        if not np.isfinite(direction).all() or slope >= 0:
            raise ArithmeticError(
                "Tangent-balance Newton direction is not a descent direction"
            )
        step = 1.0
        accepted = False
        for _ in range(40):
            trial = _state(beta + step * direction, standardized, log_prior, kappa)
            evaluations += 1
            armijo = trial[0] <= objective + 0.0001 * step * slope
            rounding_progress = (
                trial[0] <= objective + 1e-14 * (1 + abs(objective))
                and np.linalg.norm(trial[3], ord=np.inf) < residual
            )
            if armijo or rounding_progress:
                beta += step * direction
                accepted = True
                break
            step *= 0.5
        if not accepted:
            raise ArithmeticError("Tangent-balance dual line search failed")
    if not converged:
        raise ArithmeticError("Tangent-balance dual did not converge")
    result = np.zeros_like(p)
    result[active] = posterior
    prior_mean = prior @ standardized
    kl = float(posterior @ (log_posterior - log_prior))
    primal = kl + 0.5 * kappa * float(mean @ mean)
    dual_gap = float(primal + objective)
    if np.any(result < 0) or abs(result.sum() - 1) > 1e-12 or dual_gap < -1e-10:
        raise ArithmeticError("Invalid tangent-balance primal solution")
    diagnostics = dict(
        support=int(active.sum()),
        positive_weights=int(np.count_nonzero(result)),
        prior_ess=kappa,
        ess=float(1 / (result @ result)),
        kappa=kappa,
        moment_norm_before=float(np.linalg.norm(prior_mean)),
        moment_norm_after=float(np.linalg.norm(mean)),
        original_moment_norm_before=float(np.linalg.norm(prior @ coordinates)),
        original_moment_norm_after=float(np.linalg.norm(posterior @ coordinates)),
        rms_scales=scales.tolist(),
        moment_before=prior_mean.tolist(),
        moment_after=mean.tolist(),
        beta=beta.tolist(),
        kkt_residual=residual,
        dual_iterations=iteration + 1,
        dual_evaluations=evaluations,
        primal_objective=primal,
        dual_objective=objective,
        primal_dual_gap=dual_gap,
        kl_to_prior=kl,
        prior_objective=float(0.5 * kappa * (prior_mean @ prior_mean)),
        converged=True,
        standardization=SPEC["standardization"],
    )
    return (result, diagnostics)


# Nonnegative GLS weights for the specified working correlation.
#
# The active-set quadratic solver enforces the simplex constraints. Clipping an
# unconstrained GLS solution is not equivalent and is intentionally not used.

def _equality_solution(root_p, correlation, free):
    """A_SS^-1 1 via well-conditioned R_SS rather than explicit inverse p."""
    indices = np.flatnonzero(free)
    value = root_p[indices] * np.linalg.solve(
        correlation[np.ix_(indices, indices)], root_p[indices]
    )
    denominator = value.sum()
    if not np.isfinite(denominator) or denominator <= 0:
        raise ArithmeticError("Nonpositive GLS normalization for SPD covariance")
    result = np.zeros_like(root_p)
    result[indices] = value / denominator
    return result


def _simplex_qp(p, correlation, tolerance=1e-10):
    """Solve min w'Aw on the simplex; return exact-constraint KKT diagnostics."""
    root_p = np.sqrt(p)
    free = np.ones(len(p), dtype=bool)
    current = p.copy()
    limit = max(100, 10 * len(p) ** 2)
    for iteration in range(limit):
        candidate = _equality_solution(root_p, correlation, free)
        negative = free & (candidate < 0.0)
        if np.any(negative):
            ratios = current[negative] / (current[negative] - candidate[negative])
            step = min(1.0, max(0.0, float(ratios.min())))
            current += step * (candidate - current)
            negative_ids = np.flatnonzero(negative)
            limiting = negative_ids[ratios <= step + 1e-13]
            current[limiting] = 0.0
            free[limiting] = False
            current = np.maximum(current, 0.0)
            current /= current.sum()
            continue
        current = candidate
        gradient = correlation @ (current / root_p) / root_p
        multiplier = float(current @ gradient)
        inactive = np.flatnonzero(~free)
        allowed = tolerance * max(1.0, abs(multiplier))
        if len(inactive) and gradient[inactive].min() < multiplier - allowed:
            free[inactive[np.argmin(gradient[inactive])]] = True
            continue
        free_residual = float(np.max(np.abs(gradient[free] - multiplier), initial=0.0))
        inactive_violation = float(np.max(multiplier - gradient[~free], initial=0.0))
        residual = max(free_residual, inactive_violation) / max(1.0, abs(multiplier))
        if residual > 20 * tolerance or current.min(initial=0.0) < -1e-12:
            raise ArithmeticError(f"GLS QP KKT residual too large: {residual}")
        return (
            current,
            dict(
                qp_iterations=iteration + 1,
                qp_kkt_relative_residual=residual,
                qp_active_weights=int(np.count_nonzero(current > 1e-12)),
                qp_objective=multiplier,
                qp_converged=True,
            ),
        )
    raise ArithmeticError("GLS simplex active-set algorithm exceeded its iteration limit")


# Uniform cylinder weights for the YY-based comparator (Supplement F).
#
# This is the paper's finite-sample comparator, not an implementation of the
# original Yao--Yau smooth kernel-product estimator. Candidate support sizes
# are supplied explicitly by the AMASS or EFT configuration.

def _axis(first, x, z):
    norm = np.linalg.norm(first)
    scale = max(float(np.median(np.linalg.norm(x, axis=1))), 1e-15)
    if norm > 1e-12 * scale:
        return (first / norm, False)
    basis = np.asarray(z.get("basis", []), dtype=float)
    if basis.ndim == 2 and basis.shape[1] == x.shape[1] and len(basis):
        candidate = basis[0]
    else:
        candidate = x[np.argmax(np.linalg.norm(x, axis=1))]
    length = np.linalg.norm(candidate)
    if length <= 1e-15:
        candidate = np.zeros(x.shape[1])
        candidate[0] = 1.0
        length = 1.0
    return (candidate / length, True)


def yao_support_geometry(x, z, config):
    """Return weights and diagnostic geometry; x has shape [neighbors,features]."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) == 0 or (not np.isfinite(x).all()):
        raise ValueError("Expected finite nonempty [neighbors,features] residuals")
    count = len(x)
    requested = int(config.get("k_support", config.get("support", 24)))
    if requested < 1:
        raise ValueError("Requested cylinder support must be positive")
    minimum = min(requested, count)
    aspect = float(config.get("aspect", 1.0))
    if not np.isfinite(aspect) or aspect <= 0:
        raise ValueError("Cylinder aspect multiplier must be positive")
    sigma = float(np.clip(float(z["sigma"]), 1e-05, 0.95))
    if not np.isfinite(sigma):
        raise ValueError("Nonfinite sigma")
    distance = np.linalg.norm(x, axis=1)
    scale = max(float(np.sqrt(np.median(distance * distance))), 1e-15)
    denominator = np.log10(max(count, 2))
    radius0 = max(5 * sigma / denominator * scale, 1e-15)
    half0 = max(
        aspect * 10 * sigma * np.sqrt(np.log(1 / sigma)) / denominator * scale, 1e-15
    )
    pilot_minimum = min(count, max(5, (minimum + 1) // 2))
    pilot_order = np.partition(distance, pilot_minimum - 1)[pilot_minimum - 1]
    sphere_radius = max(2 * radius0, float(np.nextafter(pilot_order, np.inf)))
    sphere = distance <= sphere_radius
    first = x[sphere].mean(axis=0)
    axis, degenerate = _axis(first, x, z)
    axial = x @ axis
    radial = np.sqrt(np.maximum(np.sum(x * x, axis=1) - axial * axial, 0.0))
    ratio = np.maximum(radial / radius0, np.abs(axial) / half0)
    kth = np.partition(ratio, minimum - 1)[minimum - 1]
    enlargement = max(1.0, float(np.nextafter(kth, np.inf)))
    inside = ratio <= enlargement
    if inside.sum() < minimum:
        raise ArithmeticError("Finite cylinder failed its minimum-support guarantee")
    weights = inside.astype(np.float64) / int(inside.sum())
    return dict(
        weights=weights,
        inside=inside,
        axis=axis,
        radius=float(radius0 * enlargement),
        half_length=float(half0 * enlargement),
        initial_radius=float(radius0),
        initial_half_length=float(half0),
        enlargement=enlargement,
        support=int(inside.sum()),
        requested_support=requested,
        minimum_support=minimum,
        pilot_support=int(sphere.sum()),
        pilot_radius=float(sphere_radius),
        axis_degenerate=degenerate,
        aspect=aspect,
    )


def yao_support_weights(x, z, config):
    """Return normalized uniform finite-cylinder weights, with no attention."""
    return yao_support_geometry(x, z, config)["weights"]


# The four correction attention rules from Supplement C.1.3.

def supported_softmax(scores, mask, reliability):
    """Normalize distance scores after applying the hard cylindrical support."""
    scores = np.asarray(scores).copy()
    scores[~mask] = -np.inf
    scores -= scores.max()
    weights = np.exp(scores) * reliability
    weights /= weights.sum()
    return weights


def weight_vectors(x, fit, mask, reliability, offsets, config):
    """Return four vectors, ordered as distance2, distance8, balance, GLS."""
    if config.method == "yy":
        return [
            yao_support_weights(x, fit, dict(support=k, aspect=1.0))
            for k in config.yy_supports
        ]
    tangent = x @ fit["basis"].T
    tangent_squared = np.sum(tangent * tangent, axis=1)
    floor = max(np.median(np.sum(x * x, axis=1)), 1e-16) * 1e-8
    tangent_squared /= max(np.median(tangent_squared), floor)
    narrow = supported_softmax(-tangent_squared / 2, mask, reliability)
    broad = supported_softmax(-tangent_squared / 8, mask, reliability)
    balanced, _ = balanced_weights(broad, tangent)
    active = broad > 0
    prior = np.maximum(broad[active], 1e-14)
    prior /= prior.sum()
    delta = np.abs(offsets[active, None] - offsets[None, active])
    correlation = (1 - config.fraction) * np.eye(len(prior))
    correlation += config.fraction * np.exp(-delta / 15.0)
    gls = np.zeros_like(broad)
    gls[active], _ = _simplex_qp(prior, correlation)
    return [narrow, broad, balanced, gls]


# Construct the sixteen local constraints used by one correction method.

# A zero leaves a fitted direction free; a one constrains that direction.
# Ordering matches the published comparison: retain 1, 2, 3, or 0 directions.
SPECTRA = np.asarray([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])


def guide_from_y(y):
    scales = variation_scales(y)
    guide = np.stack([trend_solve(y[:, j], scales[j]) for j in range(y.shape[1])], axis=1)
    return guide, scales


def build(y, config, keep_planes=True, intervention="full"):
    """Recompute guide, neighbors, weights and coefficients from observations.

    The same routine is used on positive/negative perturbed observations.
    Planes may be omitted for those probes because only their local candidate
    displacements enter the finite-difference comparison.
    """
    if intervention == "no_guide":
        guide, scales = y.copy(), variation_scales(y)
    else:
        guide, scales = guide_from_y(y)
    model = estimate_covariance(y, config.fraction, config.root_indices)
    if intervention == "no_guide":
        variance = float(np.diag(model["sigma_spatial"]).reshape(-1, 3).mean(axis=1).mean())
        retention = np.ones(len(scales))
    else:
        variance, retention = retained_variance(y, guide, scales, model)
    reliability = np.maximum(
        0.02,
        np.minimum(
            1, 2.8 * scales[None] / np.maximum(np.linalg.norm(guide - y, axis=2), 1e-12)
        ),
    )
    frames, joints, _ = y.shape
    starts = list(range(0, max(1, frames - 120 + 1), 60))
    if frames > 120 and starts[-1] != frames - 120:
        starts.append(frames - 120)
    indices = []
    records = []
    for start in starts:
        stop = min(start + 120, frames)
        window_indices = np.arange(stop - start - 8)[:, None] + np.arange(9)[None]
        windows = guide[start:stop][window_indices]
        neighbors = neighbors_xyz(windows, mode=intervention)
        quality = reliability[start:stop][window_indices].mean(axis=(1, 2))
        if intervention == "no_reliability":
            quality = np.ones_like(quality)
        queries = list(range(0, len(windows), 4))
        if queries[-1] != len(windows) - 1:
            queries.append(len(windows) - 1)
        for query in queries:
            matched = neighbors[query]
            x = (windows[matched] - windows[query]).reshape(len(matched), -1)
            fit = preliminary_fit(x)
            mask = cylinder_support(x, fit)
            if intervention == "no_cylinder":
                mask = np.ones(len(x), dtype=bool)
            if intervention == "uniform_attention":
                weight = mask.astype(float) * quality[matched]
                weight /= weight.sum()
                weights = [weight.copy() for _ in range(4)]
            else:
                weights = weight_vectors(
                    x, fit, mask, quality[matched], matched - query, config
                )
            indices.append(start + window_indices[query])
            displacements, caps, centers, bases = [], [], [], []
            for weight in weights:
                mean, basis = weighted_basis(x, weight)
                for spectrum in SPECTRA:
                    displacement = mean - ((mean @ basis.T) * (1 - spectrum)) @ basis
                    energy = float(np.mean(displacement**2))
                    cap = variance / (variance + energy) if variance + energy > 0 else 0.0
                    if intervention == "no_cap":
                        cap = 1.0
                    displacements.append(displacement)
                    caps.append(cap)
                    if keep_planes:
                        centers.append(windows[query].reshape(-1) + mean)
                        bases.append(basis)
            records.append((displacements, caps, centers, bases))
    indices = np.asarray(indices)
    coverage = np.bincount(indices.ravel(), minlength=frames)
    if np.any(coverage == 0):
        raise ValueError("Geometry windows do not cover every frame")
    value = dict(
        delta=np.asarray([r[0] for r in records]).transpose(1, 0, 2),
        cap=np.asarray([r[1] for r in records]).T,
    )
    if keep_planes:
        value.update(
            target=np.asarray([r[2] for r in records]).transpose(1, 0, 2),
            basis=np.asarray([r[3] for r in records]).transpose(1, 0, 2, 3),
        )
    return dict(
        guide=guide,
        sd=scales,
        model=model,
        indices=indices,
        gamma=np.min(1 / coverage[indices], axis=1),
        variance=variance,
        retention=retention,
        candidates=value,
    )


def correction(value, eta):
    """Candidate displacement after the common variance-based shrinkage."""
    cap = value["cap"]
    gain = eta * cap / (1 + eta * cap)
    return gain[:, :, None] * value["delta"]


def geometry(state, probabilities, overlap_normalization=True):
    """Convert fixed selected candidates to the joint solver's constraints."""
    value = state["candidates"]
    heads, groups = probabilities.shape
    gamma = state["gamma"] if overlap_normalization else np.ones_like(state["gamma"])
    weights = probabilities * value["cap"] * gamma[None]
    keep = weights.ravel() > 0
    if not keep.any():
        return None
    width = state["indices"].shape[1]
    joints = state["guide"].shape[1]
    return dict(
        indices=np.tile(state["indices"], (heads, 1))[keep],
        basis=value["basis"].reshape(heads * groups, 3, -1)[keep],
        target=value["target"].reshape(heads * groups, width, joints, 3)[keep],
        tangent_weights=np.repeat(np.tile(SPECTRA, (4, 1)), groups, axis=0)[keep],
        weight=weights.ravel()[keep],
        type="spectral",
    )


# Observation-only local candidate comparison (main Eqs. 2.19--2.20).

EPSILON = 0.1
PROBES = 2


def prepare(y, config, seed, intervention="full"):
    """Score local proposals, not the final nonlinear joint solution."""
    state = build(y, config, intervention=intervention)
    indices = state["indices"]
    dimension = np.prod(y.shape[1:]) * indices.shape[1]
    residual = (state["guide"][indices] - y[indices]).reshape(len(indices), -1)
    noises = probes(state["model"], len(y), seed, count=PROBES)
    delta = correction(state["candidates"], config.eta)
    traces = []
    for noise in noises:
        positive = build(
            y + EPSILON * noise, config, keep_planes=False, intervention=intervention
        )
        negative = build(
            y - EPSILON * noise, config, keep_planes=False, intervention=intervention
        )
        if not np.array_equal(indices, positive["indices"]) or not np.array_equal(
            indices, negative["indices"]
        ):
            raise ValueError("Perturbations changed query indices")
        probe = noise[indices].reshape(len(indices), -1)
        difference = correction(positive["candidates"], config.eta) - correction(
            negative["candidates"], config.eta
        )
        traces.append(np.einsum("gd,hgd->hg", probe, difference) / (EPSILON * dimension))
    loss = np.mean(delta**2 + 2 * delta * residual[None], axis=2)
    return state, loss[None] + np.asarray(traces)


def probabilities(state, trials, policy, intervention="full"):
    """Keep the best negative score, or mix up to four negative scores.

    The mean-plus-Monte-Carlo-standard-error score measures probe variability.
    It is not an uncertainty interval for reconstruction error. Null blocks
    contribute no geometric constraint; the joint temporal solve still runs.
    """
    _, heads, groups = trials.shape
    if intervention == "equal_head_mix":
        return np.full((heads, groups), 1 / heads), dict(active_block_fraction=1.0)
    allowed = np.ones(heads, dtype=bool)
    if intervention in ("no_balance", "no_balance_gls"):
        allowed[8:12] = False
    if intervention in ("no_gls", "no_balance_gls"):
        allowed[12:16] = False
    if intervention == "normal_only":
        allowed = np.arange(heads) % 4 == 2
    if intervention == "full_displacement_only":
        allowed = np.arange(heads) % 4 == 3
    if not allowed.all():
        trials = trials.copy()
        trials[:, ~allowed] = 1e30
    output = np.zeros((heads, groups))
    active, winners, minima = [], [], []
    blocks = state["indices"][:, 4] // 60
    for block in np.unique(blocks):
        selected = blocks == block
        values = trials[:, :, selected].mean(axis=2)
        mean = values.mean(axis=0)
        standard_error = values.std(axis=0, ddof=1) / np.sqrt(len(values))
        margin = 0.0 if intervention == "no_risk_margin" else 1.0
        score = mean + margin * standard_error
        order = np.argsort(score)
        if intervention == "no_null":
            eligible = np.array([int(np.argmin(score))])
        else:
            eligible = order[score[order] < 0]
        if policy == "best":
            eligible = eligible[:1]
        elif policy == "mix":
            eligible = eligible[:4]
        else:
            raise ValueError("Unknown candidate-selection policy")
        if len(eligible):
            strength = np.maximum(-score[eligible], 1e-30)
            weights = strength / strength.sum()
            output[np.ix_(eligible, np.flatnonzero(selected))] = weights[:, None]
        active.append(bool(len(eligible)))
        winners.append(eligible.tolist())
        minima.append(float(score.min()))
    diagnostic = dict(
        active_block_fraction=float(np.mean(active)),
        selected_heads=winners,
        minimum_working_risk=minima,
        mean_cap=float(state["candidates"]["cap"].mean()),
        probes=PROBES,
        epsilon=EPSILON,
        policy=policy,
    )
    return output, diagnostic


# Fixed-geometry trajectory recovery (main Eq. 2.21, Supplement C.1.6).
#
# This is the common final solver for CMAT, YY-based, and temporal correction.
# It starts from the observed sequence, reweights observation/third-difference
# residuals, and solves each resulting linear system by preconditioned CG.

ALPHA = 12.0
MAX_ITER = 30


def _difference(n):
    return sparse.diags(
        [-np.ones(n - 3), 3 * np.ones(n - 3), -3 * np.ones(n - 3), np.ones(n - 3)],
        [0, 1, 2, 3],
        shape=(n - 3, n),
        format="csr",
    )


def _weights(y, z, sd, d, robust_observation=True):
    residual = np.linalg.norm(z - y, axis=2)
    w = np.maximum(0.02, np.minimum(1.0, 2.8 * sd[None] / np.maximum(residual, 1e-12)))
    if not robust_observation:
        w = np.ones(y.shape[:2])
    delta = (d @ z.reshape(len(z), -1)).reshape(len(z) - 3, len(sd), 3)
    v = 1.0 / np.sqrt(np.sum(delta * delta, axis=2) + (0.05 * sd[None]) ** 2)
    return (w, v)


def _temporal_system(w, v, sd, d, temporal_strength=ALPHA):
    """Return exact temporal Hessian and its per-joint band Cholesky inverse."""
    n, joints = w.shape
    bands = []
    for j in range(joints):
        h = (
            sparse.diags(w[:, j])
            + temporal_strength * sd[j] * (d.T @ sparse.diags(v[:, j]) @ d)
        ).tocsc()
        band = np.zeros((4, n))
        for k in range(4):
            band[k, : n - k] = h.diagonal(-k)
        bands.append(cholesky_banded(band, lower=True, check_finite=False))

    def multiply(z):
        dz = (d @ z.reshape(n, -1)).reshape(n - 3, joints, 3)
        diff = d.T @ (v[:, :, None] * dz).reshape(n - 3, -1)
        return w[:, :, None] * z + temporal_strength * sd[None, :, None] * diff.reshape(
            n, joints, 3
        )

    def inverse(value):
        output = np.empty_like(value)
        for j, band in enumerate(bands):
            output[:, j] = cho_solve_banded((band, True), value[:, j], check_finite=False)
        return output

    return (multiply, inverse)


class _Geometry:

    def __init__(self, data, shape):
        self.shape = shape
        self.indices = np.asarray(data["indices"], dtype=np.int64)
        self.weight = np.asarray(data["weight"], dtype=np.float64)
        self.target = np.asarray(data["target"], dtype=np.float64)
        self.kind = data.get("type", "spectral")
        g, width = self.indices.shape
        if self.kind != "spectral":
            raise ValueError("Unknown geometry type")
        if self.weight.shape != (g,) or np.any(self.weight < 0):
            raise ValueError("Geometry weights must be nonnegative and shape [G]")
        if self.target.shape != (g, width, shape[1], 3):
            raise ValueError("Target must have shape [G,window,J,3]")
        if np.any(self.indices < 0) or np.any(self.indices >= shape[0]):
            raise ValueError("Geometry indices are outside the sequence")
        self.flat_target = self.target.reshape(g, -1)
        self.basis = np.asarray(data["basis"], dtype=np.float64)
        if self.basis.ndim != 3 or self.basis.shape[0] != g:
            raise ValueError("Basis must have shape [G,rank,window*J*3]")
        if self.basis.shape[2] != self.flat_target.shape[1]:
            raise ValueError("Basis and target dimensions differ")
        gram = self.basis @ self.basis.transpose(0, 2, 1)
        identity = np.eye(self.basis.shape[1])[None]
        if np.max(np.abs(gram - identity), initial=0.0) > 1e-05:
            raise ValueError("Each geometric tangent basis must be orthonormal")
        self.tangent_weights = np.asarray(data["tangent_weights"], dtype=np.float64)
        if self.tangent_weights.shape != self.basis.shape[:2]:
            raise ValueError("Spectral tangent_weights must have shape [G,rank]")
        if (
            not np.isfinite(self.tangent_weights).all()
            or np.any(self.tangent_weights < 0)
            or np.any(self.tangent_weights > 1)
        ):
            raise ValueError("Spectral tangent_weights must be finite and in [0,1]")
        values = [self.weight, self.target, self.basis]
        if not all((np.isfinite(value).all() for value in values)):
            raise ValueError("Geometry contains nonfinite values")

    def project(self, flat):
        coefficient = np.einsum("grd,gd->gr", self.basis, flat, optimize=True)
        coefficient = (1 - self.tangent_weights) * coefficient
        return flat - np.einsum("grd,gr->gd", self.basis, coefficient, optimize=True)

    def scatter(self, flat):
        output = np.zeros(self.shape)
        weighted = (self.weight[:, None] * flat).reshape(self.target.shape)
        np.add.at(output, self.indices, weighted)
        return output

    def multiply(self, z):
        flat = z[self.indices].reshape(self.flat_target.shape)
        return self.scatter(self.project(flat))

    def rhs(self):
        return self.scatter(self.project(self.flat_target))

    def penalty(self, z):
        flat = z[self.indices].reshape(self.flat_target.shape) - self.flat_target
        residual = self.project(flat)
        return 0.5 * float(np.sum(self.weight[:, None] * flat * residual))


def _objective(
    y, z, sd, d, geometry, eta, temporal_strength=ALPHA, robust_observation=True
):
    """The exact loss whose residual-weight derivative is the floored Huber w."""
    r = np.linalg.norm(z - y, axis=2)
    threshold = 2.8 * sd[None]
    high = threshold / 0.02
    observation = np.where(
        r <= threshold,
        0.5 * r * r,
        np.where(
            r <= high,
            threshold * r - 0.5 * threshold**2,
            0.01 * r * r + 0.5 * threshold**2 / 0.02 - 0.5 * threshold**2,
        ),
    )
    delta = (d @ z.reshape(len(z), -1)).reshape(len(z) - 3, len(sd), 3)
    if not robust_observation:
        observation = 0.5 * np.sum((z - y) ** 2, axis=2)
    temporal = temporal_strength * np.sum(
        sd[None] * np.sqrt(np.sum(delta * delta, axis=2) + (0.05 * sd[None]) ** 2)
    )
    penalty = 0.0 if geometry is None else eta * geometry.penalty(z)
    return float(observation.sum() + temporal + penalty)


def solve_joint(
    y,
    sd,
    geometry,
    eta,
    initial=None,
    max_iter=MAX_ITER,
    cg_rtol=1e-05,
    cg_maxiter=800,
    temporal_strength=ALPHA,
    robust_observation=True,
):
    """Jointly minimize observation, temporal, and fixed local-geometry losses.

    No ground truth is read. eta=0 uses the same joint iterations/stopping rule
    with exact band solves. All method comparisons use the same initial=y.
    initial is only a solver warm start, never an added data target.
    CG failures are returned explicitly; reject unreliable configurations upstream.
    """
    y = np.asarray(y, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)
    if y.ndim != 3 or y.shape[2] != 3 or len(y) < 4:
        raise ValueError("Expected finite observations [T>=4,J,3]")
    if sd.shape != (y.shape[1],) or np.any(sd <= 0):
        raise ValueError("Noise scales must be positive with shape [J]")
    if not np.isfinite(y).all() or not np.isfinite(sd).all():
        raise ValueError("Observations or noise scales contain nonfinite values")
    if not np.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and nonnegative")
    if not np.isfinite(temporal_strength) or temporal_strength < 0:
        raise ValueError("temporal_strength must be finite and nonnegative")
    if geometry is None and eta > 0:
        raise ValueError("Positive eta requires geometry")
    geo = _Geometry(geometry, y.shape) if eta > 0 else None
    z = y.copy() if initial is None else np.array(initial, dtype=np.float64, copy=True)
    if z.shape != y.shape or not np.isfinite(z).all():
        raise ValueError("Initial trajectory has wrong shape or nonfinite values")
    d = _difference(len(y))
    fixed_rhs = eta * geo.rhs() if geo is not None else np.zeros_like(y)
    cg_iterations, failures, residuals, changes = ([], [], [], [])
    objectives = [_objective(y, z, sd, d, geo, eta, temporal_strength, robust_observation)]
    converged = False
    shape, size = (y.shape, y.size)
    for iteration in range(max_iter):
        old = z
        w, v = _weights(y, z, sd, d, robust_observation)
        temporal, inverse = _temporal_system(w, v, sd, d, temporal_strength)

        def multiply(flat):
            value = flat.reshape(shape)
            geometric = 0.0 if geo is None else eta * geo.multiply(value)
            return (temporal(value) + geometric).ravel()

        operator = LinearOperator((size, size), matvec=multiply, dtype=np.float64)
        preconditioner = LinearOperator(
            (size, size),
            matvec=lambda flat: inverse(flat.reshape(shape)).ravel(),
            dtype=np.float64,
        )
        rhs = (w[:, :, None] * y + fixed_rhs).ravel()
        count = [0]

        def callback(_):
            count[0] += 1

        if geo is None:
            flat, info = (inverse(rhs.reshape(shape)).ravel(), 0)
        else:
            flat, info = cg(
                operator,
                rhs,
                x0=z.ravel(),
                M=preconditioner,
                rtol=cg_rtol,
                atol=0.0,
                maxiter=cg_maxiter,
                callback=callback,
            )
        if not np.isfinite(flat).all():
            raise FloatingPointError("Nonfinite CG output")
        relative_residual = float(
            np.linalg.norm(multiply(flat) - rhs) / max(np.linalg.norm(rhs), 1e-30)
        )
        if info != 0 or relative_residual > max(2 * cg_rtol, 1e-10):
            failures.append(
                dict(
                    irls_iteration=iteration + 1,
                    info=int(info),
                    relative_residual=relative_residual,
                )
            )
        cg_iterations.append(count[0])
        residuals.append(relative_residual)
        z = flat.reshape(shape)
        joint_change = np.linalg.norm(z - old, axis=(0, 2)) / np.maximum(
            np.linalg.norm(old, axis=(0, 2)), sd * np.sqrt(len(y))
        )
        change = float(joint_change.max())
        changes.append(change)
        objectives.append(
            _objective(y, z, sd, d, geo, eta, temporal_strength, robust_observation)
        )
        if change < 1e-05 and info == 0:
            converged = True
            break
    objective_increases = int(
        np.sum(np.diff(objectives) > 1e-06 * np.maximum(np.abs(objectives[:-1]), 1e-12))
    )
    diagnostic = dict(
        mode="joint_fixed_geometry_irls",
        eta=float(eta),
        geometry_type="none" if geo is None else geo.kind,
        geometric_windows=0 if geo is None else len(geo.weight),
        iterations=len(changes),
        converged=converged,
        relative_changes=changes,
        cg_iterations=cg_iterations,
        cg_failures=failures,
        max_linear_relative_residual=max(residuals, default=0.0),
        objectives=objectives,
        objective_increases=objective_increases,
        alpha=temporal_strength,
        cg_rtol=cg_rtol,
        cg_maxiter=cg_maxiter,
    )
    return (z, diagnostic)


# Public correction interface: no clean poses, labels, or learned weights.

@dataclass
class CorrectionResult:
    """Corrected positions in input units and compact solver diagnostics."""

    positions: np.ndarray
    diagnostics: dict


def normalize_observations(observed, config):
    """Validate and retain the observed root and body scale for restoration."""
    observations = np.asarray(observed, dtype=np.float64)
    if (
        observations.ndim != 3
        or observations.shape[2] != 3
        or min(observations.shape[:2]) == 0
    ):
        raise ValueError("Expected nonempty observations [frames,joints,3]")
    if not np.isfinite(observations).all():
        raise ValueError("Observations must be finite")
    if max(config.root_indices) >= observations.shape[1]:
        raise ValueError("Root index is outside the input joint layout")
    root = observations[:, list(config.root_indices)].mean(axis=1, keepdims=True)
    centered = observations - root
    scale = max(
        float(np.median(np.sqrt(np.mean(np.sum(centered**2, axis=2), axis=1)))), 1e-8
    )
    return observations, centered / scale, root, scale


def recover_prepared(y, root, scale, config, seed, state, trials, intervention="full"):
    """Apply an explicit intervention to prepared states, then solve jointly.

    This entry point lets the experiment driver reuse the full method's guide,
    candidates and perturbations for downstream interventions. It never changes
    a module global or mutates the supplied full-method state.
    """
    validate(intervention, config.method)
    if state is None or intervention == "time_only":
        scales = variation_scales(y)
        constraints = None
        diagnostic = dict(selection=None)
    else:
        scales = state["sd"]
        probability, selection = probabilities(state, trials, config.policy, intervention)
        constraints = geometry(
            state, probability, intervention != "no_overlap_normalization"
        )
        diagnostic = dict(selection=selection, variance_after_guide=state["variance"])
    strength = config.eta if constraints is not None else 0.0
    output, solver = solve_joint(
        y,
        scales,
        constraints,
        strength,
        temporal_strength=0.0 if intervention == "no_final_temporal" else 12.0,
        robust_observation=intervention != "no_final_robust_data",
    )
    if solver["cg_failures"]:
        raise ArithmeticError(
            f"Joint solver failed its linear residual check: {solver['cg_failures']}"
        )
    diagnostic.update(
        config=asdict(config),
        seed=int(seed),
        body_scale=scale,
        solver=solver,
        short_identity=False,
        intervention=intervention,
    )
    return CorrectionResult(output * scale + root, diagnostic)


def correct(observed, config=None, seed=0, intervention="full"):
    """Correct one uninterrupted sequence of positions ``[frames,joints,3]``.

    Use the same integer seed for CMAT and YY-based on the same sequence to
    share perturbations. Root centering and one median body scale are computed
    only from observations. Missing/nonfinite data need an explicit upstream
    policy and are rejected. Sequences shorter than 16 frames pass through,
    preserving the frozen experiments' short-sequence policy.

    Returns float64 positions in the original length unit. Frozen experiment
    artifacts cast these positions to float32 when saving predictions.
    ``intervention`` is reserved for the frozen component experiments in
    Supplement D.2.1. Its default retains every operation of the full method.
    """
    config = CorrectionConfig() if config is None else config
    validate(intervention, config.method)
    observations, y, root, scale = normalize_observations(observed, config)
    if len(observations) < 16:
        return CorrectionResult(
            observations.copy(), dict(short_identity=True, config=asdict(config))
        )
    if config.method == "temporal" or config.eta == 0 or intervention == "time_only":
        state, trials = None, None
    else:
        state, trials = prepare(y, config, seed, intervention)
    return recover_prepared(y, root, scale, config, seed, state, trials, intervention)
