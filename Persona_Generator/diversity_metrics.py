"""
Diversity Metrics

Implements Section 3.5 and Appendix C of the paper.

Six metrics to quantify how well a population of response embeddings Z
covers the diversity space spanned by D.

Maximize:
    1. Coverage (Monte Carlo estimate)
    2. Convex Hull Volume
    3. Minimum Pairwise Distance
    4. Average (Mean) Pairwise Distance

Minimize:
    5. Dispersion (radius of largest empty region)
    6. KL Divergence from ideal quasi-random distribution

All metrics are computed over Z ∈ R^{N×K} where N is population size
and K is the number of diversity axes.

The paper normalizes embeddings to [0,1]^K before computing metrics,
mapping Likert scores from [1,5] to [0,1].
"""

import numpy as np
from typing import Dict, Optional
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform


def normalize_embeddings(Z: np.ndarray, score_min: float = 1.0, score_max: float = 5.0) -> np.ndarray:
    """
    Normalize response embeddings from Likert scale [1,5] to [0,1]^K.

    Args:
        Z: Array of shape (N, K) with values in [score_min, score_max]
        score_min: Minimum possible score (1 for 5-point Likert)
        score_max: Maximum possible score (5 for 5-point Likert)

    Returns:
        Normalized array with values in [0, 1]
    """
    return np.clip((Z - score_min) / (score_max - score_min), 0.0, 1.0)


def _generate_sobol_reference(num_points: int, num_dimensions: int, seed: int = 0) -> np.ndarray:
    """Generate Sobol quasi-random reference points in [0,1]^K."""
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=num_dimensions, scramble=True, seed=seed)
        m = int(np.ceil(np.log2(max(num_points, 2))))
        samples = sampler.random(2**m)
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(samples), size=num_points, replace=False)
        return samples[indices]
    except ImportError:
        rng = np.random.RandomState(seed)
        return rng.uniform(0, 1, size=(num_points, num_dimensions))


# ─────────────────────────────────────────────
# Metric 1: Coverage (Monte Carlo Estimate)
# ─────────────────────────────────────────────

def compute_coverage(
    Z: np.ndarray,
    num_mc_samples: int = 10_000,
    calibration_runs: int = 200,
    coverage_target: float = 0.99,
    seed: int = 42,
) -> float:
    """
    Estimate coverage using Monte Carlo procedure with calibrated radius k.

    From Appendix C:
    1. Generate random test points uniformly over [0,1]^K
    2. A point is "covered" if within Euclidean distance k of any point in Z
    3. Coverage = fraction of covered points

    Radius k is calibrated using an ideal Sobol reference distribution:
    find smallest k such that a perfectly distributed population of size N
    covers at least 99% of reference points. Repeat 200 times, average.

    Args:
        Z: Normalized embeddings, shape (N, K)
        num_mc_samples: Number of random test points
        calibration_runs: Number of Sobol calibrations for radius k
        coverage_target: Target coverage for calibration (0.99)
        seed: Random seed

    Returns:
        Coverage score in [0, 1]
    """
    N, K = Z.shape
    rng = np.random.RandomState(seed)

    # Step 1: Calibrate radius k using ideal Sobol distributions
    radii = []
    for run in range(calibration_runs):
        # Generate an ideal population of same size N
        ref_pop = _generate_sobol_reference(N, K, seed=seed + run)
        # Generate test points
        test_points = rng.uniform(0, 1, size=(num_mc_samples, K))

        # Find minimum distance from each test point to nearest reference point
        # Using vectorized computation for efficiency
        min_dists = np.zeros(num_mc_samples)
        for i in range(num_mc_samples):
            dists = np.linalg.norm(ref_pop - test_points[i], axis=1)
            min_dists[i] = np.min(dists)

        # Find radius where coverage_target fraction of points are covered
        sorted_dists = np.sort(min_dists)
        target_idx = int(np.ceil(coverage_target * num_mc_samples)) - 1
        radii.append(sorted_dists[target_idx])

    k = np.mean(radii)

    # Step 2: Compute actual coverage with calibrated k
    test_points = rng.uniform(0, 1, size=(num_mc_samples, K))
    covered = 0
    for i in range(num_mc_samples):
        dists = np.linalg.norm(Z - test_points[i], axis=1)
        if np.min(dists) <= k:
            covered += 1

    return covered / num_mc_samples


# ─────────────────────────────────────────────
# Metric 2: Convex Hull Volume
# ─────────────────────────────────────────────

def compute_convex_hull_volume(Z: np.ndarray) -> float:
    """
    Compute the volume of the convex hull of the population embeddings.

    A larger volume means personas span a wider region of the diversity space.
    The maximum possible volume for [0,1]^K is 1.0 (the entire unit hypercube).

    For K=2 this is area; for K=3 this is volume.

    Args:
        Z: Normalized embeddings, shape (N, K)

    Returns:
        Convex hull volume (area for 2D)
    """
    N, K = Z.shape
    if N <= K:
        return 0.0

    try:
        hull = ConvexHull(Z)
        return float(hull.volume)
    except Exception:
        # Degenerate case (e.g., all points coplanar)
        return 0.0


# ─────────────────────────────────────────────
# Metric 3: Minimum Pairwise Distance
# ─────────────────────────────────────────────

def compute_min_pairwise_distance(Z: np.ndarray) -> float:
    """
    Compute the minimum Euclidean distance between any two personas.

    Ensures no two personas are (near-)identical. This was the noisiest
    metric in the paper, since it's dominated by the single closest pair.

    Args:
        Z: Normalized embeddings, shape (N, K)

    Returns:
        Minimum pairwise Euclidean distance
    """
    if Z.shape[0] < 2:
        return 0.0

    distances = pdist(Z, metric="euclidean")
    return float(np.min(distances))


# ─────────────────────────────────────────────
# Metric 4: Mean Pairwise Distance
# ─────────────────────────────────────────────

def compute_mean_pairwise_distance(Z: np.ndarray) -> float:
    """
    Compute the average Euclidean distance between all pairs of personas.

    Measures the overall spread of the population. Higher is better.

    Args:
        Z: Normalized embeddings, shape (N, K)

    Returns:
        Mean pairwise Euclidean distance
    """
    if Z.shape[0] < 2:
        return 0.0

    distances = pdist(Z, metric="euclidean")
    return float(np.mean(distances))


# ─────────────────────────────────────────────
# Metric 5: Dispersion (Largest Empty Region)
# ─────────────────────────────────────────────

def compute_dispersion(
    Z: np.ndarray,
    num_samples: int = 10_000,
    seed: int = 42,
) -> float:
    """
    Compute dispersion: the radius of the largest empty region.

    Estimated by sampling random points in [0,1]^K and finding the one
    farthest from any persona embedding. Lower is better — means fewer
    large gaps in coverage.

    Args:
        Z: Normalized embeddings, shape (N, K)
        num_samples: Number of random test points
        seed: Random seed

    Returns:
        Dispersion (radius of largest empty ball)
    """
    N, K = Z.shape
    rng = np.random.RandomState(seed)

    test_points = rng.uniform(0, 1, size=(num_samples, K))

    max_min_dist = 0.0
    for i in range(num_samples):
        dists = np.linalg.norm(Z - test_points[i], axis=1)
        min_dist = np.min(dists)
        if min_dist > max_min_dist:
            max_min_dist = min_dist

    return float(max_min_dist)


# ─────────────────────────────────────────────
# Metric 6: KL Divergence from Ideal Distribution
# ─────────────────────────────────────────────

def compute_kl_divergence(
    Z: np.ndarray,
    num_sobol_samples: int = 200,
    num_bins: int = 10,
    seed: int = 42,
) -> float:
    """
    Compute KL divergence between empirical distribution of Z and an ideal
    quasi-random (Sobol) reference distribution.

    Penalizes both excessive clustering and uneven densities.
    Lower is better.

    Uses histogram-based KL estimation: discretize the space into bins
    and compare bin occupancy between Z and the Sobol reference.

    Args:
        Z: Normalized embeddings, shape (N, K)
        num_sobol_samples: Number of Sobol reference runs to average
        num_bins: Number of bins per dimension for histogram
        seed: Random seed

    Returns:
        KL divergence (lower = closer to ideal distribution)
    """
    N, K = Z.shape

    # Create histogram of empirical distribution
    bins = np.linspace(0, 1, num_bins + 1)

    def points_to_histogram(points: np.ndarray) -> np.ndarray:
        """Convert points to a normalized histogram."""
        if K == 1:
            hist, _ = np.histogram(points[:, 0], bins=bins, density=False)
        elif K == 2:
            hist, _, _ = np.histogram2d(
                points[:, 0], points[:, 1], bins=[bins, bins]
            )
            hist = hist.flatten()
        else:
            # For K >= 3, use per-dimension marginal histograms concatenated
            hists = []
            for d in range(K):
                h, _ = np.histogram(points[:, d], bins=bins, density=False)
                hists.append(h)
            hist = np.concatenate(hists)

        # Normalize to probability distribution with Laplace smoothing
        hist = hist.astype(float) + 1e-10
        return hist / hist.sum()

    p = points_to_histogram(Z)

    # Average KL over multiple Sobol reference distributions
    kl_values = []
    for run in range(num_sobol_samples):
        ref = _generate_sobol_reference(N, K, seed=seed + run)
        q = points_to_histogram(ref)

        # KL(P || Q) = sum(P * log(P/Q))
        kl = np.sum(p * np.log(p / q))
        kl_values.append(kl)

    return float(np.mean(kl_values))


# ─────────────────────────────────────────────
# Combined Metrics
# ─────────────────────────────────────────────

def compute_all_metrics(
    Z_raw: np.ndarray,
    seed: int = 42,
    fast_mode: bool = False,
) -> Dict[str, float]:
    """
    Compute all 6 diversity metrics on the population embeddings.

    Args:
        Z_raw: Raw response embeddings, shape (N, K), values in [1, 5]
        seed: Random seed
        fast_mode: If True, use fewer samples for faster computation

    Returns:
        Dict mapping metric names to values.
        For maximize metrics (coverage, hull, mean_dist, min_dist): higher is better.
        For minimize metrics (dispersion, kl_div): lower is better.
    """
    Z = normalize_embeddings(Z_raw)
    N, K = Z.shape

    mc_samples = 2_000 if fast_mode else 10_000
    cal_runs = 50 if fast_mode else 200
    disp_samples = 2_000 if fast_mode else 10_000
    kl_samples = 50 if fast_mode else 200

    print(f"\n  Computing diversity metrics (N={N}, K={K})...")

    coverage = compute_coverage(Z, num_mc_samples=mc_samples,
                                calibration_runs=cal_runs, seed=seed)
    print(f"    Coverage: {coverage:.4f}")

    hull_volume = compute_convex_hull_volume(Z)
    print(f"    Convex Hull Volume: {hull_volume:.4f}")

    min_dist = compute_min_pairwise_distance(Z)
    print(f"    Min Pairwise Distance: {min_dist:.4f}")

    mean_dist = compute_mean_pairwise_distance(Z)
    print(f"    Mean Pairwise Distance: {mean_dist:.4f}")

    dispersion = compute_dispersion(Z, num_samples=disp_samples, seed=seed)
    print(f"    Dispersion: {dispersion:.4f}")

    kl_div = compute_kl_divergence(Z, num_sobol_samples=kl_samples, seed=seed)
    print(f"    KL Divergence: {kl_div:.4f}")

    return {
        "coverage": coverage,
        "convex_hull_volume": hull_volume,
        "min_pairwise_distance": min_dist,
        "mean_pairwise_distance": mean_dist,
        "dispersion": dispersion,
        "kl_divergence": kl_div,
    }


def print_metrics(metrics: Dict[str, float]):
    """Pretty-print diversity metrics with direction indicators."""
    print(f"\n{'='*50}")
    print("DIVERSITY METRICS")
    print(f"{'='*50}")
    directions = {
        "coverage": "↑ higher is better",
        "convex_hull_volume": "↑ higher is better",
        "min_pairwise_distance": "↑ higher is better",
        "mean_pairwise_distance": "↑ higher is better",
        "dispersion": "↓ lower is better",
        "kl_divergence": "↓ lower is better",
    }
    for name, value in metrics.items():
        direction = directions.get(name, "")
        print(f"  {name:<30} {value:>8.4f}  {direction}")
    print(f"{'='*50}")
