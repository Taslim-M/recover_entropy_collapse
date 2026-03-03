"""
Diversity Sampler

Implements the quasi-random Monte Carlo sampling scheme described in Section 3.3.

The third (and ultimately dominant) initial generator uses Sobol quasi-random
sequences to sample continuous positions along each diversity axis. These positions
are then translated into persona trait descriptions by the LLM.

After the first extinction event (~100 iterations), only generators from the
quasi-random Monte Carlo sampling family survived in Stage 1.
"""

import numpy as np
from typing import List, Tuple


def sobol_sequence(num_points: int, num_dimensions: int, seed: int = 0) -> np.ndarray:
    """
    Generate quasi-random Sobol sequence points in [0, 1]^d.

    Sobol sequences provide better coverage of the unit hypercube than
    uniform random sampling, with lower discrepancy. This is why the paper
    found them superior to other sampling approaches.

    Uses scipy if available, falls back to a simple stratified approach.

    Args:
        num_points: Number of points to generate (N personas)
        num_dimensions: Number of diversity axes (K)
        seed: Random seed for reproducibility

    Returns:
        Array of shape (num_points, num_dimensions) with values in [0, 1]
    """
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=num_dimensions, scramble=True, seed=seed)
        # Sobol generates 2^m points; we take what we need
        # Find smallest power of 2 >= num_points
        m = int(np.ceil(np.log2(max(num_points, 2))))
        samples = sampler.random(2**m)
        # Randomly select num_points from the generated set
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(samples), size=num_points, replace=False)
        return samples[indices]
    except ImportError:
        # Fallback: stratified random sampling
        # Divide each axis into sqrt(N) strata and sample within
        return _stratified_sample(num_points, num_dimensions, seed)


def _stratified_sample(
    num_points: int, num_dimensions: int, seed: int = 0
) -> np.ndarray:
    """
    Fallback stratified sampling when scipy is unavailable.
    Provides better coverage than pure uniform random.
    """
    rng = np.random.RandomState(seed)

    # Use Latin Hypercube-style sampling
    samples = np.zeros((num_points, num_dimensions))
    for d in range(num_dimensions):
        # Create evenly spaced intervals and sample within each
        intervals = np.linspace(0, 1, num_points + 1)
        for i in range(num_points):
            samples[i, d] = rng.uniform(intervals[i], intervals[i + 1])
        # Shuffle to break the correlation between dimensions
        rng.shuffle(samples[:, d])

    return samples


def positions_to_labels(
    positions: np.ndarray,
    dimension_names: List[str],
) -> List[dict]:
    """
    Convert numerical Sobol positions to human-readable trait labels.

    Maps continuous [0, 1] positions to descriptive labels:
      [0.0, 0.15) -> "Extremely low"
      [0.15, 0.3) -> "Very low"
      [0.3, 0.45) -> "Low"
      [0.45, 0.55) -> "Moderate"
      [0.55, 0.7) -> "High"
      [0.7, 0.85) -> "Very high"
      [0.85, 1.0] -> "Extremely high"

    Args:
        positions: Array of shape (N, K) with values in [0, 1]
        dimension_names: List of K dimension names

    Returns:
        List of N dicts, each mapping dimension name to
        {"value": float, "label": str}
    """
    def value_to_label(v: float) -> str:
        if v < 0.15:
            return "extremely low"
        elif v < 0.30:
            return "very low"
        elif v < 0.45:
            return "low"
        elif v < 0.55:
            return "moderate"
        elif v < 0.70:
            return "high"
        elif v < 0.85:
            return "very high"
        else:
            return "extremely high"

    results = []
    for i in range(positions.shape[0]):
        persona_axes = {}
        for j, dim_name in enumerate(dimension_names):
            val = float(positions[i, j])
            persona_axes[dim_name] = {
                "value": round(val, 3),
                "label": value_to_label(val),
            }
        results.append(persona_axes)

    return results


def generate_diversity_positions(
    num_personas: int,
    num_dimensions: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate well-distributed positions in diversity space.

    This is the core of the quasi-random Monte Carlo approach that
    dominated the evolutionary search. Points are distributed to
    maximize coverage of the [0, 1]^K hypercube.

    Args:
        num_personas: N, number of personas to position
        num_dimensions: K, number of diversity axes
        seed: Random seed

    Returns:
        Array of shape (N, K) with positions in [0, 1]
    """
    positions = sobol_sequence(num_personas, num_dimensions, seed=seed)

    # Ensure we have extreme positions represented
    # The paper found that covering axis extremes is critical
    # We nudge the most extreme existing points slightly further out
    for d in range(num_dimensions):
        min_idx = np.argmin(positions[:, d])
        max_idx = np.argmax(positions[:, d])
        positions[min_idx, d] = max(0.0, positions[min_idx, d] * 0.5)
        positions[max_idx, d] = min(1.0, 1.0 - (1.0 - positions[max_idx, d]) * 0.5)

    return np.clip(positions, 0.0, 1.0)


def print_positions(positions: np.ndarray, dimension_names: List[str]):
    """Pretty-print the sampled diversity positions."""
    labels = positions_to_labels(positions, dimension_names)
    print(f"\nDiversity Positions ({len(labels)} personas, {len(dimension_names)} axes):")
    print("-" * 70)
    for i, persona_axes in enumerate(labels):
        parts = []
        for dim_name, info in persona_axes.items():
            parts.append(f"{dim_name}={info['value']:.2f} ({info['label']})")
        print(f"  Persona {i+1:2d}: {', '.join(parts)}")
    print("-" * 70)
