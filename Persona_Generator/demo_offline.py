"""
Offline Demo: Sobol Sampling + Diversity Metrics

This demo runs WITHOUT an API key to demonstrate:
1. How Sobol quasi-random sampling positions personas in diversity space
2. How all 6 diversity metrics work
3. Comparison: Sobol sampling vs uniform random vs clustered (simulating mode collapse)

This illustrates WHY quasi-random sampling dominates in the paper's
evolutionary search — it provides much better coverage than naive approaches.

Usage:
    python demo_offline.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diversity_sampler import generate_diversity_positions, positions_to_labels
from diversity_metrics import (
    compute_all_metrics,
    print_metrics,
    normalize_embeddings,
)


def simulate_likert_responses(positions: np.ndarray, noise_std: float = 0.3) -> np.ndarray:
    """
    Simulate what Likert-scale responses would look like for given diversity positions.

    Maps [0,1] positions to [1,5] Likert scores with some noise,
    simulating what would happen after Concordia evaluation.
    """
    # Map [0,1] → [1,5] with noise
    scores = positions * 4.0 + 1.0  # [1, 5]
    noise = np.random.randn(*scores.shape) * noise_std
    return np.clip(scores + noise, 1.0, 5.0)


def demo_sampling_comparison():
    """Compare three sampling strategies on diversity metrics."""

    N = 25  # Same as paper
    K = 3   # 3 diversity axes (like elderly rural Japan)
    dimensions = ["community_cohesion", "technology_adoption", "adherence_to_tradition"]

    print("=" * 70)
    print("DEMO: Sampling Strategy Comparison")
    print(f"N={N} personas, K={K} axes: {dimensions}")
    print("=" * 70)

    strategies = {}

    # ── Strategy 1: Sobol Quasi-Random (paper's winning approach) ──
    print("\n\n--- Strategy 1: Sobol Quasi-Random Sampling ---")
    print("(This is what survived after the first extinction event)")
    sobol_positions = generate_diversity_positions(N, K, seed=42)
    sobol_responses = simulate_likert_responses(sobol_positions, noise_std=0.2)
    print(f"Position range per axis:")
    for d in range(K):
        print(f"  {dimensions[d]}: [{sobol_positions[:, d].min():.3f}, "
              f"{sobol_positions[:, d].max():.3f}]")
    strategies["sobol"] = sobol_responses

    # ── Strategy 2: Uniform Random ──
    print("\n--- Strategy 2: Uniform Random Sampling ---")
    print("(Naive approach — no structure)")
    np.random.seed(42)
    random_positions = np.random.uniform(0, 1, size=(N, K))
    random_responses = simulate_likert_responses(random_positions, noise_std=0.2)
    print(f"Position range per axis:")
    for d in range(K):
        print(f"  {dimensions[d]}: [{random_positions[:, d].min():.3f}, "
              f"{random_positions[:, d].max():.3f}]")
    strategies["uniform_random"] = random_responses

    # ── Strategy 3: Clustered (simulating LLM mode collapse) ──
    print("\n--- Strategy 3: Clustered / Mode Collapse ---")
    print("(Simulates what happens when you ask an LLM to 'generate diverse personas')")
    print("(Personas cluster around stereotypical moderate positions)")
    np.random.seed(42)
    # Most points cluster around center with a few mild outliers
    clustered_positions = np.random.normal(0.5, 0.12, size=(N, K))
    clustered_positions = np.clip(clustered_positions, 0.05, 0.95)
    clustered_responses = simulate_likert_responses(clustered_positions, noise_std=0.2)
    print(f"Position range per axis:")
    for d in range(K):
        print(f"  {dimensions[d]}: [{clustered_positions[:, d].min():.3f}, "
              f"{clustered_positions[:, d].max():.3f}]")
    strategies["clustered_mode_collapse"] = clustered_responses

    # ── Compute and Compare Metrics ──
    print("\n\n" + "=" * 70)
    print("DIVERSITY METRICS COMPARISON")
    print("=" * 70)

    all_metrics = {}
    for name, responses in strategies.items():
        print(f"\n--- {name} ---")
        metrics = compute_all_metrics(responses, seed=42, fast_mode=True)
        all_metrics[name] = metrics

    # Summary table
    print(f"\n\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    metric_names = list(all_metrics["sobol"].keys())
    header = f"{'Metric':<30}"
    for strategy in strategies:
        header += f" {strategy[:18]:>18}"
    print(header)
    print("-" * len(header))

    maximize_metrics = {"coverage", "convex_hull_volume", "min_pairwise_distance",
                        "mean_pairwise_distance"}

    for metric_name in metric_names:
        row = f"{metric_name:<30}"
        values = [all_metrics[s][metric_name] for s in strategies]
        is_maximize = metric_name in maximize_metrics
        best_idx = np.argmax(values) if is_maximize else np.argmin(values)

        for i, strategy in enumerate(strategies):
            marker = " ★" if i == best_idx else "  "
            row += f" {all_metrics[strategy][metric_name]:>16.4f}{marker}"
        print(row)

    print(f"\n★ = best for that metric")
    print(f"\nKey takeaway: Sobol quasi-random sampling provides better coverage")
    print(f"and more uniform distribution than random sampling, and MUCH better")
    print(f"than the clustered mode-collapse pattern typical of naive LLM generation.")

    return all_metrics


def demo_show_persona_positions():
    """Show what Sobol-sampled persona positions look like for elderly rural Japan."""
    N = 25
    dimensions = ["community_cohesion", "technology_adoption", "adherence_to_tradition"]

    print("\n\n" + "=" * 70)
    print("DEMO: Sobol-Sampled Persona Positions")
    print("Context: Elderly Rural Japan 2010")
    print(f"{'='*70}")

    positions = generate_diversity_positions(N, len(dimensions), seed=42)
    labels = positions_to_labels(positions, dimensions)

    print(f"\n{'#':<4} {'Comm. Cohesion':>18} {'Tech Adoption':>18} {'Tradition':>18}")
    print("-" * 62)
    for i, persona_axes in enumerate(labels):
        vals = list(persona_axes.values())
        print(f"{i+1:<4} {vals[0]['value']:>8.3f} ({vals[0]['label'][:8]:>8}) "
              f"{vals[1]['value']:>8.3f} ({vals[1]['label'][:8]:>8}) "
              f"{vals[2]['value']:>8.3f} ({vals[2]['label'][:8]:>8})")

    # Show interesting combinations
    print(f"\nNotable combinations that Sobol sampling discovers:")
    for i, persona_axes in enumerate(labels):
        vals = list(persona_axes.values())
        # Find interesting extremes
        scores = [v['value'] for v in vals]
        if max(scores) > 0.85 and min(scores) < 0.15:
            lbls = [v['label'] for v in vals]
            print(f"  Persona {i+1}: {dimensions[0]}={lbls[0]}, "
                  f"{dimensions[1]}={lbls[1]}, "
                  f"{dimensions[2]}={lbls[2]}")
            print(f"    → Rare combination that naive LLM generation would miss")


if __name__ == "__main__":
    demo_show_persona_positions()
    metrics = demo_sampling_comparison()
