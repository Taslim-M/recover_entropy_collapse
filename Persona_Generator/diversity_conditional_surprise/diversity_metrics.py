"""
diversity_metrics.py
====================
Computes the diversity metrics proposed in:
  "A Base-Model Diversity Metric for LLM Outputs via Progressive Conditional Surprise"

All information-theoretic quantities are normalized per UTF-8 byte (not per token),
making results tokenizer-agnostic and comparable across base models.

Core quantities
---------------
  ak   – per-byte conditional cross-entropy of response k given all prior responses
  E    – excess entropy (bits/byte): learnable inter-response structure above the noise floor
  C    – coherence: geometric-mean per-byte probability the base model assigns to outputs
  σ_ℓ  – coherence spread: std of per-response cross-entropies (detects mixed quality)
  D    – per-byte diversity score = C × E
  D_total – total diversity score = C_total × E_total  (in bits, not normalized by length)

Usage
-----
  client = DiversityMetricsClient(base_url="http://<your-vllm-host>:8000", model="<model-name>")
  result = client.compute(prompt="Write a short story.", responses=[r1, r2, ..., rn])
  result.summary()
"""

from __future__ import annotations

import math
import time
import random
import logging
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response labels used to format the conditioning context (Response A, B, …)
# Falls back to "Response {k}" for k > 26.
# ---------------------------------------------------------------------------
_LABELS = [chr(65 + i) for i in range(26)]  # A … Z


def _label(k: int) -> str:
    """Return the label for the k-th response (0-indexed)."""
    return _LABELS[k] if k < len(_LABELS) else f"R{k + 1}"


def _byte_len(s: str) -> int:
    return len(s.encode("utf-8"))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DiversityResult:
    """
    Full result object returned by DiversityMetricsClient.compute().

    Attributes
    ----------
    prompt : str
        The prompt used to generate responses.
    responses : list[str]
        The n responses evaluated.
    ak_curve : list[float]
        Per-byte conditional cross-entropy for each response (bits/byte).
        ak_curve[0] = unconditional; ak_curve[k] conditioned on responses 0..k-1.
    unconditional_h : list[float]
        Per-byte unconditional cross-entropy hθ(rᵢ | p) for each response.
    excess_entropy_rate : float
        E (bits/byte): sum of (ak - a_inf) across k, where a_inf ≈ ak[-1].
    excess_entropy_total : float
        E_total (bits): byte-weighted sum of (ak - a_inf).
    coherence : float
        C: geometric mean per-byte probability = 2^(-mean unconditional_h).
    coherence_total : float
        C_total: geometric mean of total (string-level) probability across responses.
    coherence_spread : float
        σ_ℓ: std of unconditional_h values.
    diversity_rate : float
        D = C × E  (bits/byte).
    diversity_total : float
        D_total = C_total × E_total  (bits).
    diversity_lower : float
        D⁻ = C⁻ × E  (lower diversity band when coherence is heterogeneous).
    diversity_upper : float
        D⁺ = C⁺ × E  (upper diversity band).
    n_permutations : int
        Number of random permutations averaged over.
    permutation_results : list[dict]
        Raw per-permutation ak curves (if n_permutations > 1).
    """
    prompt: str
    responses: list[str]
    ak_curve: list[float]
    unconditional_h: list[float]
    excess_entropy_rate: float       # E  (bits/byte)
    excess_entropy_total: float      # E_total (bits)
    coherence: float                 # C  (per-byte)
    coherence_total: float           # C_total (string-level)
    coherence_spread: float          # σ_ℓ
    diversity_rate: float            # D = C × E
    diversity_total: float           # D_total = C_total × E_total
    diversity_lower: float           # D⁻
    diversity_upper: float           # D⁺
    n_permutations: int = 1
    permutation_results: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  Diversity Metric Summary",
            "=" * 60,
            f"  n responses     : {len(self.responses)}",
            f"  n permutations  : {self.n_permutations}",
            "",
            "  Progressive Conditional Surprise curve (ak):",
        ]
        for i, a in enumerate(self.ak_curve):
            label = "  (unconditional)" if i == 0 else f"  | prev {i}"
            lines.append(f"    a[{i+1:>2}] = {a:.4f} bits/byte  {label}")
        lines += [
            "",
            f"  a_inf  (floor) ≈ {self.ak_curve[-1]:.4f} bits/byte",
            "",
            "  Scalar Summaries",
            "  ----------------",
            f"  E  (excess entropy rate)   = {self.excess_entropy_rate:.4f}  bits/byte",
            f"  E_total                    = {self.excess_entropy_total:.2f}  bits",
            f"  C  (coherence, per-byte)   = {self.coherence:.4f}  (1/PPL)",
            f"  C_total (string-level)     = {self.coherence_total:.6f}",
            f"  σ_ℓ (coherence spread)     = {self.coherence_spread:.4f}",
            "",
            f"  D  = C × E  (rate)         = {self.diversity_rate:.4f}  bits/byte",
            f"  D_total = C_total × E_total = {self.diversity_total:.4f}  bits",
            f"  Diversity band: [{self.diversity_lower:.4f}, {self.diversity_upper:.4f}]  bits/byte",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "responses": self.responses,
            "ak_curve": self.ak_curve,
            "unconditional_h": self.unconditional_h,
            "excess_entropy_rate": self.excess_entropy_rate,
            "excess_entropy_total": self.excess_entropy_total,
            "coherence": self.coherence,
            "coherence_total": self.coherence_total,
            "coherence_spread": self.coherence_spread,
            "diversity_rate": self.diversity_rate,
            "diversity_total": self.diversity_total,
            "diversity_lower": self.diversity_lower,
            "diversity_upper": self.diversity_upper,
            "n_permutations": self.n_permutations,
            "permutation_results": self.permutation_results,
        }


# ---------------------------------------------------------------------------
# vLLM client
# ---------------------------------------------------------------------------

class VLLMClient:
    """
    Thin wrapper around the vLLM OpenAI-compatible completions endpoint.

    We use the ``/v1/completions`` endpoint with ``echo=True`` and
    ``max_tokens=0`` to score existing strings without generating new tokens.
    This returns logprob values for every token in the submitted prompt,
    letting us extract the log-probability of any substring.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 120,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._session = requests.Session()

    # ------------------------------------------------------------------
    def _post(self, payload: dict) -> dict:
        url = f"{self.base_url}/v1/completions"
        for attempt in range(self.max_retries):
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                if attempt == self.max_retries - 1:
                    raise
                wait = self.retry_backoff ** attempt
                logger.warning("Request failed (%s); retrying in %.1fs …", exc, wait)
                time.sleep(wait)
        raise RuntimeError("Unreachable")

    # ------------------------------------------------------------------
    def score_completion(
        self,
        full_text: str,
        completion_start_char: int,
    ) -> tuple[float, int]:
        """
        Score the substring full_text[completion_start_char:] conditioned on
        full_text[:completion_start_char].

        Returns
        -------
        total_logprob : float   (natural log, summed over completion tokens)
        n_bytes       : int     (UTF-8 byte count of the completion)

        How it works
        ------------
        We submit the entire ``full_text`` as the prompt with ``echo=True``
        and ``max_tokens=0``. vLLM returns one logprob per input token.
        We identify which tokens belong to the completion portion by finding
        the token whose cumulative character offset first reaches
        ``completion_start_char``, then sum their log-probabilities.

        Note: logprobs from the API are in natural log (nats). We keep them
        in nats internally and convert to bits (÷ ln 2) only at the end.
        """
        payload = {
            "model": self.model,
            "prompt": full_text,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,
        }
        data = self._post(payload)

        choice = data["choices"][0]
        logprob_info = choice.get("logprobs", {})

        # vLLM returns parallel lists: tokens, token_logprobs, text_offset
        tokens: list[str] = logprob_info.get("tokens", [])
        token_logprobs: list[Optional[float]] = logprob_info.get("token_logprobs", [])
        text_offsets: list[int] = logprob_info.get("text_offset", [])

        if not tokens:
            raise ValueError(
                "No logprob data returned. Ensure the vLLM endpoint is started "
                "with --max-logprobs 1 (or higher) and that echo=True is supported."
            )

        # Sum log-probs for tokens that belong to the completion portion.
        # text_offset[i] is the character offset of the START of token i in full_text.
        # A token belongs to the completion if its start offset >= completion_start_char.
        # The first token's logprob is None (no predecessor), so we skip it.
        total_logprob = 0.0
        for i, (tok, lp, offset) in enumerate(
            zip(tokens, token_logprobs, text_offsets)
        ):
            if offset < completion_start_char:
                continue
            if lp is None:
                # Can happen for the very first token of the entire sequence.
                continue
            total_logprob += lp  # nats

        n_bytes = _byte_len(full_text[completion_start_char:])
        return total_logprob, n_bytes


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _format_context(prompt: str, responses: list[str], include_response_k: bool = True) -> str:
    """
    Build the conditioning context string for the ak computation.

    Format (following Section 7.1 of the paper):
        <prompt>
        Response A: <r1>
        Response B: <r2>
        ...
    """
    parts = [prompt.rstrip()]
    for i, r in enumerate(responses):
        parts.append(f"\nResponse {_label(i)}: {r}")
    return "".join(parts)


def _compute_ak_single_pass(
    client: VLLMClient,
    prompt: str,
    responses: list[str],
) -> list[float]:
    """
    Compute the full ak curve for one ordering of responses using a single
    forward pass (concatenate all responses, extract per-response logprobs).

    Returns ak values in bits/byte.
    """
    n = len(responses)
    ak_values: list[float] = []

    # We build the full concatenated context incrementally to track char offsets.
    # For each response k, the "conditioning prefix" is everything up to and
    # including "Response {label(k)}: " and the completion is response k itself.

    # Build prefix portions: prompt + label header for each response
    prefix = prompt.rstrip()
    char_offsets: list[int] = []  # char offset where response k's text begins

    for i, r in enumerate(responses):
        header = f"\nResponse {_label(i)}: "
        prefix += header
        char_offsets.append(len(prefix))  # start of r_i's text
        prefix += r

    full_text = prefix

    # Single forward pass over the full concatenation
    payload = {
        "model": client.model,
        "prompt": full_text,
        "max_tokens": 0,
        "echo": True,
        "logprobs": 1,
    }
    data = client._post(payload)

    choice = data["choices"][0]
    logprob_info = choice.get("logprobs", {})
    tokens: list[str] = logprob_info.get("tokens", [])
    token_logprobs: list[Optional[float]] = logprob_info.get("token_logprobs", [])
    text_offsets: list[int] = logprob_info.get("text_offset", [])

    if not tokens:
        raise ValueError("No logprob data returned from vLLM. Check --max-logprobs setting.")

    # For each response k, sum logprobs of tokens whose offsets fall within
    # [char_offsets[k], char_offsets[k+1]) — i.e. within response k's text.
    response_end_offsets = char_offsets[1:] + [len(full_text)]

    for k in range(n):
        start = char_offsets[k]
        end = response_end_offsets[k]
        total_logprob = 0.0
        for tok, lp, offset in zip(tokens, token_logprobs, text_offsets):
            if offset < start:
                continue
            if offset >= end:
                break
            if lp is None:
                continue
            total_logprob += lp  # nats

        n_bytes = _byte_len(responses[k])
        if n_bytes == 0:
            logger.warning("Response %d has zero bytes; skipping.", k)
            ak_values.append(float("nan"))
            continue

        # Convert nats → bits, divide by bytes
        ak_bits_per_byte = (-total_logprob / math.log(2)) / n_bytes
        ak_values.append(ak_bits_per_byte)

    return ak_values


def _compute_unconditional_h(
    client: VLLMClient,
    prompt: str,
    responses: list[str],
) -> list[float]:
    """
    Compute hθ(rᵢ | p) for each response independently (unconditional).
    Each call is a short forward pass: just [prompt + header + response].

    Returns per-byte cross-entropy values in bits/byte.
    """
    results = []
    for i, r in enumerate(responses):
        context = prompt.rstrip() + f"\nResponse {_label(i)}: "
        full_text = context + r
        completion_start = len(context)

        total_logprob, n_bytes = client.score_completion(full_text, completion_start)
        if n_bytes == 0:
            results.append(float("nan"))
            continue
        h_bits_per_byte = (-total_logprob / math.log(2)) / n_bytes
        results.append(h_bits_per_byte)

    return results


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

class DiversityMetricsClient:
    """
    Main entry point for computing diversity metrics.

    Parameters
    ----------
    base_url : str
        Base URL of the vLLM server, e.g. ``"http://localhost:8000"``.
    model : str
        Model name as registered in the vLLM server.
    n_permutations : int
        Number of random response orderings to average over. The paper
        recommends 3–5 to reduce ordering noise. Set to 1 for speed.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        n_permutations: int = 3,
        timeout: int = 120,
    ):
        self.client = VLLMClient(base_url=base_url, model=model, timeout=timeout)
        self.n_permutations = n_permutations

    # ------------------------------------------------------------------
    def compute(
        self,
        prompt: str,
        responses: list[str],
        seed: int = 42,
    ) -> DiversityResult:
        """
        Compute all diversity metrics for the given prompt and responses.

        Parameters
        ----------
        prompt : str
            The prompt that generated the responses.
        responses : list[str]
            The policy's outputs to evaluate (typically 10–20 strings).
        seed : int
            Random seed for permutation sampling.

        Returns
        -------
        DiversityResult
        """
        rng = random.Random(seed)
        n = len(responses)
        if n < 2:
            raise ValueError("Need at least 2 responses to compute diversity.")

        logger.info("Computing unconditional cross-entropies hθ(rᵢ | p) …")
        unconditional_h = _compute_unconditional_h(self.client, prompt, responses)

        logger.info(
            "Computing ak curves over %d permutation(s) …", self.n_permutations
        )
        perm_ak_curves: list[list[float]] = []
        permutation_results: list[dict] = []

        for perm_idx in range(self.n_permutations):
            if perm_idx == 0:
                order = list(range(n))  # First pass uses original order
            else:
                order = list(range(n))
                rng.shuffle(order)

            ordered_responses = [responses[i] for i in order]
            ak = _compute_ak_single_pass(self.client, prompt, ordered_responses)
            perm_ak_curves.append(ak)
            permutation_results.append({"order": order, "ak_curve": ak})
            logger.info("  Permutation %d/%d done.", perm_idx + 1, self.n_permutations)

        # Average ak curves across permutations (element-wise)
        ak_curve = [
            sum(perm_ak_curves[p][k] for p in range(self.n_permutations))
            / self.n_permutations
            for k in range(n)
        ]

        # --- Scalar summaries ---
        a_inf = ak_curve[-1]  # Estimate floor as last observed value

        # Excess entropy rate E (bits/byte): sum of (ak - a_inf)
        excess_entropy_rate = sum(max(a - a_inf, 0.0) for a in ak_curve)

        # Byte-weighted excess entropy E_total (bits)
        byte_counts = [_byte_len(r) for r in responses]
        total_bytes = sum(byte_counts)
        excess_entropy_total = sum(
            bc * max(a - a_inf, 0.0)
            for bc, a in zip(byte_counts, ak_curve)
        )

        # Coherence C = 2^(-mean_h)  [per-byte, geometric mean]
        valid_h = [h for h in unconditional_h if not math.isnan(h)]
        mean_h = sum(valid_h) / len(valid_h)
        coherence = 2.0 ** (-mean_h)

        # Coherence spread σ_ℓ
        variance_h = sum((h - mean_h) ** 2 for h in valid_h) / max(len(valid_h) - 1, 1)
        coherence_spread = math.sqrt(variance_h)

        # Coherence band: C± = C · 2^(±σ_ℓ)
        coherence_upper = coherence * (2.0 ** coherence_spread)
        coherence_lower = coherence * (2.0 ** (-coherence_spread))

        # Total coherence C_total = geometric mean of string-level probabilities
        # C_total = 2^(-mean_i [ ||r_i|| * h_i ])
        mean_total_h = sum(
            bc * h for bc, h in zip(byte_counts, unconditional_h)
            if not math.isnan(h)
        ) / total_bytes
        coherence_total = 2.0 ** (-mean_total_h)

        # Diversity scores
        diversity_rate = coherence * excess_entropy_rate          # D  (bits/byte)
        diversity_total = coherence_total * excess_entropy_total  # D_total (bits)
        diversity_upper = coherence_upper * excess_entropy_rate
        diversity_lower = coherence_lower * excess_entropy_rate

        return DiversityResult(
            prompt=prompt,
            responses=responses,
            ak_curve=ak_curve,
            unconditional_h=unconditional_h,
            excess_entropy_rate=excess_entropy_rate,
            excess_entropy_total=excess_entropy_total,
            coherence=coherence,
            coherence_total=coherence_total,
            coherence_spread=coherence_spread,
            diversity_rate=diversity_rate,
            diversity_total=diversity_total,
            diversity_lower=diversity_lower,
            diversity_upper=diversity_upper,
            n_permutations=self.n_permutations,
            permutation_results=permutation_results,
        )

    # ------------------------------------------------------------------
    def compare(
        self,
        prompt: str,
        policy_responses: dict[str, list[str]],
        seed: int = 42,
    ) -> dict[str, DiversityResult]:
        """
        Compute metrics for multiple policies (e.g., base vs. steered).

        Parameters
        ----------
        prompt : str
            Shared prompt for all policies.
        policy_responses : dict mapping policy_name → list of responses

        Returns
        -------
        dict mapping policy_name → DiversityResult
        """
        results = {}
        for name, resps in policy_responses.items():
            logger.info("Computing metrics for policy: %s", name)
            results[name] = self.compute(prompt=prompt, responses=resps, seed=seed)
        return results
