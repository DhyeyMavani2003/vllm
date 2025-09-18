# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
from dataclasses import dataclass
import importlib.util
import pathlib

# Import the module directly by path to avoid importing the top-level vllm
# package (which brings in heavy deps like torch) during unit tests.
_ROOT = pathlib.Path(__file__).resolve().parents[3]
_NG_MOD_PATH = _ROOT / "vllm" / "v1" / "spec_decode" / "ngram_proposer.py"
_spec = importlib.util.spec_from_file_location("_ngram_proposer_mod",
                                              str(_NG_MOD_PATH))
_ng_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_ng_mod)

NgramProposer = _ng_mod.NgramProposer
_find_longest_matched_ngram_and_propose_tokens = (
    _ng_mod._find_longest_matched_ngram_and_propose_tokens)


@dataclass
class _DummyModelConfig:
    max_model_len: int = 1024


@dataclass
class _DummySpeculativeConfig:
    prompt_lookup_min: int
    prompt_lookup_max: int
    num_speculative_tokens: int
    method: str = "ngram"


@dataclass
class _DummyVllmConfig:
    model_config: _DummyModelConfig
    speculative_config: _DummySpeculativeConfig


def test_find_longest_matched_ngram_and_propose_tokens():
    tokens = np.array([1, 2, 3, 4, 1, 2, 3, 5, 6])
    assert _find_longest_matched_ngram_and_propose_tokens(origin_tokens=tokens,
                                                          min_ngram=2,
                                                          max_ngram=2,
                                                          max_model_len=1024,
                                                          k=2) is None

    tokens = np.array([1, 2, 3, 4, 1, 2, 3])
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(origin_tokens=tokens,
                                                       min_ngram=2,
                                                       max_ngram=2,
                                                       max_model_len=1024,
                                                       k=3),
        np.array([4, 1, 2]))
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(origin_tokens=tokens,
                                                       min_ngram=2,
                                                       max_ngram=2,
                                                       max_model_len=1024,
                                                       k=2), np.array([4, 1]))
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(origin_tokens=tokens,
                                                       min_ngram=1,
                                                       max_ngram=1,
                                                       max_model_len=1024,
                                                       k=3),
        np.array([4, 1, 2]))
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(origin_tokens=tokens,
                                                       min_ngram=1,
                                                       max_ngram=1,
                                                       max_model_len=1024,
                                                       k=2), np.array([4, 1]))

    tokens = np.array([1, 3, 6, 2, 3, 4, 1, 2, 3])
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(origin_tokens=tokens,
                                                       min_ngram=2,
                                                       max_ngram=2,
                                                       max_model_len=1024,
                                                       k=3),
        np.array([4, 1, 2]))
    # Recency bias: when multiple 1-gram matches, pick the most recent prior
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(origin_tokens=tokens,
                                                       min_ngram=1,
                                                       max_ngram=1,
                                                       max_model_len=1024,
                                                       k=2), np.array([4, 1]))


def test_ngram_proposer():

    def ngram_proposer(min_n: int, max_n: int, k: int) -> NgramProposer:
        # Use lightweight dummy config objects to avoid importing heavy deps.
        vllm_config = _DummyVllmConfig(
            model_config=_DummyModelConfig(max_model_len=1024),
            speculative_config=_DummySpeculativeConfig(
                prompt_lookup_min=min_n,
                prompt_lookup_max=max_n,
                num_speculative_tokens=k,
                method="ngram",
            ),
        )
        return NgramProposer(vllm_config=vllm_config)  # type: ignore[arg-type]

    # No match.
    result = ngram_proposer(
        min_n=2, max_n=2,
        k=2).propose(context_token_ids=np.array([1, 2, 3, 4, 5]))
    assert result is None

    # No match for 4-gram.
    result = ngram_proposer(
        min_n=4, max_n=4,
        k=2).propose(context_token_ids=np.array([1, 2, 3, 4, 1, 2, 3]))
    assert result is None

    # No match for 4-gram but match for 3-gram.
    result = ngram_proposer(
        min_n=3, max_n=4,
        k=2).propose(context_token_ids=np.array([1, 2, 3, 4, 1, 2, 3]))
    assert np.array_equal(result, np.array([4, 1]))

    # Match for both 4-gram and 3-gram.
    # In this case, the proposer should return the 4-gram match.
    result = ngram_proposer(min_n=3, max_n=4, k=2).propose(
        context_token_ids=np.array([2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4]))
    assert np.array_equal(result, np.array([1, 2]))  # Not [5, 1]

    # Match for 2-gram and 3-gram, but not 4-gram.
    result = ngram_proposer(min_n=2, max_n=4, k=2).propose(
        context_token_ids=np.array([3, 4, 5, 2, 3, 4, 1, 2, 3, 4]))
    assert np.array_equal(result, np.array([1, 2]))  # Not [5, 2]

    # Multiple 3-gram matched; prefer the most recent prior occurrence.
    result = ngram_proposer(
        min_n=3, max_n=3, k=2).propose(context_token_ids=np.array(
            [1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3]))
    assert np.array_equal(result, np.array([300, 1]))
