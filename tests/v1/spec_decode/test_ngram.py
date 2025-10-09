# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np

from vllm.config import ModelConfig, SpeculativeConfig, VllmConfig
from vllm.v1.spec_decode.ngram_proposer import (
    NgramProposer,
    _find_longest_matched_ngram_and_propose_tokens,
)


def test_find_longest_matched_ngram_and_propose_tokens():
    tokens = np.array([1, 2, 3, 4, 1, 2, 3, 5, 6])
    result = _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens=tokens, min_ngram=2, max_ngram=2, max_model_len=1024, k=2
    )
    assert len(result) == 0

    tokens = np.array([1, 2, 3, 4, 1, 2, 3])
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=2, max_ngram=2, max_model_len=1024, k=3
        ),
        np.array([4, 1, 2]),
    )
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=2, max_ngram=2, max_model_len=1024, k=2
        ),
        np.array([4, 1]),
    )
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=1, max_ngram=1, max_model_len=1024, k=3
        ),
        np.array([4, 1, 2]),
    )
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=1, max_ngram=1, max_model_len=1024, k=2
        ),
        np.array([4, 1]),
    )

    tokens = np.array([1, 3, 6, 2, 3, 4, 1, 2, 3])
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=2, max_ngram=2, max_model_len=1024, k=3
        ),
        np.array([4, 1, 2]),
    )
    # Return on the first match
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=1, max_ngram=1, max_model_len=1024, k=2
        ),
        np.array([6, 2]),
    )


def test_ngram_proposer():
    def get_ngram_proposer(min_n: int, max_n: int, k: int) -> NgramProposer:
        # Dummy model config. Just to set max_model_len.
        model_config = ModelConfig(model="facebook/opt-125m")
        return NgramProposer(
            vllm_config=VllmConfig(
                model_config=model_config,
                speculative_config=SpeculativeConfig(
                    prompt_lookup_min=min_n,
                    prompt_lookup_max=max_n,
                    num_speculative_tokens=k,
                    method="ngram",
                ),
            )
        )

    # No match.
    token_ids_cpu = np.array([[1, 2, 3, 4, 5]])
    result = get_ngram_proposer(min_n=2, max_n=2, k=2).propose(
        sampled_token_ids=[[0]],
        req_ids=["0"],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
        spec_decode_unsupported_reqs=(),
    )
    assert len(result[0]) == 0

    # No match for 4-gram.
    token_ids_cpu = np.array([[1, 2, 3, 4, 1, 2, 3]])
    result = get_ngram_proposer(min_n=4, max_n=4, k=2).propose(
        sampled_token_ids=[[0]],
        req_ids=["0"],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
        spec_decode_unsupported_reqs=(),
    )
    assert len(result[0]) == 0

    # No match for 4-gram but match for 3-gram.
    token_ids_cpu = np.array([[1, 2, 3, 4, 1, 2, 3]])
    result = get_ngram_proposer(min_n=3, max_n=4, k=2).propose(
        sampled_token_ids=[[0]],
        req_ids=["0"],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
        spec_decode_unsupported_reqs=(),
    )
    assert np.array_equal(result, np.array([[4, 1]]))

    # Match for both 4-gram and 3-gram.
    # In this case, the proposer should return the 4-gram match.
    token_ids_cpu = np.array([[2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4]])
    result = get_ngram_proposer(min_n=3, max_n=4, k=2).propose(
        sampled_token_ids=[[0]],
        req_ids=["0"],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
        spec_decode_unsupported_reqs=(),
    )
    assert np.array_equal(result, np.array([[1, 2]]))  # Not [5, 1]]

    # Match for 2-gram and 3-gram, but not 4-gram.
    token_ids_cpu = np.array([[3, 4, 5, 2, 3, 4, 1, 2, 3, 4]])
    result = get_ngram_proposer(min_n=2, max_n=4, k=2).propose(
        sampled_token_ids=[[0]],
        req_ids=["0"],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
        spec_decode_unsupported_reqs=(),
    )
    assert np.array_equal(result, np.array([[1, 2]]))  # Not [5, 2]]

    # Multiple 3-gram matched, but always pick the first one.
    token_ids_cpu = np.array([[1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3]])
    result = get_ngram_proposer(min_n=3, max_n=3, k=2).propose(
        sampled_token_ids=[[0]],
        req_ids=["0"],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
        spec_decode_unsupported_reqs=(),
    )
    assert np.array_equal(result, np.array([[100, 1]]))

    # check empty input
    token_ids_cpu = np.array([[]])
    result = get_ngram_proposer(min_n=2, max_n=2, k=2).propose(
        sampled_token_ids=[[0]],
        req_ids=["0"],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
        spec_decode_unsupported_reqs=(),
    )
    assert len(result[0]) == 0

    # check multibatch input
    # first request has 5 tokens and a match
    # second request has 3 tokens and no match. Padded with -1 for max len 5
    token_ids_cpu = np.array([[1, 2, 3, 1, 2], [4, 5, 6, -1, -1]])
    result = get_ngram_proposer(min_n=2, max_n=2, k=2).propose(
        sampled_token_ids=[[0], [1]],
        req_ids=["0", "1"],
        num_tokens_no_spec=np.array([5, 3]),
        token_ids_cpu=token_ids_cpu,
        spec_decode_unsupported_reqs=(),
    )
    assert len(result[0]) == 2
    assert np.array_equal(result[0], np.array([3, 1]))
    assert np.array_equal(result[1], np.array([]))

    # test if 0 threads available: can happen if TP size > CPU count
    ngram_proposer = get_ngram_proposer(min_n=2, max_n=2, k=2)
    ngram_proposer.num_numba_thread_available = 0
    # set max_model_len to 2 * threshold to ensure multithread is used
    num_tokens_threshold = ngram_proposer.num_tokens_threshold
    ngram_proposer.max_model_len = 2 * num_tokens_threshold
    # using multibatch test
    middle_integer = num_tokens_threshold // 2
    input_1 = [_ for _ in range(num_tokens_threshold)]
    input_1 += [middle_integer, middle_integer + 1]
    input_2 = [-1] * len(input_1)
    input_2[:3] = [4, 5, 6]
    token_ids_cpu = np.array([input_1, input_2])
    result = ngram_proposer.propose(
        sampled_token_ids=[[0], [1]],
        req_ids=["0", "1"],
        num_tokens_no_spec=np.array([len(input_1), 3]),
        token_ids_cpu=token_ids_cpu,
        spec_decode_unsupported_reqs=(),
    )
    assert len(result[0]) == 2
    assert np.array_equal(result[0], np.array([middle_integer + 2, middle_integer + 3]))
    assert np.array_equal(result[1], np.array([]))


def test_ngram_recency_bias():
    """Test that recency bias prefers the last match instead of the first."""
    # Multiple 3-gram matches: [1, 2, 3] appears 4 times
    tokens = np.array([1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3])
    
    # Without recency bias (default) - returns tokens after FIRST match
    result_first = _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens=tokens,
        min_ngram=3,
        max_ngram=3,
        max_model_len=1024,
        k=2,
        use_recency_bias=False,
    )
    assert np.array_equal(result_first, np.array([100, 1]))
    
    # With recency bias - returns tokens after LAST match
    result_last = _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens=tokens,
        min_ngram=3,
        max_ngram=3,
        max_model_len=1024,
        k=2,
        use_recency_bias=True,
    )
    assert np.array_equal(result_last, np.array([300, 1]))
    
    # Test with 2-gram matches
    tokens = np.array([5, 6, 10, 5, 6, 20, 5, 6])
    
    # Without recency bias - first match
    result_first = _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens=tokens,
        min_ngram=2,
        max_ngram=2,
        max_model_len=1024,
        k=1,
        use_recency_bias=False,
    )
    assert np.array_equal(result_first, np.array([10]))
    
    # With recency bias - last match
    result_last = _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens=tokens,
        min_ngram=2,
        max_ngram=2,
        max_model_len=1024,
        k=1,
        use_recency_bias=True,
    )
    assert np.array_equal(result_last, np.array([20]))


def test_ngram_proposer_with_recency_bias():
    """Integration test with NgramProposer using recency bias."""
    def get_ngram_proposer_with_recency(
        min_n: int, 
        max_n: int, 
        k: int, 
        recency_bias: bool = False,
    ) -> NgramProposer:
        model_config = ModelConfig(model="facebook/opt-125m")
        return NgramProposer(
            vllm_config=VllmConfig(
                model_config=model_config,
                speculative_config=SpeculativeConfig(
                    prompt_lookup_min=min_n,
                    prompt_lookup_max=max_n,
                    num_speculative_tokens=k,
                    method="ngram",
                    ngram_recency_bias=recency_bias,
                ),
            )
        )
    
    # Test with recency bias enabled
    token_ids_cpu = np.array([[1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3]])
    proposer = get_ngram_proposer_with_recency(
        min_n=3, max_n=3, k=2, recency_bias=True
    )
    result = proposer.propose(
        sampled_token_ids=[[0]],
        req_ids=["0"],
        num_tokens_no_spec=np.array([len(token_ids_cpu[0])]),
        token_ids_cpu=token_ids_cpu,
        spec_decode_unsupported_reqs=(),
    )
    assert np.array_equal(result[0], np.array([200, 1]))
    
    # Test without recency bias (default behavior)
    proposer_no_bias = get_ngram_proposer_with_recency(
        min_n=3, max_n=3, k=2, recency_bias=False
    )
    result_no_bias = proposer_no_bias.propose(
        sampled_token_ids=[[0]],
        req_ids=["0"],
        num_tokens_no_spec=np.array([len(token_ids_cpu[0])]),
        token_ids_cpu=token_ids_cpu,
        spec_decode_unsupported_reqs=(),
    )
    assert np.array_equal(result_no_bias[0], np.array([100, 1]))
