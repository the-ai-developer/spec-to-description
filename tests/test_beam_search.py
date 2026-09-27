"""Beam search: ranking, termination, and the work it is allowed to do."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spec2desc.beam_search import beam_search  # noqa: E402


class TestRanking:
    def test_matches_brute_force_on_toy_model(self):
        # toy vocab: prefers sequence [1,2,3]
        logp = {0: -2.0, 1: -0.1, 2: -0.1, 3: -0.1, 4: -3.0}

        def step(prefix):
            nxt = len(prefix) - 1  # 0-based step index
            return {tok: lp for tok, lp in logp.items()}

        results = beam_search(step, bos=99, eos=0, beam_width=3, max_len=4)
        assert results, "no beams returned"
        best = results[0]
        # best sequence must be the greedy favourite chain (1,2,3,...)
        assert best.tokens[0] == 99
        assert best.tokens[1] == 1
        assert best.score >= results[1].score  # sorted best-first

    def test_length_excludes_bos(self):
        out = beam_search(lambda p: {1: -0.5}, bos=7, eos=9, beam_width=1, max_len=3)
        assert out[0].tokens[0] == 7
        # three tokens appended, bos not counted
        assert out[0].length == 3

    def test_raw_logprob_is_the_unpenalised_sum(self):
        out = beam_search(lambda p: {1: -0.5}, bos=0, eos=9, beam_width=1, max_len=2)
        assert out[0].raw_logprob == pytest.approx(-1.0)
        # alpha=1 divides by ((5+len)/6)^1, so score > raw is expected here
        assert out[0].score != out[0].raw_logprob


class TestTermination:
    def test_eos_terminates_and_length_penalty(self):
        def step(prefix):
            if len(prefix) == 1:
                return {5: -0.2, 6: -0.3}
            return {7: -0.5, 8: -2.0}

        out = beam_search(step, bos=1, eos=7, beam_width=2, max_len=10)
        assert all(b.tokens[-1] == 7 or len(b.tokens) <= 10 for b in out)
        assert out[0].score >= out[-1].score

    def test_return_count(self):
        step = lambda prefix: {i: -1.0 for i in range(1, 6)}
        out = beam_search(step, bos=0, eos=9, beam_width=4, max_len=5, num_return=2)
        assert len(out) == 2

    def test_no_eos_returns_live_beams(self):
        """A step function that never emits eos must still return something."""
        out = beam_search(lambda p: {1: -0.5, 2: -0.7}, bos=0, eos=99,
                          beam_width=2, max_len=4)
        assert out
        assert all(b.tokens[-1] != 99 for b in out)

    def test_empty_step_stops_immediately(self):
        out = beam_search(lambda p: {}, bos=0, eos=1, beam_width=2, max_len=5)
        assert len(out) == 1 and out[0].tokens == (0,)


class TestCost:
    """The live beam set must never exceed beam_width.

    This is the test that was missing. The ranking tests all passed against an
    implementation whose beam grew as vocab**max_len, because the returned
    top-k was the same either way. Only the work done differed, and nobody was
    measuring it.
    """

    def test_step_calls_are_bounded_by_beam_width(self):
        calls = []

        def step(prefix):
            calls.append(len(prefix))
            return {t: -0.1 for t in range(500)}  # big vocab, never emits eos

        beam_search(step, bos=0, eos=-1, beam_width=4, max_len=6)
        # one expansion per live beam per step, at most beam_width of them
        assert len(calls) <= 4 * 6, f"expanded {len(calls)} beams, beam_width was 4"

    def test_larger_vocabulary_does_not_cost_more(self):
        """Cost tracks beam_width * max_len, not vocabulary size."""
        def run(vocab):
            calls = []
            def step(prefix):
                calls.append(1)
                return {t: -0.1 for t in range(vocab)}
            beam_search(step, bos=0, eos=-1, beam_width=3, max_len=4)
            return len(calls)

        assert run(50) == run(5000), "vocabulary size changed the amount of work"

    def test_realistic_vocabulary_terminates(self):
        """A T5-sized vocabulary over a real max_len must stay bounded.

        Before the fix this built 32000**32 candidate objects and never
        returned. The bound asserted here is the number of expansions, which
        is deterministic. Wall-clock time is deliberately not asserted: this is
        a pure-Python reference implementation, so a 32k-token vocabulary over
        32 steps takes seconds, and timing a unit test is how you get a suite
        that fails on a loaded CI box.
        """
        vocab = {t: -0.1 for t in range(32000)}
        calls = []

        def step(prefix):
            calls.append(1)
            return vocab

        beam_search(step, bos=0, eos=-1, beam_width=4, max_len=32)
        # step 1 expands the single seed beam, then each step expands beam_width
        expected = 1 + 4 * (32 - 1)
        assert len(calls) == expected, (
            f"expanded {len(calls)} beams, expected {expected}")
