"""Run the corpus matrix as part of the normal test run, when configured.

Skips cleanly when `tests/corpora.json` is absent, because the corpora are
installed packages rather than fixtures the repository ships.
"""
import pytest

from tests.corpus_runner import CONFIG, main


@pytest.mark.skipif(not CONFIG.exists(), reason="tests/corpora.json not configured")
def test_no_metric_moved_without_explanation():
    assert main([]) == 0, "a recorded metric changed; see the diff above"
