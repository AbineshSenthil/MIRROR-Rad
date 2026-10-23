"""Shared fixtures.

Most tests are pure Python and run anywhere.  Tests that need the fine-tuned
checkpoint use the ``auditor`` fixture, which skips cleanly when the file is
still a Git LFS pointer (for example in CI, which checks out without LFS).
Run ``git lfs pull`` locally to enable them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CHECKPOINT = ROOT / "artifacts" / "mirror-rad-clinical-nli" / "best" / "model.safetensors"
ARTIFACT = ROOT / "artifacts" / "mirror-rad-system" / "tuned_verifier"
CALIBRATOR = ROOT / "artifacts" / "mirror-rad-risk-calibrator.pkl"


def checkpoint_available() -> bool:
    """True when the real weights are present rather than an LFS pointer file."""

    return CHECKPOINT.is_file() and CHECKPOINT.stat().st_size > 10_000_000


@pytest.fixture(scope="session")
def auditor():
    if not checkpoint_available():
        pytest.skip("needs the fine-tuned checkpoint: run `git lfs pull`")
    from radtrace.bidirectional import BidirectionalAuditor, ReportRiskCalibrator
    from radtrace.verifier import RadTraceVerifier

    verifier = RadTraceVerifier.load_artifact(ARTIFACT, device_override="cpu")
    assert verifier.nli_predictor.available, verifier.nli_predictor.load_error
    calibrator = ReportRiskCalibrator.load(CALIBRATOR)
    return BidirectionalAuditor(verifier, calibrator=calibrator)
