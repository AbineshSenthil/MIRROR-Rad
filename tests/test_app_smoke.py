"""Drive the real Streamlit app headlessly (skipped without the checkpoint)."""

import pytest

from conftest import ROOT, checkpoint_available

pytestmark = pytest.mark.skipif(not checkpoint_available(), reason="needs `git lfs pull`")


def _app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.run()
    assert not at.exception
    return at


def _markdown(at) -> str:
    return " ".join(item.value for item in at.markdown)


@pytest.mark.parametrize("page", ["Results & Benchmarks", "Study Design", "System Readiness"])
def test_pages_render_without_errors(page):
    at = _app()
    at.radio(key="primary_workspace").set_value(page).run()
    assert not at.exception
    assert not at.error


def test_demo_supported_summary_is_routine():
    at = _app()
    at.toggle(key="audit_demo_mode").set_value(True).run()
    at.checkbox(key="audit_acknowledgement").check().run()
    next(b for b in at.button if b.label.startswith("Run reciprocal evidence audit")).click().run()
    assert not at.exception
    text = _markdown(at)
    assert "Routine review queue" in text
    assert "Items to check" in text


def test_dataset_lab_runs_showcase_and_opens_a_case():
    at = _app()
    at.radio(key="primary_workspace").set_value("Dataset Audit").run()
    at.toggle(key="dataset_demo_mode").set_value(True).run()
    at.checkbox(key="batch_acknowledgement").check().run()
    next(b for b in at.button if b.label.startswith("Run reciprocal audit on")).click().run()
    assert not at.exception
    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics["Completed"] == "9"
    assert metrics["Failed"] == "0"
    # The per-case drill-down rendered (this crashed before the fix).
    assert "Audit outcome" in _markdown(at)
