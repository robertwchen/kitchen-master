import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baseline_detector import classify


def test_clear_legal():
    assert classify(150, 130, fault_threshold_px=2, uncertain_margin_px=8) == "legal"


def test_clear_fault():
    assert classify(150, 165, fault_threshold_px=2, uncertain_margin_px=8) == "fault"


def test_borderline_uncertain():
    assert classify(150, 147, fault_threshold_px=2, uncertain_margin_px=8) == "uncertain"


def test_missing_line():
    assert classify(None, 147, fault_threshold_px=2, uncertain_margin_px=8) == "uncertain"


def test_missing_foot():
    assert classify(150, None, fault_threshold_px=2, uncertain_margin_px=8) == "uncertain"


def test_exactly_at_line():
    # foot_bottom == line_y → gap = 0, inside margin → uncertain
    assert classify(150, 150, fault_threshold_px=2, uncertain_margin_px=8) == "uncertain"
