"""Placeholder interface for volley event detection. Not implemented in Phase 1."""


def detect_volley_events(frames: list, fps: float, cfg: dict | None = None) -> list[dict]:
    """
    Detect frame indices where a volley event occurs near the kitchen line.

    Args:
        frames: list of BGR numpy arrays in temporal order
        fps:    video frame rate
        cfg:    optional config dict

    Returns:
        list of dicts, each with:
            frame_index  int
            timestamp_s  float
            confidence   float 0–1

    Phase 1 status: not implemented. Raises NotImplementedError.
    """
    raise NotImplementedError("Event detection is scheduled for Phase 2.")
