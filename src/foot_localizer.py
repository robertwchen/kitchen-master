"""Placeholder interface for foot localization. Not implemented in Phase 1."""


def localize_foot(frame, cfg: dict | None = None) -> dict | None:
    """
    Locate the player's foot region in a single frame.

    Args:
        frame: BGR numpy array
        cfg:   optional config dict (thresholds, color ranges, etc.)

    Returns:
        dict with keys:
            bbox          (x, y, w, h) bounding box of the foot region
            foot_bottom_y  y-coordinate of the lowest foot pixel
            confidence     float 0–1
        or None if no foot detected.

    Phase 1 status: not implemented. Raises NotImplementedError.
    """
    raise NotImplementedError("Foot localization is scheduled for Phase 2.")
