"""Projective visibility evidence between the two epochs.

Importing the package does not load torch; the vote imports it when it runs.
"""

from spatial_change.visibility.voting import (
    CrossEvidence,
    PointEvidence,
    PointState,
    VisibilityConfig,
    Vote,
    vote_epochs,
    vote_points,
)

__all__ = [
    "CrossEvidence",
    "PointEvidence",
    "PointState",
    "VisibilityConfig",
    "Vote",
    "vote_epochs",
    "vote_points",
]
