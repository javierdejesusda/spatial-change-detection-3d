"""Association of the objects that disappeared with those that appeared.

An object that moved is lifted twice, once in the BEFORE epoch where it
vanished and once in the AFTER epoch where it appeared. Each disappeared
object is paired with at most one appeared object by a minimum-cost
assignment over appearance, registered shape, and distance, and a pair whose
cost is too high is not made. A pair that fits as one rigid object and shows
a displacement or a rotation is MOVED; a pair that is in place, or whose
registration cannot be trusted, is UNCERTAIN. The objects left over are
REMOVED or ADDED.

Importing this module does not load open3d or torch.
"""

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from spatial_change.change.assignment import gated_assignment
from spatial_change.change.candidates import Candidate, ChangeCandidates
from spatial_change.change.lifting import LiftedObjects
from spatial_change.domain import (
    BoxModel,
    Change,
    ChangeType,
    Epoch,
    EvidenceCounts,
    ObjectInstance,
)
from spatial_change.geometry import (
    Registration,
    RegistrationConfig,
    register_objects,
    rotation_angle_deg,
)

_DEFAULT_LABEL = "object"
# Confidence of an unmatched object that no candidate supports.
_NO_SUPPORT = 0.5
# Upper bound of the confidence of a pair that is not confirmed as moved.
_UNCERTAIN_CEILING = 0.5


@dataclass(frozen=True)
class MatchConfig:
    """Parameters of the association. The defaults are provisional.

    They are placeholders until T22 calibrates them on real captures.
    Lengths are in the units of the world frame, meters after the alignment
    stage. The cost of a pair is the weighted sum of three terms that each
    lie in ``[0, 1]``.

    Attributes:
        w_appearance: Weight of ``1 - cosine`` of the two features, at
            least 0 and finite.
        w_shape: Weight of ``1 - fitness`` of the registration, at least 0
            and finite.
        w_distance: Weight of the distance between the box centers over
            ``distance_scale``, capped at 1; a weak prior that separates
            identical objects. At least 0 and finite.
        distance_scale: Distance at which the distance term saturates,
            positive and finite.
        unknown_cost: Value of the appearance term when a feature is
            missing and of the shape term when the registration failed, in
            ``[0, 1]``.
        max_extent_ratio: Pairs whose largest box extents differ by more
            than this factor are not registered and not made, at least 1 and
            finite. A partial view keeps about the largest extent of the
            whole, so this admits one and refuses a stool against a sofa.
        max_cost: Pairs costing this or more are not made, positive and
            finite.
        min_fitness: Fitness a registration needs for its pair to be made,
            in ``[0, 1]``; a pair that fits worse is two different shapes.
        min_displacement_m: Distance the disappeared box center travels
            under the registration from which a pair counts as moved, at
            least 0 and finite.
        min_rotation_deg: Rotation from which a pair counts as moved, in
            ``[0, 180]``.
        support_scale: Number of observations (free plus consistent votes)
            at which the evidence behind a change earns 63% of its
            confidence; fewer observations scale the confidence of every
            change down, positive and finite.
        candidate_overlap: Share of a candidate's points that must lie in an
            object's box for the candidate to count as its evidence, in
            ``(0, 1]``.
        registration: Settings of the shape registration.
    """

    w_appearance: float = 1.0
    w_shape: float = 1.0
    w_distance: float = 0.2
    distance_scale: float = 1.0
    unknown_cost: float = 0.5
    max_extent_ratio: float = 3.0
    max_cost: float = 0.9
    min_fitness: float = 0.85
    min_displacement_m: float = 0.05
    min_rotation_deg: float = 10.0
    support_scale: float = 500.0
    candidate_overlap: float = 0.5
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)

    def __post_init__(self) -> None:
        """Validates the parameters.

        Raises:
            ValueError: If a weight, length, or share is out of its range or
                not finite.
        """
        for name in ("w_appearance", "w_shape", "w_distance"):
            if not 0.0 <= getattr(self, name) < math.inf:
                raise ValueError(f"{name} must be >= 0 and finite")
        if not 0.0 < self.distance_scale < math.inf:
            raise ValueError("distance_scale must be positive and finite")
        if not 0.0 <= self.unknown_cost <= 1.0:
            raise ValueError("unknown_cost must be within [0, 1]")
        if not 1.0 <= self.max_extent_ratio < math.inf:
            raise ValueError("max_extent_ratio must be >= 1 and finite")
        if not 0.0 < self.max_cost < math.inf:
            raise ValueError("max_cost must be positive and finite")
        if not 0.0 <= self.min_fitness <= 1.0:
            raise ValueError("min_fitness must be within [0, 1]")
        if not 0.0 <= self.min_displacement_m < math.inf:
            raise ValueError("min_displacement_m must be >= 0 and finite")
        if not 0.0 <= self.min_rotation_deg <= 180.0:
            raise ValueError("min_rotation_deg must be within [0, 180]")
        if not 0.0 < self.support_scale < math.inf:
            raise ValueError("support_scale must be positive and finite")
        if not 0.0 < self.candidate_overlap <= 1.0:
            raise ValueError("candidate_overlap must be within (0, 1]")


_DEFAULT_CONFIG = MatchConfig()


@dataclass(frozen=True, eq=False)
class _Pair:
    """The cost of pairing two objects and the registration behind it.

    Attributes:
        cost: Weighted sum of the cost terms.
        registration: The registration of the disappeared object onto the
            appeared one, or None when the pair was gated, whether before
            or after the registration ran.
    """

    cost: float
    registration: Registration | None


def _center_distance(before: ObjectInstance, after: ObjectInstance) -> float:
    """Measures the distance between the centers of two boxes."""
    return float(np.linalg.norm(after.box.center - before.box.center))


def _appearance_term(
    before: ObjectInstance, after: ObjectInstance
) -> float | None:
    """Gives ``1 - cosine`` of two features, capped to ``[0, 1]``.

    Returns:
        The term, or None when a feature is missing.
    """
    if before.feature is None or after.feature is None:
        return None
    cosine = float(
        np.dot(before.feature, after.feature)
        / max(
            float(np.linalg.norm(before.feature))
            * float(np.linalg.norm(after.feature)),
            1e-12,
        )
    )
    return min(max(1.0 - cosine, 0.0), 1.0)


def _pair(
    before: ObjectInstance, after: ObjectInstance, config: MatchConfig
) -> _Pair:
    """Prices the pairing of a disappeared and an appeared object.

    The registration is run only when the appearance and distance terms
    alone leave the pair under ``max_cost`` and the largest box extents are
    comparable, since the shape term can only raise the cost. A pair whose
    registration fits worse than ``min_fitness`` cannot be made.

    Args:
        before: The disappeared object.
        after: The appeared object.
        config: Weights and gates.

    Returns:
        The cost and the registration, None when the pair was gated.
    """
    term = _appearance_term(before, after)
    appearance = config.unknown_cost if term is None else term
    distance = min(_center_distance(before, after) / config.distance_scale, 1.0)
    known = config.w_appearance * appearance + config.w_distance * distance
    unmatched = max(known + config.w_shape, config.max_cost)
    sizes = sorted((before.box.extent.max(), after.box.extent.max()))
    if (
        known >= config.max_cost
        or sizes[1] > config.max_extent_ratio * sizes[0]
    ):
        return _Pair(unmatched, None)
    registration = register_objects(
        before.points, after.points, config.registration
    )
    if not registration.succeeded:
        shape = config.unknown_cost
    elif registration.fitness < config.min_fitness:
        return _Pair(unmatched, None)
    else:
        shape = 1.0 - registration.fitness
    return _Pair(known + config.w_shape * shape, registration)


def _supporting(
    instance: ObjectInstance,
    candidates: tuple[Candidate, ...],
    config: MatchConfig,
) -> list[Candidate]:
    """Lists the candidates whose points lie mostly in an object's box.

    Args:
        instance: The object.
        candidates: Candidates of the object's epoch.
        config: Supplies ``candidate_overlap``.

    Returns:
        The candidates in their order.
    """
    return [
        c
        for c in candidates
        if instance.box.contains(c.points).mean() >= config.candidate_overlap
    ]


def _evidence(supporting: list[Candidate]) -> EvidenceCounts:
    """Sums the vote counts of candidates.

    Args:
        supporting: The candidates.

    Returns:
        The summed counts, zero for no candidate.
    """
    return EvidenceCounts(
        consistent=sum(c.evidence.consistent for c in supporting),
        free=sum(c.evidence.free for c in supporting),
        occluded=sum(c.evidence.occluded for c in supporting),
        unobserved=sum(c.evidence.unobserved for c in supporting),
    )


def _support_confidence(supporting: list[Candidate]) -> float:
    """Gives the changed fraction of candidates, weighted by their points.

    Args:
        supporting: The candidates.

    Returns:
        The weighted mean, or a neutral value for no candidate.
    """
    weights = np.array([c.point_count for c in supporting], np.float64)
    if not len(supporting) or weights.sum() == 0.0:
        return _NO_SUPPORT
    fractions = np.array([c.changed_fraction for c in supporting])
    return float(np.average(fractions, weights=weights))


def _support_factor(evidence: EvidenceCounts, config: MatchConfig) -> float:
    """Gives how far the observations of a location back a change.

    Args:
        evidence: The summed counts of the change.
        config: Supplies ``support_scale``.

    Returns:
        ``1 - exp(-n / support_scale)`` for the ``n`` observations that saw
        the location, free or consistent; 0 for none.
    """
    seen = evidence.free + evidence.consistent
    return 1.0 - math.exp(-seen / config.support_scale)


def _check_epochs(lifted: LiftedObjects) -> None:
    """Checks that the instances come from the epoch of their side.

    Args:
        lifted: The lifted instances.

    Raises:
        ValueError: If a disappeared instance is not from the BEFORE epoch
            or an appeared one not from the AFTER epoch.
    """
    for side, epoch, items in (
        ("disappeared", Epoch.BEFORE, lifted.disappeared),
        ("appeared", Epoch.AFTER, lifted.appeared),
    ):
        for item in items:
            if item.epoch != epoch:
                raise ValueError(
                    f"{side} instance {item.object_id} is from epoch "
                    f"{item.epoch.value}, expected {epoch.value}"
                )


def _moved_or_uncertain(
    before: ObjectInstance,
    after: ObjectInstance,
    pair: _Pair,
    evidence: EvidenceCounts,
    config: MatchConfig,
    change_id: str,
) -> Change:
    """Builds the change of a matched pair.

    The pair is MOVED when the disappeared box center travels at least
    ``min_displacement_m`` under the recovered transform, or the rotation
    reaches ``min_rotation_deg``. The gap between the two box centers is
    not used for this decision, because a different view of one object also
    opens it; the reported ``displacement_m`` is still that gap (the
    contract of the spatial relations stage). Any other pair is UNCERTAIN:
    it is in place, so only its look changed, or the registration cannot
    vouch that it is one rigid object.

    Args:
        before: The disappeared object.
        after: The appeared object.
        pair: The pairing, with its registration.
        evidence: The summed evidence of both objects.
        config: Thresholds.
        change_id: Identifier of the change.

    Returns:
        The change.
    """
    registration = pair.registration
    displacement = _center_distance(before, after)
    margin = 1.0 - pair.cost / config.max_cost
    kind = ChangeType.UNCERTAIN
    rotation = None
    confidence = _UNCERTAIN_CEILING * margin
    if registration is not None and registration.succeeded:
        angle = rotation_angle_deg(registration.transform.rotation)
        travel = float(
            np.linalg.norm(
                registration.transform.apply(before.box.center[None])[0]
                - before.box.center
            )
        )
        if (
            travel >= config.min_displacement_m
            or angle >= config.min_rotation_deg
        ):
            kind = ChangeType.MOVED
            rotation = angle
            confidence = 0.5 * registration.fitness + 0.5 * margin
    return Change(
        change_id=change_id,
        type=kind,
        label=_pair_label(before, after),
        description="",
        confidence=confidence * _support_factor(evidence, config),
        before_box=BoxModel.from_box(before.box),
        after_box=BoxModel.from_box(after.box),
        displacement_m=displacement,
        rotation_deg=rotation,
        evidence=evidence,
    )


def _pair_label(before: ObjectInstance, after: ObjectInstance) -> str:
    """Picks the label of a matched pair.

    Args:
        before: The BEFORE instance.
        after: The AFTER instance.

    Returns:
        The label with the higher score, the BEFORE one on a tie and a
        label without a score ranking lowest; the only label if just one
        side has one, and the default label if neither does.
    """
    named = [(i.label, i.label_score) for i in (before, after) if i.label]
    if not named:
        return _DEFAULT_LABEL
    label, _ = max(
        named, key=lambda pair: -math.inf if pair[1] is None else pair[1]
    )
    return str(label)


def _single(
    item: ObjectInstance,
    supporting: list[Candidate],
    kind: ChangeType,
    change_id: str,
    config: MatchConfig,
) -> Change:
    """Builds the change of an object that found no partner.

    Args:
        item: The object.
        supporting: The candidates that support it.
        kind: REMOVED for a disappeared object, ADDED for an appeared one.
        change_id: Identifier of the change.
        config: Supplies ``support_scale``.

    Returns:
        The change.
    """
    box = BoxModel.from_box(item.box)
    removed = kind is ChangeType.REMOVED
    evidence = _evidence(supporting)
    return Change(
        change_id=change_id,
        type=kind,
        label=item.label or _DEFAULT_LABEL,
        description="",
        confidence=_support_confidence(supporting)
        * _support_factor(evidence, config),
        before_box=box if removed else None,
        after_box=None if removed else box,
        displacement_m=None,
        rotation_deg=None,
        evidence=evidence,
    )


def associate(
    lifted: LiftedObjects,
    candidates: ChangeCandidates,
    config: MatchConfig = _DEFAULT_CONFIG,
) -> tuple[Change, ...]:
    """Pairs the disappeared objects with the appeared ones and classifies.

    The cost of every pair of a disappeared and an appeared object is the
    weighted sum of the cosine distance of their features, the misfit of
    registering the smaller cloud onto the larger one, and a capped distance
    between their boxes. A missing feature or a failed registration counts
    as ``unknown_cost``. The pairs are those of a minimum-cost assignment
    that never makes a pair costing ``max_cost`` or more (see
    ``MatchConfig``). The confidence of every change is scaled down when few
    observations saw its location.

    Args:
        lifted: The object instances of both directions; give them features
            with ``describe_instances`` to use the appearance term.
        candidates: The candidates that were lifted, whose evidence supports
            the changes.
        config: Weights, gates, and thresholds.

    Returns:
        The changes: first the pairs in the order of their disappeared
        object, then the removed objects, then the added ones, each in the
        order of ``lifted``. Their ids run ``change-0``, ``change-1``, and
        so on.

    Raises:
        ValueError: If an instance is from the wrong epoch.
    """
    _check_epochs(lifted)
    before, after = lifted.disappeared, lifted.appeared
    pairs = [[_pair(b, a, config) for a in after] for b in before]
    costs = np.array([[p.cost for p in row] for row in pairs]).reshape(
        len(before), len(after)
    )
    rows, cols = gated_assignment(costs, config.max_cost)
    support_before = [
        _supporting(item, candidates.disappeared, config) for item in before
    ]
    support_after = [
        _supporting(item, candidates.appeared, config) for item in after
    ]
    changes: list[Change] = []
    change_ids = (f"change-{n}" for n in itertools.count())
    for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
        changes.append(
            _moved_or_uncertain(
                before[row],
                after[col],
                pairs[row][col],
                _evidence(support_before[row] + support_after[col]),
                config,
                next(change_ids),
            )
        )
    matched_rows, matched_cols = set(rows.tolist()), set(cols.tolist())
    unmatched = (
        (ChangeType.REMOVED, before, support_before, matched_rows),
        (ChangeType.ADDED, after, support_after, matched_cols),
    )
    for kind, items, supports, matched in unmatched:
        for index, item in enumerate(items):
            if index not in matched:
                changes.append(
                    _single(
                        item,
                        supports[index],
                        kind,
                        next(change_ids),
                        config,
                    )
                )
    return tuple(changes)
