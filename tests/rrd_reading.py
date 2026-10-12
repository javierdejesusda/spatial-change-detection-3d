"""Reading a saved Rerun recording back, for the visualization tests."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rerun as rr

from spatial_change.visualization.layers import EPOCH_TIMELINE


@dataclass(frozen=True, eq=False)
class Row:
    """One logged value of one component.

    Attributes:
        step: The ``epoch`` timeline step, or None for static data.
        value: The component value as Python lists.
    """

    step: int | None
    value: Any


class Rrd:
    """The contents of a ``.rrd`` file.

    Args:
        path: The recording to read.
    """

    def __init__(self, path: Path) -> None:
        reader = rr.chunk.RrdReader(str(path))
        self.blueprints = reader.blueprints()
        self._chunks = list(reader.store().stream().to_chunks())
        self._blueprint_chunks = [
            chunk
            for entry in self.blueprints
            for chunk in reader.store(store=entry).stream().to_chunks()
        ]

    @property
    def entities(self) -> set[str]:
        """Entity paths with data, leading slash included."""
        return {
            c.entity_path
            for c in self._chunks
            if not c.entity_path.startswith("/__properties")
        }

    def views(self) -> list[tuple[str, str, str]]:
        """Lists the views of the blueprint.

        Returns:
            One ``(class, name, origin)`` triple per view, such as
            ``("3D", "Scene", "world")``.
        """
        found = []
        for chunk in self._blueprint_chunks:
            parts = chunk.entity_path.split("/")
            if len(parts) != 3 or parts[1] != "view":
                continue
            batch = chunk.to_record_batch()
            columns = ("class_identifier", "display_name", "space_origin")
            class_id, name, origin = (
                batch.column(f"ViewBlueprint:{c}").to_pylist()[0][0]
                for c in columns
            )
            found.append((class_id, name, origin))
        return found

    def blueprint_timeline(self) -> str | None:
        """Returns the timeline the blueprint's time panel shows.

        Returns:
            The timeline name, or None if the panel names none.
        """
        for chunk in self._blueprint_chunks:
            if chunk.entity_path == "/time_panel":
                batch = chunk.to_record_batch()
                column = "TimePanelBlueprint:timeline"
                if column in batch.schema.names:
                    return str(batch.column(column).to_pylist()[0][0])
        return None

    def rows(self, entity: str, component: str) -> list[Row]:
        """Collects the values of one component of one entity.

        Args:
            entity: Entity path with a leading slash.
            component: Component column such as ``Boxes3D:colors``.

        Returns:
            The values in log order; empty if there are none.
        """
        found: list[Row] = []
        for chunk in self._chunks:
            if chunk.entity_path != entity:
                continue
            batch = chunk.to_record_batch()
            if component not in batch.schema.names:
                continue
            steps = (
                batch.column(EPOCH_TIMELINE).to_pylist()
                if EPOCH_TIMELINE in batch.schema.names
                else [None] * batch.num_rows
            )
            values = batch.column(component).to_pylist()
            found.extend(Row(s, v) for s, v in zip(steps, values, strict=True))
        return found

    def at(self, entity: str, component: str, step: int | None) -> Any:
        """Returns the one value logged at a step.

        Args:
            entity: Entity path with a leading slash.
            component: Component column.
            step: Timeline step, or None for static data.

        Returns:
            The value.

        Raises:
            AssertionError: If the step does not hold exactly one value.
        """
        values = [
            r.value for r in self.rows(entity, component) if r.step == step
        ]
        assert len(values) == 1, (entity, component, step, len(values))
        return values[0]


def rgba(packed: int) -> tuple[int, int, int, int]:
    """Unpacks a Rerun color.

    Args:
        packed: The color as ``0xRRGGBBAA``.

    Returns:
        Red, green, blue, and alpha bytes.
    """
    return (
        (packed >> 24) & 255,
        (packed >> 16) & 255,
        (packed >> 8) & 255,
        packed & 255,
    )
