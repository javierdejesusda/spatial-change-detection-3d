"""The colors of the change types, shared by the HTML report and the GLB.

Both outputs read this one mapping, so a type has the same color in the
report badges, the embedded 3D view, and the exported scene. The hues stay
apart under the common kinds of color blindness, and each is dark enough to
read as a badge dot on a light page and bright enough on a dark one.
"""

from types import MappingProxyType

from spatial_change.domain import ChangeType

RGB = tuple[int, int, int]

# ADDED is green, REMOVED red, MOVED blue, and UNCERTAIN amber.
CHANGE_COLORS: MappingProxyType[ChangeType, RGB] = MappingProxyType(
    {
        ChangeType.ADDED: (47, 158, 68),
        ChangeType.REMOVED: (224, 49, 49),
        ChangeType.MOVED: (28, 126, 214),
        ChangeType.UNCERTAIN: (240, 140, 0),
    }
)


# Regions that only one capture saw are not changes, so they are gray.
UNOBSERVED_COLOR: RGB = (128, 134, 144)


def change_color_hex(kind: ChangeType) -> str:
    """Returns the color of a change type as a CSS hex string.

    Args:
        kind: The change type.

    Returns:
        Lowercase ``#rrggbb``.
    """
    red, green, blue = CHANGE_COLORS[kind]
    return f"#{red:02x}{green:02x}{blue:02x}"
