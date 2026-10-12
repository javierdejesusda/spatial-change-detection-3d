"""Label vocabularies of the open-vocabulary labeler.

Names are plain nouns, so they fit the prompt template of the labeler.
"""

INDOOR_OBJECTS: tuple[str, ...] = (
    "backpack",
    "bag",
    "ball",
    "basket",
    "bed",
    "bench",
    "book",
    "bookshelf",
    "bottle",
    "bowl",
    "box",
    "bucket",
    "cabinet",
    "chair",
    "clock",
    "counter",
    "crate",
    "cup",
    "desk",
    "dresser",
    "drum",
    "fan",
    "guitar",
    "keyboard",
    "ladder",
    "lamp",
    "laptop",
    "mirror",
    "monitor",
    "mug",
    "pillow",
    "plant",
    "printer",
    "refrigerator",
    "shelf",
    "shoe",
    "sofa",
    "speaker",
    "stool",
    "suitcase",
    "table",
    "television",
    "toolbox",
    "toy",
    "trash can",
    "umbrella",
    "vase",
    "wardrobe",
)
"""Common indoor objects, alphabetical; includes every synthetic class."""

FURNITURE: tuple[str, ...] = (
    "bed",
    "bookshelf",
    "cabinet",
    "counter",
    "desk",
    "dresser",
    "refrigerator",
    "shelf",
    "sofa",
    "stool",
    "table",
    "wardrobe",
)
"""Furniture that rarely moves, the candidates for landmarks.

Chairs are left out because they move often.
"""

STRUCTURE: tuple[str, ...] = ("ceiling", "door", "floor", "wall", "window")
"""Room structure, used as negatives when labeling landmark proposals."""
