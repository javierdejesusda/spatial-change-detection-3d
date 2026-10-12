import json

import numpy as np
import pytest
from facts import make_fact, moved_fact, near
from numpy.typing import NDArray

from spatial_change.domain import ChangeType
from spatial_change.perception.fake import FakeVlm
from spatial_change.ports import ChangeCrops, Describer
from spatial_change.semantics.describers import (
    LocalVlmConfig,
    LocalVlmDescriber,
    TemplateDescriber,
)
from spatial_change.semantics.relations import ChangeFact
from spatial_change.semantics.template import template_sentence


def test_the_template_describer_satisfies_the_port() -> None:
    assert isinstance(TemplateDescriber(), Describer)


def test_it_returns_the_template_sentence_and_ignores_the_crops() -> None:
    fact = moved_fact()
    crops = ChangeCrops(before=(np.zeros((4, 4, 3), np.uint8),))

    assert TemplateDescriber().describe(fact, crops) == template_sentence(fact)
    assert TemplateDescriber().describe(fact, ChangeCrops()) == (
        template_sentence(fact)
    )


def test_unload_does_nothing() -> None:
    TemplateDescriber().unload()


def _crop(value: int, size: int = 4) -> NDArray[np.uint8]:
    return np.full((size, size, 3), value, np.uint8)


def _reply(text: str) -> str:
    return json.dumps({"description": text})


FALLBACK = TemplateDescriber()


def _describer(
    replies: list[str], config: LocalVlmConfig | None = None
) -> tuple[LocalVlmDescriber, FakeVlm]:
    vlm = FakeVlm(replies)
    return LocalVlmDescriber(vlm, FALLBACK, config), vlm


def test_the_local_describer_satisfies_the_port() -> None:
    describer, _ = _describer([])

    assert isinstance(describer, Describer)


def test_a_valid_json_reply_is_the_description() -> None:
    describer, vlm = _describer([_reply("The chair slid left by 30 cm.")])

    text = describer.describe(moved_fact(), ChangeCrops())

    assert text == "The chair slid left by 30 cm."
    assert len(vlm.calls) == 1


@pytest.mark.parametrize(
    "wrapper",
    [
        "```json\nPAYLOAD\n```",
        "Sure! PAYLOAD",
        "  PAYLOAD  \n",
        "<think>\n\n</think>\n\nPAYLOAD",
    ],
)
def test_json_inside_other_text_is_found(wrapper: str) -> None:
    reply = wrapper.replace("PAYLOAD", _reply("The chair slid to the left."))
    describer, vlm = _describer([reply])

    text = describer.describe(moved_fact(), ChangeCrops())

    assert text == "The chair slid to the left."
    assert len(vlm.calls) == 1


def test_the_description_is_stripped() -> None:
    describer, _ = _describer([_reply("  The chair slid left.  ")])

    assert describer.describe(moved_fact(), ChangeCrops()) == (
        "The chair slid left."
    )


@pytest.mark.parametrize(
    "bad",
    [
        "not json at all",
        "",
        "{broken",
        "[]",
        json.dumps({"text": "The chair slid left."}),
        json.dumps({"description": 3}),
        json.dumps({"description": ""}),
        json.dumps({"description": "   "}),
        json.dumps({"description": "x", "extra": 1}),
        json.dumps({"description": "word " * 100}),
    ],
)
def test_an_unusable_reply_is_retried_once_with_a_corrective_prompt(
    bad: str,
) -> None:
    describer, vlm = _describer([bad, _reply("The chair slid left.")])

    text = describer.describe(moved_fact(), ChangeCrops())

    assert text == "The chair slid left."
    assert len(vlm.calls) == 2
    first, second = vlm.calls[0][1], vlm.calls[1][1]
    assert second.startswith(first)
    assert "not valid" in second[len(first) :]


def test_two_unusable_replies_fall_back_to_the_template() -> None:
    describer, vlm = _describer(["nope", "still nope"])
    fact = moved_fact()

    text = describer.describe(fact, ChangeCrops())

    assert text == template_sentence(fact)
    assert len(vlm.calls) == 2


def test_a_contradicting_sentence_is_replaced_without_a_retry() -> None:
    describer, vlm = _describer([_reply("The chair moved to the right.")])
    fact = moved_fact()

    text = describer.describe(fact, ChangeCrops())

    assert text == template_sentence(fact)
    assert len(vlm.calls) == 1


def test_a_far_number_is_replaced() -> None:
    describer, _ = _describer([_reply("The chair moved 3 m to the left.")])
    fact = moved_fact()

    assert describer.describe(fact, ChangeCrops()) == template_sentence(fact)


def test_the_fallback_receives_the_crops() -> None:
    class Recording:
        def __init__(self) -> None:
            self.crops: list[ChangeCrops] = []

        def describe(self, fact: ChangeFact, crops: ChangeCrops) -> str:
            self.crops.append(crops)
            return "fallback"

        def unload(self) -> None:
            pass

    fallback = Recording()
    vlm = FakeVlm(["bad", "bad"])
    crops = ChangeCrops(before=(_crop(1),))

    text = LocalVlmDescriber(vlm, fallback).describe(moved_fact(), crops)

    assert text == "fallback"
    assert fallback.crops == [crops]


PROMPT_FACTS = [
    (
        moved_fact(
            support_before="floor",
            support_after="shelf",
            near_before=[near("lamp", "next to", 0.2)],
            near_after=[near("sofa", "near", 0.7)],
        ),
        [
            "moved",
            "chair",
            "to the left",
            "table",
            "about 30 cm",
            "about 70 degrees",
            "floor",
            "shelf",
            "next to the lamp",
            "near the sofa",
        ],
    ),
    (
        make_fact(ChangeType.ADDED, label="drum", support_after="table"),
        ["added", "drum", "table"],
    ),
    (
        make_fact(ChangeType.REMOVED, label="crate", support_before="floor"),
        ["removed", "crate", "floor"],
    ),
]


@pytest.mark.parametrize(("fact", "expected"), PROMPT_FACTS)
def test_the_prompt_states_the_fact(
    fact: ChangeFact, expected: list[str]
) -> None:
    describer, vlm = _describer([_reply("x")])

    describer.describe(fact, ChangeCrops())

    prompt = vlm.calls[0][1]
    for part in expected:
        assert part in prompt
    assert '{"description":' in prompt
    assert "JSON" in prompt


def test_the_prompt_omits_what_the_fact_lacks() -> None:
    describer, vlm = _describer([_reply("x")])
    fact = make_fact(ChangeType.ADDED, label="drum")

    describer.describe(fact, ChangeCrops())

    prompt = vlm.calls[0][1]
    for absent in ("Direction", "Distance", "Rotation", "Moved toward"):
        assert absent not in prompt


def test_the_images_are_the_before_crops_then_the_after_crops() -> None:
    describer, vlm = _describer([_reply("x")])
    crops = ChangeCrops(before=(_crop(10), _crop(20)), after=(_crop(30),))

    describer.describe(moved_fact(), crops)

    images, prompt = vlm.calls[0]
    assert [int(image[0, 0, 0]) for image in images] == [10, 30]
    assert "first image" in prompt


def test_up_to_the_configured_number_of_crops_per_side_is_sent() -> None:
    describer, vlm = _describer(
        [_reply("x")], LocalVlmConfig(max_crops_per_side=2)
    )
    crops = ChangeCrops(
        before=(_crop(1), _crop(2), _crop(3)), after=(_crop(4), _crop(5))
    )

    describer.describe(moved_fact(), crops)

    images = vlm.calls[0][0]
    assert [int(image[0, 0, 0]) for image in images] == [1, 2, 4, 5]


def test_an_added_object_sends_only_after_crops() -> None:
    describer, vlm = _describer([_reply("x")])
    crops = ChangeCrops(after=(_crop(7),))

    describer.describe(make_fact(ChangeType.ADDED, label="drum"), crops)

    images, prompt = vlm.calls[0]
    assert len(images) == 1
    assert "after" in prompt


def test_without_crops_no_image_is_sent_or_mentioned() -> None:
    describer, vlm = _describer([_reply("x")])

    describer.describe(moved_fact(), ChangeCrops())

    images, prompt = vlm.calls[0]
    assert images == ()
    assert "image" not in prompt.lower()


def test_the_corrective_retry_sends_the_same_images() -> None:
    describer, vlm = _describer(["bad", _reply("x")])
    crops = ChangeCrops(before=(_crop(1),), after=(_crop(2),))

    describer.describe(moved_fact(), crops)

    assert len(vlm.calls[0][0]) == len(vlm.calls[1][0]) == 2


def test_unload_is_forwarded_to_the_model() -> None:
    describer, vlm = _describer([])

    describer.unload()

    assert vlm.unload_count == 1


@pytest.mark.parametrize("count", [0, -1])
def test_a_config_needs_at_least_one_crop_per_side(count: int) -> None:
    with pytest.raises(ValueError, match="max_crops_per_side"):
        LocalVlmConfig(max_crops_per_side=count)


def test_the_description_length_limit_is_not_configurable() -> None:
    with pytest.raises(TypeError):
        LocalVlmConfig(max_description_chars=100)  # type: ignore[call-arg]


def _describe(description: str) -> tuple[str, FakeVlm]:
    vlm = FakeVlm([json.dumps({"description": description})])
    describer = LocalVlmDescriber(vlm, TemplateDescriber())
    return describer.describe(moved_fact(), ChangeCrops()), vlm


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("The chair moved.\nIt is by the sofa. Extra.", "The chair moved."),
        ("The chair moved.\nLine two. Extra.", "The chair moved."),
        (
            "The chair   moved\n to the left.",
            "The chair moved to the left.",
        ),
        (
            "The chair moved to the left! Then it stopped.",
            "The chair moved to the left!",
        ),
        (
            "The chair moved 0.3 m to the left.",
            "The chair moved 0.3 m to the left.",
        ),
        ("\tThe chair moved.  ", "The chair moved."),
    ],
)
def test_the_description_is_one_collapsed_sentence(
    reply: str, expected: str
) -> None:
    text, vlm = _describe(reply)

    assert text == expected
    assert len(vlm.calls) == 1


def test_only_the_first_sentence_is_validated() -> None:
    text, _ = _describe("The chair moved. It moved to the right.")

    assert text == "The chair moved."


def test_a_bad_first_sentence_still_falls_back() -> None:
    text, _ = _describe("The chair moved to the right. Fine.")

    assert text == template_sentence(moved_fact())


@pytest.mark.parametrize(
    "sentence",
    [
        "The chair moved approx. 30 cm to the left.",
        "The chair moved 30 cm, i.e. a short way to the left.",
        "The chair moved to the left, e.g. toward the table.",
        "The chair moved approx. The distance is 30 cm to the left.",
        "The chair moved to the left etc. The table stayed.",
    ],
)
def test_the_sentence_is_not_cut_at_an_abbreviation(sentence: str) -> None:
    assert _describe(sentence)[0] == sentence


def test_a_sentence_end_before_a_lowercase_word_is_not_a_cut() -> None:
    assert _describe("The chair moved. then it moved to the left.")[0] == (
        "The chair moved. then it moved to the left."
    )


def test_a_real_sentence_end_still_cuts() -> None:
    assert _describe("The chair moved left. The table stayed.")[0] == (
        "The chair moved left."
    )
