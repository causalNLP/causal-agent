"""Tests for the one-shot design-context extraction in the query interpreter.

The interpreter must extract semantic design facts (mediator, encouragement
design, treatment / placebo periods) from the query + description only, and
must never guess: anything the text does not support stays None.
"""

import unittest
from unittest.mock import MagicMock

from cais.components.query_interpreter import identify_design_context
from cais.models import LLMDesignContext


COLUMNS = ["t", "m", "y", "year", "state", "encourage"]

CATEGORIES = {c: "binary" if c == "t" else "continuous_numeric" for c in COLUMNS}


def _llm_returning(result):
    fake = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = result
    fake.with_structured_output.return_value = structured
    return fake


def _llm_failing():
    fake = MagicMock()
    fake.with_structured_output.side_effect = RuntimeError("llm down")
    return fake


class TestIdentifyDesignContext(unittest.TestCase):

    def test_extracts_supported_fields(self):
        llm = _llm_returning(LLMDesignContext(
            mediator_variable="m",
            treatment_period_start="1988",
            placebo_period_start="1986",
            is_encouragement_design=True,
            reasoning="text describes an ad-exposure pathway",
        ))

        design = identify_design_context(
            llm, "Did the campaign raise sales?", "Encouragement via ads; ad exposure mediates.",
            COLUMNS, CATEGORIES, "t", "y",
        )

        self.assertEqual(design["mediator_variable"], "m")
        self.assertEqual(design["treatment_period_start"], 1988)
        self.assertEqual(design["placebo_period_start"], 1986)
        self.assertTrue(design["is_encouragement_design"])

    def test_unsupported_columns_are_rejected(self):
        llm = _llm_returning(LLMDesignContext(
            mediator_variable="not_a_column",
            treatment_period_start="policy begins soon",  # not numeric
            placebo_period_start=None,
            is_encouragement_design=None,
            reasoning="hallucinated",
        ))

        design = identify_design_context(
            llm, "Effect of t on y?", "No specifics.", COLUMNS, CATEGORIES, "t", "y"
        )

        self.assertIsNone(design["mediator_variable"])
        self.assertIsNone(design["treatment_period_start"])
        self.assertIsNone(design["placebo_period_start"])
        self.assertIsNone(design["is_encouragement_design"])

    def test_mediator_cannot_be_treatment_or_outcome(self):
        llm = _llm_returning(LLMDesignContext(
            mediator_variable="y",
            treatment_period_start=None,
            placebo_period_start=None,
            is_encouragement_design=None,
            reasoning="bad extraction",
        ))

        design = identify_design_context(
            llm, "Effect of t on y?", "text", COLUMNS, CATEGORIES, "t", "y"
        )

        self.assertIsNone(design["mediator_variable"])

    def test_llm_failure_returns_all_none(self):
        design = identify_design_context(
            _llm_failing(), "q", "d", COLUMNS, CATEGORIES, "t", "y"
        )
        self.assertEqual(design, {
            "mediator_variable": None,
            "treatment_period_start": None,
            "placebo_period_start": None,
            "is_encouragement_design": None,
        })

    def test_no_llm_returns_all_none_without_calling(self):
        llm = MagicMock()
        design = identify_design_context(None, "q", "d", COLUMNS, CATEGORIES, "t", "y")
        llm.with_structured_output.assert_not_called()
        self.assertIsNone(design["mediator_variable"])


if __name__ == "__main__":
    unittest.main()
