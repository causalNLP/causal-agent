"""User-editable gate-threshold config: loader, validation, application."""

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from cais.config import load_thresholds, THRESHOLDS_FILE_ENV
from cais.components.decision_tree_v2 import (
    ADJUSTABLE_PARAMS,
    build_default_checks,
    select_method_v2,
)


def _write_thresholds(data: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh)
    return path


class TestLoadThresholds(unittest.TestCase):

    def tearDown(self):
        os.environ.pop(THRESHOLDS_FILE_ENV, None)

    def test_no_file_means_no_overrides(self):
        os.environ[THRESHOLDS_FILE_ENV] = "definitely/not/a/file.json"
        self.assertEqual(load_thresholds(), {})

    def test_reads_the_env_selected_file(self):
        path = _write_thresholds({"iv_relevance": {"f_threshold": 7}})
        os.environ[THRESHOLDS_FILE_ENV] = path
        self.assertEqual(load_thresholds(), {"iv_relevance": {"f_threshold": 7}})

    def test_malformed_json_raises_loudly(self):
        path = _write_thresholds({"iv_relevance": {"f_threshold": 7}})
        content = open(path).read()
        broken = content[:-1] + "trailing garbage"
        with open(path, "w") as fh:
            fh.write(broken)
        os.environ[THRESHOLDS_FILE_ENV] = path
        with self.assertRaises(ValueError):
            load_thresholds()

    def test_wrong_shape_raises_loudly(self):
        path = _write_thresholds({"iv_relevance": 10})
        os.environ[THRESHOLDS_FILE_ENV] = path
        with self.assertRaises(ValueError):
            load_thresholds()

    def test_underscore_keys_are_reserved_for_documentation(self):
        path = _write_thresholds({"_note": {"text": "my plan"}, "iv_relevance": {"f_threshold": 15}})
        os.environ[THRESHOLDS_FILE_ENV] = path
        self.assertEqual(load_thresholds(), {"iv_relevance": {"f_threshold": 15}})


class TestThresholdValidation(unittest.TestCase):

    def _base_props(self):
        return dict(
            treatment_variable="t",
            outcome_variable="y",
            treatment_variable_type="binary",
            covariates=["x1"],
            instrument_variable=None,
            mediator_variable=None,
            running_variable="r",
            cutoff_value=0,
            time_variable=None,
            group_variable="unit",
            is_rct=False,
            has_temporal_structure=False,
        )

    def test_unknown_check_name_raises(self):
        with self.assertRaises(ValueError) as ctx:
            build_default_checks(self._base_props(), None, "desc", None,
                                 thresholds={"positivty": {"alpha": 0.1}})
        self.assertIn("positivty", str(ctx.exception))

    def test_unknown_parameter_key_raises(self):
        with self.assertRaises(ValueError) as ctx:
            build_default_checks(self._base_props(), None, "desc", None,
                                 thresholds={"iv_relevance": {"f_stat_threshold": 12}})
        self.assertIn("f_stat_threshold", str(ctx.exception))

    def test_documented_keys_pass(self):
        checks = build_default_checks(
            self._base_props(), None, "desc", None,
            thresholds={name: {} for name in ADJUSTABLE_PARAMS},
        )
        self.assertIn("positivity", checks)


class TestAppliedThresholds(unittest.TestCase):

    def _positivity_df(self):
        # Treatment and control propensity-score ranges overlap, but ~25% of
        # the propensity scores leave [0.1, 0.9], so the check fails its
        # default extreme-share cap yet passes once it is loosened.
        x1 = np.linspace(-3, 3, 200)
        rng = np.random.default_rng(0)
        return pd.DataFrame({"t": (x1 + rng.normal(0, 1.5, size=200) > 0).astype(int), "x1": x1})

    def test_positivity_verdict_changes_with_max_extreme_pct(self):
        df = self._positivity_df()
        base = build_default_checks(
            {"treatment_variable": "t", "outcome_variable": "y", "covariates": ["x1"]},
            df, "desc", None, thresholds=None,
        )["positivity"]()
        self.assertFalse(base.passed)  # ~25% extreme PS at the default cap

        looser = build_default_checks(
            {"treatment_variable": "t", "outcome_variable": "y", "covariates": ["x1"]},
            df, "desc", None,
            thresholds={"positivity": {"max_extreme_pct": 1.0}},
        )["positivity"]()
        self.assertTrue(looser.passed)

    def test_json_list_bounds_are_coerced_to_tuples(self):
        df = self._positivity_df()
        # A list that stayed a list would TypeError inside the comparisons.
        result = build_default_checks(
            {"treatment_variable": "t", "outcome_variable": "y", "covariates": ["x1"]},
            df, "desc", None,
            thresholds={"positivity": {"extreme_ps_bounds": [0.01, 0.99]}},
        )["positivity"]()
        self.assertIn(result.passed, (True, False))

    def test_iv_relevance_threshold_flows_into_details(self):
        rng = np.random.default_rng(3)
        n = 400
        df = pd.DataFrame(
            {
                "z": rng.normal(size=n),
                "x1": rng.normal(size=n),
            }
        )
        df["t"] = (0.8 * df["z"] + 0.5 * df["x1"] + rng.normal(size=n)) > 0
        from cais.models import AssumptionVariables
        vars = AssumptionVariables(
            df=df, treatment="t", outcome="y", covariates=["x1"],
            instruments=["z"], variables_summary={},
        )

        default = build_default_checks(
            {"treatment_variable": "t", "outcome_variable": "y", "covariates": ["x1"],
             "instrument_variable": "z"},
            df, "desc", None, thresholds=None,
        )["iv_relevance"]()
        self.assertEqual(default.details.get("threshold"), 10.0)

        raised = build_default_checks(
            {"treatment_variable": "t", "outcome_variable": "y", "covariates": ["x1"],
             "instrument_variable": "z"},
            df, "desc", None,
            thresholds={"iv_relevance": {"f_threshold": 50.0}},
        )["iv_relevance"]()
        self.assertEqual(raised.details.get("threshold"), 50.0)
        if default.details.get("f_statistic") is not None:
            f = default.details["f_statistic"]
            if 10 <= f < 50:
                self.assertFalse(default.passed)
                self.assertFalse(raised.passed)


class TestTreeRecordsThresholds(unittest.TestCase):

    def _did_props(self):
        return dict(
            treatment_variable="t",
            outcome_variable="y",
            treatment_variable_type="binary",
            covariates=["x1"],
            instrument_variable=None,
            mediator_variable=None,
            running_variable=None,
            cutoff_value=None,
            time_variable="year",
            group_variable="unit",
            is_rct=False,
            has_temporal_structure=True,
        )

    def _stub_checks(self):
        def stub():
            from cais.models import AssumptionResult
            return AssumptionResult(passed=True, reasoning="stub pass")
        return {name: stub for name in [
            "sutva", "stable_group_composition", "baseline_outcome_balance",
            "parallel_trends", "no_anticipation",
        ]}

    def test_explicit_thresholds_recorded_on_result_and_to_dict(self):
        thresholds = {"no_anticipation": {"alpha": 0.2}}
        # No checks injected: the default factory consumes the thresholds, so
        # the run's recorded policy must echo them.
        res = select_method_v2(self._did_props(), thresholds=thresholds)

        self.assertEqual(res.thresholds, thresholds)
        self.assertIn("thresholds", res.to_dict())
        self.assertEqual(res.to_dict()["thresholds"], thresholds)
        self.assertEqual(res.method, "difference_in_differences")

    def test_injected_checks_are_not_recorded_as_threshold_users(self):
        thresholds = {"no_anticipation": {"alpha": 0.2}}
        res = select_method_v2(self._did_props(), checks=self._stub_checks(), thresholds=thresholds)

        self.assertEqual(res.thresholds, {})  # injected stubs never consumed them

    def test_validation_failure_is_loud_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            select_method_v2(self._did_props(), thresholds={"parallel_trends": {"significance": 0.1}})


if __name__ == "__main__":
    unittest.main()
