"""Tests for the revised decision tree (decision_tree_v2).

Traversal tests use injected stub verdicts, so no data or API key is needed.
One integration test binds the real checks to the repo's smoking2 dataset.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cais.models import AssumptionResult
from cais.components.decision_tree_v2 import (
    ALL_CHECK_KEYS,
    build_default_checks,
    extract_period_starts,
    M_DID,
    M_DID_TIME_WINDOW,
    M_DIFF_IN_MEANS,
    M_DONUT_RDD,
    M_FRONTDOOR,
    M_GPS,
    M_IPW,
    M_IV,
    M_OLS_PRE_TREATMENT,
    M_PSM,
    M_RDD,
    M_TRIMMED_IPW,
    select_method_v2,
)


def stub_checks(**overrides):
    """Every check passes unless overridden. Overrides are verdicts or callables."""
    checks = {}
    for name in ALL_CHECK_KEYS:
        checks[name] = lambda: AssumptionResult(passed=True, reasoning="stub pass")
    for name, verdict in overrides.items():
        if callable(verdict):
            checks[name] = verdict
        else:
            checks[name] = (lambda v=verdict: AssumptionResult(passed=v, reasoning=f"stub {v}"))
    return checks


def base_props(**overrides):
    props = dict(
        treatment_variable="t",
        outcome_variable="y",
        treatment_variable_type="binary",
        covariates=["x1"],
        instrument_variable=None,
        mediator_variable=None,
        running_variable=None,
        cutoff_value=None,
        time_variable=None,
        group_variable="unit",
        is_rct=False,
        has_temporal_structure=False,
    )
    props.update(overrides)
    return props


def run(properties=None, checks=None, checks_factory=None, **kwargs):
    if checks is None and checks_factory is None:
        checks = stub_checks()  # default: everything passes
    return select_method_v2(
        properties or base_props(),
        checks=checks,
        checks_factory=checks_factory,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Entry: description, data structure, SUTVA
# ---------------------------------------------------------------------------

class TestEntry(unittest.TestCase):

    def test_structure_not_supported_ends(self):
        res = run(base_props(is_structure_supported=False))
        self.assertTrue(res.ended)
        self.assertIsNone(res.method)
        self.assertIn("Data structure", res.end_reason)

    def test_missing_description_warns_and_continues(self):
        res = run(base_props(has_temporal_structure=True))
        self.assertFalse(res.ended)
        self.assertEqual(res.method, M_DID)
        self.assertTrue(any("description" in w.lower() for w in res.warnings))

    def test_description_via_prompt_callback(self):
        res = run(
            base_props(has_temporal_structure=True),
            prompt_callback=lambda _msg: "Panel of cigarette sales.",
        )
        self.assertEqual(res.description, "Panel of cigarette sales.")
        self.assertTrue(any(s.node == "Prompt user for dataset description" for s in res.steps))

    def test_sutva_false_warns_but_continues(self):
        res = run(base_props(has_temporal_structure=True), checks=stub_checks(sutva=False))
        self.assertFalse(res.ended)
        self.assertEqual(res.method, M_DID)
        self.assertTrue(any("SUTVA" in w for w in res.warnings))

    def test_sutva_untestable_warns(self):
        res = run(base_props(has_temporal_structure=True), checks=stub_checks(sutva=None))
        self.assertFalse(res.ended)
        self.assertTrue(any("SUTVA" in w for w in res.warnings))

    def test_sutva_prompt_loop_retries_until_tested(self):
        calls = {"n": 0}

        def make_checks(desc):
            if desc and "villages" in desc:
                sutva = lambda: AssumptionResult(passed=True, reasoning="resolvable with the answer")
            else:
                sutva = lambda: AssumptionResult(
                    passed=None,
                    reasoning="inconclusive",
                    details={"missing_info": "How are units arranged (households, villages, network)?"},
                )
            calls["n"] += 1
            return stub_checks(sutva=sutva, no_anticipation=True, parallel_trends=True)

        res = run(
            base_props(has_temporal_structure=True),
            checks_factory=make_checks,
            prompt_callback=lambda _msg: "Units are households in different villages.",
            description="Panel of outcomes.",
        )
        self.assertEqual(calls["n"], 2)  # initial + retry after the answer
        self.assertTrue(res.assumptions["sutva"].passed)
        self.assertEqual(res.asks_used, 1)
        self.assertEqual(res.method, M_DID)

    def test_sutva_prompt_loop_gives_up_after_three(self):
        calls = {"n": 0, "prompts": 0}

        def sutva():
            calls["n"] += 1
            return AssumptionResult(
                passed=None,
                reasoning="always inconclusive",
                details={"missing_info": "Describe any interference between units."},
            )

        def prompt(_msg):
            calls["prompts"] += 1
            return "still no useful information"

        res = run(
            base_props(has_temporal_structure=True),
            checks=stub_checks(sutva=sutva, parallel_trends=True),
            prompt_callback=prompt,
            description="Stub description to skip the description prompt.",
        )
        self.assertEqual(calls["n"], 4)  # initial + 3 retries
        self.assertEqual(calls["prompts"], 3)
        self.assertFalse(res.ended)

    def test_sutva_without_question_never_asks_even_interactively(self):
        # no LLM: the check is can't tell with no question, so a wired user
        # must NOT be prompted and the ask budget must stay untouched
        res = run(
            base_props(has_temporal_structure=True),
            checks=stub_checks(sutva=None, parallel_trends=True),
            prompt_callback=lambda _msg: "anything",
            description="Stub description to skip the description prompt.",
        )
        self.assertFalse(res.ended)
        self.assertEqual(res.asks_used, 0)


# ---------------------------------------------------------------------------
# Gate follow-up questions (SUTVA-style loop generalized to gates, budgeted)
# ---------------------------------------------------------------------------

class TestGateReaskLoop(unittest.TestCase):

    def _did_props(self):
        return base_props(has_temporal_structure=True)

    def test_gate_question_asks_then_passes(self):
        prompts = {"n": 0}

        def make_checks(desc):
            if desc and "1990" in desc:
                na = AssumptionResult(passed=True, reasoning="placebo test ran with the provided periods")
            else:
                na = AssumptionResult(
                    passed=None,
                    reasoning="periods unknown",
                    details={"missing_info": "When does the placebo period and treatment start?"},
                )
            return stub_checks(sutva=True, parallel_trends=True, no_anticipation=lambda: na)

        def prompt(msg):
            prompts["n"] += 1
            if "placebo" in msg.lower():
                return "placebo starts 1990, treatment starts 2000"
            return "Panel of cigarette sales across states."

        res = run(
            self._did_props(),
            checks_factory=make_checks,
            prompt_callback=prompt,
            description="Panel of cigarette sales across states.",
        )

        self.assertFalse(res.ended)
        self.assertEqual(res.method, M_DID)
        self.assertTrue(res.assumptions["no_anticipation"].passed)
        self.assertEqual(res.asks_used, 1)
        self.assertEqual(prompts["n"], 1)  # only the gate question; description was given
        self.assertTrue(any(s.kind == "process" and "answered" in s.outcome for s in res.steps))
        self.assertFalse(any("No-anticipation" in w for w in res.warnings))

    def test_ask_budget_shared_with_sutva(self):
        prompts = {"n": 0}

        def make_checks(_desc):
            return stub_checks(
                sutva=lambda: AssumptionResult(
                    passed=None,
                    reasoning="inconclusive",
                    details={"missing_info": "Describe any interference between units."},
                ),
                parallel_trends=True,
                no_anticipation=lambda: AssumptionResult(
                    passed=None,
                    reasoning="periods unknown",
                    details={"missing_info": "When does the placebo period start?"},
                ),
            )

        def prompt(_msg):
            prompts["n"] += 1
            return "irrelevant answer, SUTVA stays inconclusive"

        res = run(
            self._did_props(),
            checks_factory=make_checks,
            prompt_callback=prompt,
            description="Skip the description prompt.",
        )

        # SUTVA consumed the whole global budget (3 asks); the gate may not ask
        self.assertEqual(prompts["n"], 3)
        self.assertIsNone(res.assumptions["no_anticipation"].passed)
        self.assertTrue(any("ask budget exhausted" in s.outcome for s in res.steps))
        self.assertTrue(any("No-anticipation" in w for w in res.warnings))

    def test_gate_fail_with_question_asks_then_ends(self):
        prompts = {"n": 0}

        def make_checks(desc):
            if desc and "no major attrition" in desc:
                comp = AssumptionResult(passed=False, reasoning="still unstable after answer")
            else:
                comp = AssumptionResult(
                    passed=False,
                    reasoning="attrition suspected",
                    details={"missing_info": "Describe any group membership changes over time."},
                )
            return stub_checks(sutva=True, parallel_trends=True, no_anticipation=None,
                               stable_group_composition=lambda: comp)

        def prompt(msg):
            prompts["n"] += 1
            if "Override this gate" in msg:
                return "n"
            return "no major attrition occurs"

        res = run(
            self._did_props(),
            checks_factory=make_checks,
            prompt_callback=prompt,
            description="Panel of cigarette sales across states.",
        )

        self.assertTrue(res.ended)
        self.assertIsNone(res.method)
        self.assertIn("Group composition", res.end_reason)
        # 1 ask for the check's own question + 1 consult offer the user declined
        self.assertEqual(res.asks_used, 1)
        self.assertEqual(prompts["n"], 2)
        self.assertFalse(res.assumptions["stable_group_composition"].details.get("user_override"))

    def test_interactive_fail_overridden_to_pass(self):
        prompts = []

        def prompt(msg):
            prompts.append(msg)
            return "y" if "Override this gate" in msg else "answer"

        checks = stub_checks(
            sutva=True, parallel_trends=True, no_anticipation=True,
            stable_group_composition=lambda: AssumptionResult(
                passed=False, reasoning="attrition suspected"
            ),
        )
        res = run(
            self._did_props(), checks=checks, prompt_callback=prompt,
            description="Panel data.",
        )

        self.assertFalse(res.ended)
        self.assertEqual(res.method, M_DID)
        self.assertEqual(len(prompts), 1)  # only the consult; no question ever asked
        self.assertTrue(any("user override" in w for w in res.warnings))
        comp = res.assumptions["stable_group_composition"]
        self.assertFalse(comp.passed)  # the fail stays a fail, honestly recorded
        self.assertTrue(comp.details.get("user_override"))
        self.assertEqual(res.asks_used, 0)  # consult does not consume the ask budget

    def test_interactive_fail_declined_ends(self):
        checks = stub_checks(
            sutva=True, parallel_trends=True, no_anticipation=True,
            stable_group_composition=False,
        )
        res = run(
            self._did_props(), checks=checks,
            prompt_callback=lambda m: "y" if "Override this gate" in m and False else "n",
            description="Panel data.",
        )

        self.assertTrue(res.ended)
        self.assertIn("Group composition", res.end_reason)
        self.assertFalse(any("user override" in w for w in res.warnings))

    def test_batch_fail_consults_nobody_and_ends(self):
        checks = stub_checks(
            sutva=True, parallel_trends=True, no_anticipation=True,
            stable_group_composition=False,
        )
        res = run(self._did_props(), checks=checks)  # no prompt_callback

        self.assertTrue(res.ended)
        self.assertTrue(any(s.outcome == "no user available to consult" for s in res.steps))
        self.assertFalse(any("override" in w.lower() for w in res.warnings))

    def test_weak_instrument_can_be_overridden(self):
        checks = stub_checks(iv_relevance=False)
        res = run(
            base_props(instrument_variable="z"), checks=checks,
            prompt_callback=lambda m: "y" if "Override this gate" in m else None,
        )

        self.assertFalse(res.ended)
        self.assertEqual(res.method, M_IV)

    def test_no_anticipation_fail_can_be_overridden(self):
        checks = stub_checks(
            sutva=True, parallel_trends=True, stable_group_composition=True,
            no_anticipation=False,
        )
        res = run(
            self._did_props(), checks=checks,
            prompt_callback=lambda m: "y" if "Override this gate" in m else None,
        )

        self.assertFalse(res.ended)
        self.assertEqual(res.method, M_DID)

    def test_nonbinary_positivity_fail_can_be_overridden(self):
        checks = stub_checks(positivity=False)
        res = run(
            base_props(treatment_variable_type="continuous", covariates=["x1"]),
            checks=checks,
            prompt_callback=lambda m: "y" if "Override this gate" in m else None,
        )

        self.assertFalse(res.ended)
        self.assertEqual(res.method, M_GPS)

    def test_gate_can_tell_without_question_never_asks(self):
        prompts = {"n": 0}

        res = run(
            self._did_props(),
            checks=stub_checks(sutva=True, parallel_trends=lambda: AssumptionResult(passed=None, reasoning="insufficient pre data")),
            prompt_callback=lambda _m: prompts.__setitem__("n", prompts["n"] + 1) or "answer",
            description="Skip the description prompt.",
        )

        # statistical can't-tell has no question -> no ask, old behavior
        self.assertEqual(prompts["n"], 0)
        self.assertFalse(res.ended)
        self.assertTrue(any("Gate: parallel trends" in w for w in res.warnings))

    def test_answer_with_period_numbers_unblocks_real_no_anticipation(self):
        """Real checks end to end: the parsed numbers must re-run the placebo test.

        Uses the production default wiring (build_default_checks) because the
        answer is meant to update the traversal's props, which the default
        factory reads live. Pre-treatment noise is tiny so the parallel-trends
        gate cannot fail by chance.
        """
        rng = np.random.default_rng(11)
        rows = []
        for unit, treated in (("a", 0), ("b", 0), ("c", 1), ("d", 1)):
            for year in range(1970, 2001):
                rows.append(dict(
                    unit=unit,
                    year=year,
                    treated_group=treated,
                    # a real post-1988 effect makes this a sensible DiD; the
                    # placebo window (1980-1987) has no group-specific pattern
                    y=10 + 0.3 * (year - 1970) + 0.2 * treated * (year >= 1988)
                    + rng.normal(0, 1.0),
                ))
        df = pd.DataFrame(rows)
        props = self._did_props()
        props.update(
            treatment_variable="treated_group",
            outcome_variable="y",
            covariates=[],
            time_variable="year",
            group_variable="unit",
            treatment_period_start=1988,
        )

        def prompt(_msg):
            return "placebo=1980, treatment=1988"

        res = select_method_v2(
            props,
            df=df,
            prompt_callback=prompt,
            description="Panel of outcomes for four units, treatment starts 1988.",
        )

        self.assertFalse(res.ended)
        self.assertEqual(res.method, M_DID)
        # the answer was parsed into the missing placebo period and the real
        # placebo test ran; before the ask it was always can't tell
        self.assertEqual(res.asks_used, 1)
        self.assertTrue(any("parsed period starts" in s.outcome for s in res.steps))
        self.assertIsNotNone(res.assumptions["no_anticipation"].passed)
        self.assertIn("Placebo treatment effect", res.assumptions["no_anticipation"].reasoning)

    def test_batch_run_with_gate_question_skips_asking(self):
        res = run(
            self._did_props(),
            checks=stub_checks(
                sutva=True,
                parallel_trends=True,
                no_anticipation=lambda: AssumptionResult(
                    passed=None, reasoning="periods unknown",
                    details={"missing_info": "When does the placebo period start?"},
                ),
            ),
        )

        self.assertFalse(res.ended)
        self.assertIsNone(res.assumptions["no_anticipation"].passed)
        self.assertTrue(any("no user available" in s.outcome for s in res.steps))
        self.assertEqual(res.asks_used, 0)


# ---------------------------------------------------------------------------
# RCT branch
# ---------------------------------------------------------------------------

class TestRCTBranch(unittest.TestCase):

    def test_rct_with_pretreatment_vars_ols(self):
        res = run(base_props(is_rct=True, covariates=["x1", "x2"]))
        self.assertEqual(res.method, M_OLS_PRE_TREATMENT)

    def test_rct_without_pretreatment_vars_dim(self):
        res = run(base_props(is_rct=True, covariates=[], has_pre_treatment_variables=False))
        self.assertEqual(res.method, M_DIFF_IN_MEANS)

    def test_encouragement_strong_f_assignment_analysis(self):
        res = run(
            base_props(is_rct=True, instrument_variable="z", is_encouragement_design=True),
            checks=stub_checks(iv_relevance=True),
        )
        self.assertEqual(res.method, M_DIFF_IN_MEANS)
        self.assertEqual(res.variant, "on_assignment")

    def test_encouragement_weak_f_with_valid_instrument_iv(self):
        res = run(
            base_props(is_rct=True, instrument_variable="z", is_encouragement_design=True),
            checks=stub_checks(iv_relevance=False),
        )
        self.assertEqual(res.method, M_IV)
        self.assertIn("iv_exclusion", res.assumptions)
        self.assertIn("iv_exogeneity", res.assumptions)
        self.assertIn("iv_monotonicity", res.assumptions)

    def test_encouragement_weak_f_without_instrument_ends(self):
        res = run(
            base_props(is_rct=True, instrument_variable=None, is_encouragement_design=True),
            checks=stub_checks(iv_relevance=False),
        )
        self.assertTrue(res.ended)
        self.assertIn("Weak instrument", res.end_reason)

    def test_encouragement_relevance_untestable_routes_to_validity(self):
        res = run(
            base_props(is_rct=True, instrument_variable="z", is_encouragement_design=True),
            checks=stub_checks(iv_relevance=None),
        )
        self.assertEqual(res.method, M_IV)
        self.assertTrue(any("relevance" in w.lower() for w in res.warnings))


# ---------------------------------------------------------------------------
# DiD path
# ---------------------------------------------------------------------------

DID_PROPS = dict(
    has_temporal_structure=True,
    time_variable="year",
    treatment_period_start=1988,
)


class TestDIDPath(unittest.TestCase):

    def test_all_gates_pass_selects_did(self):
        res = run(base_props(**DID_PROPS))
        self.assertFalse(res.ended)
        self.assertEqual(res.method, M_DID)
        outcomes = {s.node: s.outcome for s in res.steps}
        self.assertEqual(outcomes["Gate: parallel trends"], "pass")
        self.assertEqual(outcomes["Gate: no anticipation (placebo)"], "pass")

    def test_composition_gate_fail_ends(self):
        res = run(base_props(**DID_PROPS), checks=stub_checks(stable_group_composition=False))
        self.assertTrue(res.ended)
        self.assertIn("composition", res.end_reason.lower())

    def test_parallel_trends_fail_ends(self):
        res = run(base_props(**DID_PROPS), checks=stub_checks(parallel_trends=False))
        self.assertTrue(res.ended)
        self.assertIn("Parallel trends", res.end_reason)

    def test_baseline_balance_fail_is_advisory(self):
        res = run(base_props(**DID_PROPS), checks=stub_checks(baseline_outcome_balance=False))
        self.assertFalse(res.ended)
        self.assertEqual(res.method, M_DID)
        self.assertTrue(any("Baseline outcome balance" in w for w in res.warnings))

    def test_no_anticipation_fail_unbounded_ends(self):
        res = run(base_props(**DID_PROPS), checks=stub_checks(no_anticipation=False))
        self.assertTrue(res.ended)
        self.assertIn("Anticipation", res.end_reason)

    def test_no_anticipation_fail_bounded_goes_time_window(self):
        res = run(
            base_props(**DID_PROPS, anticipation_known_bounded=True),
            checks=stub_checks(no_anticipation=False),
        )
        self.assertEqual(res.method, M_DID_TIME_WINDOW)
        self.assertFalse(res.implemented)
        self.assertTrue(any("not implemented" in w for w in res.warnings))

    def test_untestable_gate_warns_and_continues(self):
        res = run(base_props(**DID_PROPS), checks=stub_checks(parallel_trends=None))
        self.assertFalse(res.ended)
        self.assertEqual(res.method, M_DID)
        self.assertTrue(any("could not be tested" in w for w in res.warnings))

    def test_no_anticipation_untestable_warns_and_continues(self):
        res = run(base_props(**DID_PROPS), checks=stub_checks(no_anticipation=None))
        self.assertFalse(res.ended)
        self.assertEqual(res.method, M_DID)
        self.assertTrue(any("No-anticipation" in w for w in res.warnings))


# ---------------------------------------------------------------------------
# RDD path
# ---------------------------------------------------------------------------

RDD_PROPS = dict(
    has_temporal_structure=False,
    running_variable="score",
    cutoff_value=0.0,
    covariates=["x1"],
)


class TestRDDPath(unittest.TestCase):

    def test_gates_pass_selects_rdd(self):
        res = run(base_props(**RDD_PROPS))
        self.assertFalse(res.ended)
        self.assertEqual(res.method, M_RDD)

    def test_manipulation_fail_local_goes_donut(self):
        res = run(
            base_props(**RDD_PROPS, local_manipulation=True, donut_width=0.5),
            checks=stub_checks(rdd_no_manipulation=False),
        )
        self.assertEqual(res.method, M_DONUT_RDD)
        self.assertTrue(res.implemented)
        self.assertEqual(res.execution["method"], M_DONUT_RDD)
        self.assertEqual(res.execution["options"]["donut_width"], 0.5)

    def test_manipulation_fail_local_without_width_ends(self):
        res = run(
            base_props(**RDD_PROPS, local_manipulation=True),
            checks=stub_checks(rdd_no_manipulation=False),
        )
        self.assertTrue(res.ended)
        self.assertIn("no donut width", res.end_reason)

    def test_manipulation_fail_no_local_ends(self):
        res = run(base_props(**RDD_PROPS), checks=stub_checks(rdd_no_manipulation=False))
        self.assertTrue(res.ended)
        self.assertIn("manipulation", res.end_reason)

    def test_covariate_continuity_fail_local_goes_donut(self):
        res = run(
            base_props(**RDD_PROPS, local_manipulation=True, donut_width=0.5),
            checks=stub_checks(rdd_covariate_continuity=False),
        )
        self.assertEqual(res.method, M_DONUT_RDD)
        self.assertTrue(res.implemented)

    def test_untestable_rdd_gate_warns_and_continues(self):
        res = run(base_props(**RDD_PROPS), checks=stub_checks(rdd_no_manipulation=None))
        self.assertFalse(res.ended)
        self.assertEqual(res.method, M_RDD)
        self.assertTrue(any("RDD gate" in w for w in res.warnings))


# ---------------------------------------------------------------------------
# Identification options: instrument / mediator / backdoor
# ---------------------------------------------------------------------------

class TestOptions(unittest.TestCase):

    def test_instrument_has_priority(self):
        res = run(base_props(instrument_variable="z", mediator_variable="m", covariates=["x1"]))
        self.assertEqual(res.method, M_IV)

    def test_weak_instrument_ends(self):
        res = run(base_props(instrument_variable="z"), checks=stub_checks(iv_relevance=False))
        self.assertTrue(res.ended)
        self.assertIn("Weak instrument", res.end_reason)

    def test_mediator_selects_frontdoor_adjustment(self):
        res = run(base_props(mediator_variable="m", treatment_reference_level=0))
        self.assertEqual(res.method, M_FRONTDOOR)
        self.assertTrue(res.implemented)
        self.assertEqual(res.execution["method"], M_FRONTDOOR)
        self.assertEqual(res.execution["display_name"], "Frontdoor Adjustment")
        self.assertEqual(res.execution["options"]["mediator_variable"], "m")
        self.assertEqual(res.execution["options"]["treatment_reference_level"], 0)
        self.assertTrue(any("frontdoor" in w.lower() for w in res.warnings))

    def test_mediator_positivity_fail_ends(self):
        res = run(base_props(mediator_variable="m"),
                  checks=stub_checks(frontdoor_positivity=False))
        self.assertTrue(res.ended)
        self.assertIn("frontdoor", res.end_reason.lower())

    def test_frontdoor_unusable_mediator_falls_back_to_backdoor(self):
        # 12 distinct mediator values -> beyond the frontdoor level cap.
        df = pd.DataFrame({
            "treatment": [0, 1] * 6,
            "mediator": [round(0.05 + 0.08 * i, 2) for i in range(12)],
            "outcome": [1.0 + 0.2 * i for i in range(12)],
            "x1": [round(0.05 + 0.07 * i, 2) for i in range(12)],
        })
        res = run(base_props(mediator_variable="mediator"), checks=stub_checks(), df=df)
        self.assertEqual(res.method, M_IPW)
        self.assertFalse(res.ended)
        self.assertTrue(any("frontdoor" in w.lower() and "continuing to the backdoor" in w.lower()
                            for w in res.warnings))

    def test_frontdoor_continuous_treatment_falls_back_to_gps(self):
        df = pd.DataFrame({
            "treatment": [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "mediator": [0, 1, 0, 1, 0, 1, 0, 1],
            "outcome": [1.0, 2.0, 1.5, 2.5, 1.8, 2.8, 2.1, 3.1],
            "x1": [0.1, 0.3, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8],
        })
        res = run(
            base_props(mediator_variable="mediator", treatment_variable_type="continuous"),
            checks=stub_checks(),
            df=df,
        )
        self.assertEqual(res.method, M_GPS)

    def test_backdoor_balanced_selects_ipw(self):
        res = run(base_props(), checks=stub_checks(cond_ignorability=True, positivity=True))
        self.assertEqual(res.method, M_IPW)
        self.assertIn("balance_after_weighting", res.planned_post_checks)

    def test_backdoor_imbalanced_selects_matching(self):
        res = run(base_props(), checks=stub_checks(cond_ignorability=False))
        self.assertEqual(res.method, M_PSM)
        self.assertIn("balance_after_matching", res.planned_post_checks)
        self.assertTrue(any(s.node == "Are the covariates in the two groups balanced?" and s.outcome == "No"
                            for s in res.steps))

    def test_backdoor_positivity_fail_goes_trimmed_ipw(self):
        res = run(base_props(), checks=stub_checks(positivity=False))
        self.assertEqual(res.method, M_TRIMMED_IPW)
        self.assertTrue(res.implemented)
        self.assertEqual(res.execution["method"], M_TRIMMED_IPW)
        self.assertEqual(res.execution["options"]["ps_bounds"], [0.1, 0.9])
        self.assertEqual(res.execution["display_name"], "Trimmed Inverse Probability Weighting")
        self.assertIn("balance_after_weighting", res.planned_post_checks)
        self.assertTrue(any("trimmed ipw" in w.lower() for w in res.warnings))
        self.assertTrue(any(s.node == "Trimmed IPW" and s.outcome == "ps_bounds applied"
                            for s in res.steps))

    def test_trimmed_ipw_uses_configured_positivity_bounds(self):
        res = run(
            base_props(),
            checks=stub_checks(positivity=False),
            thresholds={"positivity": {"extreme_ps_bounds": [0.05, 0.95]}},
        )
        self.assertEqual(res.execution["options"]["ps_bounds"], [0.05, 0.95])
        self.assertIn("[0.05, 0.95]", " ".join(res.warnings))

    def test_trimmed_ipw_falls_back_to_default_bounds(self):
        res = run(
            base_props(),
            checks=stub_checks(positivity=False),
            thresholds={"positivity": {"extreme_ps_bounds": [0.9, 0.1]}},
        )
        self.assertEqual(res.execution["options"]["ps_bounds"], [0.1, 0.9])

    def test_backdoor_untestable_balance_defaults_to_matching(self):
        res = run(base_props(), checks=stub_checks(cond_ignorability=None))
        self.assertEqual(res.method, M_PSM)
        self.assertTrue(any("balance could not be assessed" in w.lower() for w in res.warnings))

    def test_no_options_available_ends(self):
        res = run(base_props(covariates=[]))
        self.assertTrue(res.ended)
        self.assertIn("No viable identification strategy", res.end_reason)

    def test_nonbinary_backdoor_selects_gps(self):
        res = run(base_props(treatment_variable_type="continuous", covariates=["x1"]))
        self.assertEqual(res.method, M_GPS)
        # The graph's GPS branch carries post-check advisories (balance, specification).
        self.assertEqual(res.planned_post_checks, ["balance_after_gps", "gps_specification"])

    def test_nonbinary_positivity_fail_ends(self):
        res = run(
            base_props(treatment_variable_type="continuous", covariates=["x1"]),
            checks=stub_checks(positivity=False),
        )
        self.assertTrue(res.ended)
        self.assertIn("propensity score", res.end_reason.lower())

    def test_nonbinary_instrument_selects_iv(self):
        res = run(base_props(treatment_variable_type="continuous", instrument_variable="z"))
        self.assertEqual(res.method, M_IV)

    def test_nonbinary_no_options_ends(self):
        res = run(base_props(treatment_variable_type="continuous", covariates=[]))
        self.assertTrue(res.ended)


# ---------------------------------------------------------------------------
# Propensity-score wiring
# ---------------------------------------------------------------------------

class TestPropensityWiring(unittest.TestCase):

    def test_propensity_scores_use_covariates_only(self):
        import numpy as np
        from cais.models import AssumptionVariables
        from cais.components.decision_tree_v2 import _fit_propensity_scores

        rng = np.random.default_rng(0)
        n = 400
        x = rng.normal(size=n)
        t = (rng.uniform(size=n) < 0.5).astype(int)  # treatment independent of x
        df = pd.DataFrame({"t": t, "y": rng.normal(size=n), "x": x})
        vars = AssumptionVariables(df=df, treatment="t", outcome="y", covariates=["x"])

        ps = _fit_propensity_scores(vars)
        self.assertEqual(len(ps), n)
        # If treatment leaked into its own model this would saturate near 0/1.
        self.assertLess(ps.max(), 0.99)
        self.assertGreater(ps.min(), 0.01)


# ---------------------------------------------------------------------------
# Period parsing for the no-anticipation follow-up question
# ---------------------------------------------------------------------------

class TestExtractPeriodStarts(unittest.TestCase):

    def test_named_tokens_win(self):
        parsed = extract_period_starts("placebo=1986, treatment=1988", {})
        self.assertEqual(parsed, {"placebo_period_start": 1986.0, "treatment_period_start": 1988.0})

    def test_named_tokens_do_not_override_known_values(self):
        parsed = extract_period_starts(
            "placebo=1986, treatment=1950", {"treatment_period_start": 1988}
        )
        self.assertEqual(parsed, {"placebo_period_start": 1986.0})

    def test_two_plain_numbers_assign_by_size(self):
        parsed = extract_period_starts("it ran from 1986 and effects could start 1988", {})
        self.assertEqual(parsed["placebo_period_start"], 1986.0)
        self.assertEqual(parsed["treatment_period_start"], 1988.0)

    def test_single_number_fills_only_missing_field(self):
        parsed = extract_period_starts("1986", {"treatment_period_start": 1988})
        self.assertEqual(parsed, {"placebo_period_start": 1986.0})

    def test_single_number_with_both_missing_is_ambiguous(self):
        self.assertEqual(extract_period_starts("1986", {}), {})

    def test_garbage_returns_empty(self):
        self.assertEqual(extract_period_starts("no idea at all", {}), {})

    def test_skip_is_empty(self):
        self.assertEqual(extract_period_starts("", {}), {})


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------

class TestResultShape(unittest.TestCase):

    def test_to_dict_is_serializable_and_complete(self):
        res = run(base_props(**DID_PROPS))
        d = res.to_dict()
        for key in ("method", "implemented", "ended", "end_reason", "warnings",
                    "planned_post_checks", "path", "steps", "assumptions", "execution"):
            self.assertIn(key, d)
        self.assertIsInstance(d["path"], str)
        for entry in d["assumptions"].values():
            for key in ("passed", "reasoning", "details"):
                self.assertIn(key, entry)

    def test_path_records_key_nodes(self):
        res = run(base_props(**DID_PROPS))
        self.assertIn("Is this a randomized trial?", res.path)
        self.assertIn("Is treatment binary?", res.path)
        self.assertIn("Is temporal information available?", res.path)
        self.assertIn("Gate: parallel trends", res.path)
        self.assertIn(M_DID, res.path)


# ---------------------------------------------------------------------------
# Integration: real checks on the repo's smoking2 data, no LLM needed
# ---------------------------------------------------------------------------

class TestIntegrationRealChecks(unittest.TestCase):

    def test_smoking2_did_with_default_wiring(self):
        data_path = Path(__file__).resolve().parents[2] / "test_data" / "smoking2.csv"
        if not data_path.exists():
            self.skipTest(f"missing test data: {data_path}")

        df = pd.read_csv(data_path)
        properties = base_props(
            treatment_variable="california",
            outcome_variable="cigsale",
            covariates=[],
            time_variable="year",
            group_variable="state",
            unit_variable="state",
            has_temporal_structure=True,
            treatment_period_start=1988,
            is_rct=False,
        )
        res = select_method_v2(
            properties,
            df=df,
            description=(
                "Cigarette sales across US states 1970-2000. California passed "
                "Proposition 99 in 1988 imposing a tobacco tax."
            ),
        )
        self.assertFalse(res.ended)
        self.assertEqual(res.method, M_DID)
        self.assertTrue(res.assumptions["parallel_trends"].passed)
        self.assertIsNotNone(res.assumptions["parallel_trends"].details.get("p_value"))
        # Only one treated state, so differential attrition is not testable.
        self.assertIsNone(res.assumptions["stable_group_composition"].passed)
        self.assertIn("too few units", res.assumptions["stable_group_composition"].reasoning)
        # placebo period is not available anywhere yet -> inconclusive, warn and continue
        self.assertIsNone(res.assumptions["no_anticipation"].passed)
        # LLM checks without an LLM must be inconclusive
        self.assertIsNone(res.assumptions["sutva"].passed)


if __name__ == "__main__":
    unittest.main()
