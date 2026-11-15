import pandas as pd

from cais.models import AssumptionVariables
from cais.methods.pre_model_assumption_utils import (
    ASSUMPTION_SPECS,
    evaluate_assumptions,
    get_assumption_specs,
)


def test_all_executable_routes_have_explicit_assumption_contracts():
    from cais.methods import METHOD_MAPPING

    missing = set(METHOD_MAPPING) - set(ASSUMPTION_SPECS)
    assert not missing
    for method in METHOD_MAPPING:
        for spec in get_assumption_specs(method):
            assert spec.classification in {"testable", "advisory", "mixed"}
            assert spec.evidence_type in {"statistical_diagnostic", "llm_argument", "mixed"}
            assert spec.failure_policy in {"block", "warn", "advisory"}


def test_advisory_failures_never_become_blocking_failures():
    df = pd.DataFrame({"t": [0, 1, 0, 1], "y": [1., 2., 1., 3.], "x": [0., 1., 0., 1.]})
    variables = AssumptionVariables(df=df, treatment="t", outcome="y", covariates=["x"])

    report = evaluate_assumptions("linear_regression", variables, llm=None)
    advisory = {item["assumption_id"]: item for item in report["advisory_assessments"]}

    assert advisory["sutva"]["status"] == "insufficient_evidence"
    assert advisory["conditional_exchangeability"]["status"] == "insufficient_evidence"
    assert not advisory["sutva"]["blocking"]
    assert not advisory["conditional_exchangeability"]["blocking"]


def test_iv_separates_relevance_from_advisory_identification_assumptions():
    ids = {spec.assumption_id: spec for spec in get_assumption_specs("instrumental_variable")}

    assert ids["iv_relevance"].classification == "testable"
    assert ids["iv_relevance"].failure_policy == "block"
    for name in ("iv_exclusion", "iv_exogeneity", "iv_monotonicity"):
        assert ids[name].classification == "advisory"
        assert ids[name].failure_policy == "advisory"
