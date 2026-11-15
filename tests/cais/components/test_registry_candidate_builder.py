from cais.components.registry_candidate_builder import RegistryCandidateBuilder


def test_registry_builder_uses_estimand_links_and_retains_unknown_features():
    builder = RegistryCandidateBuilder(
        links=({"Strategy_ID": "S7", "Estimand_ID": "E01", "Review_status": "reviewed"},
               {"Strategy_ID": "S16", "Estimand_ID": "E07", "Review_status": "reviewed"}),
        rules=({"Strategy_ID": "S7", "Feature_ID": "candidate_covariates", "Required_value": "yes", "When_confirmed_absent": "exclude_this_variant_only"},),
        routes=({"Strategy_ID": "S7", "Estimator_ID": "backdoor_adjustment", "Implemented_in_CAIS": "Yes", "Supported_Estimand_IDs": "E01"},
                {"Strategy_ID": "S16", "Estimator_ID": "instrumental_variable", "Implemented_in_CAIS": "Yes", "Supported_Estimand_IDs": "E07"}),
    )
    candidates = builder.build("E01", feature_evidence={"candidate_covariates": "unknown"})
    assert len(candidates) == 1
    assert candidates[0]["strategy_id"] == "S7"
    assert candidates[0]["structural_status"] == "eligible"
    assert "candidate_covariates" in candidates[0]["unknown_structural_features"]


def test_registry_builder_blocks_explicit_structural_contradiction():
    builder = RegistryCandidateBuilder(
        links=({"Strategy_ID": "S7", "Estimand_ID": "E01", "Review_status": "reviewed"},),
        rules=({"Strategy_ID": "S7", "Feature_ID": "candidate_covariates", "Required_value": "yes", "When_confirmed_absent": "exclude_this_variant_only"},),
        routes=({"Strategy_ID": "S7", "Estimator_ID": "backdoor_adjustment", "Implemented_in_CAIS": "Yes", "Supported_Estimand_IDs": "E01"},),
    )
    candidates = builder.build("E01", feature_evidence={"candidate_covariates": False})
    assert candidates[0]["structural_status"] == "confirmed_violation"
    assert candidates[0]["structural_failures"] == ["candidate_covariates"]
