import pytest
import pandas as pd
import numpy as np
from auto_causal.methods.instrumental_variable.estimator import estimate_effect, build_iv_graph_gml

# Consistent random state for reproducibility
SEED = 42
TRUE_EFFECT = 2.0

@pytest.fixture
def synthetic_iv_data():
    """Generates synthetic data suitable for IV estimation."""
    np.random.seed(SEED)
    n_samples = 2000
    # Instrument (relevant and exogenous)
    Z = np.random.normal(loc=5, scale=1, size=n_samples)
    # Observed Covariate
    W = np.random.normal(loc=2, scale=1, size=n_samples)
    # Unobserved Confounder
    U = np.random.normal(loc=0, scale=1, size=n_samples)

    # Treatment model: T = alpha*Z + beta*W + gamma*U + error
    # Ensure Z has a reasonably strong effect on T (relevance)
    T = 0.6 * Z + 0.4 * W + 0.7 * U + np.random.normal(loc=0, scale=1, size=n_samples)

    # Outcome model: Y = TRUE_EFFECT*T + delta*W + eta*U + error
    # Z should NOT directly affect Y (exclusion restriction)
    Y = TRUE_EFFECT * T + 0.3 * W + 0.9 * U + np.random.normal(loc=0, scale=1, size=n_samples)

    df = pd.DataFrame({
        'outcome': Y,
        'treatment': T,
        'instrument': Z,
        'covariate': W,
        'unobserved': U # Keep for reference, but should not be used in estimation
    })
    return df


def test_build_iv_graph_gml():
    """Tests the GML graph construction."""
    gml = build_iv_graph_gml(
        treatment='T', outcome='Y', instruments=['Z1', 'Z2'], covariates=['W1', 'W2']
    )
    assert 'node [ id "T" label "T" ]' in gml
    assert 'node [ id "Y" label "Y" ]' in gml
    assert 'node [ id "Z1" label "Z1" ]' in gml
    assert 'node [ id "Z2" label "Z2" ]' in gml
    assert 'node [ id "W1" label "W1" ]' in gml
    assert 'node [ id "W2" label "W2" ]' in gml
    assert 'node [ id "U" label "U" ]' in gml # Unobserved confounder

    assert 'edge [ source "Z1" target "T" ]' in gml
    assert 'edge [ source "Z2" target "T" ]' in gml
    assert 'edge [ source "W1" target "T" ]' in gml
    assert 'edge [ source "W2" target "T" ]' in gml
    assert 'edge [ source "W1" target "Y" ]' in gml
    assert 'edge [ source "W2" target "Y" ]' in gml
    assert 'edge [ source "T" target "Y" ]' in gml
    assert 'edge [ source "U" target "T" ]' in gml
    assert 'edge [ source "U" target "Y" ]' in gml

    assert 'edge [ source "Z1" target "Y" ]' not in gml # Exclusion
    assert 'edge [ source "Z2" target "Y" ]' not in gml # Exclusion

def test_estimate_effect_dowhy_path(synthetic_iv_data):
    """Tests the IV estimation using the primary DoWhy path."""
    df = synthetic_iv_data
    results = estimate_effect(
        df=df,
        treatment='treatment',
        outcome='outcome',
        instrument='instrument',
        covariates=['covariate']
    )

    print("DoWhy Path Results:", results)
    assert results is not None
    assert 'error' not in results
    assert results['method_used'] == 'dowhy'
    assert results['effect_estimate'] == pytest.approx(TRUE_EFFECT, abs=0.2) # Allow some tolerance
    assert 'diagnostics' in results
    assert results['diagnostics']['first_stage_f_statistic'] > 10
    assert results['diagnostics']['is_instrument_weak'] is False
    assert results['diagnostics']['overid_test_applicable'] is False # Only 1 instrument

def test_estimate_effect_statsmodels_fallback(synthetic_iv_data):
    """Tests the IV estimation using the statsmodels fallback path."""
    df = synthetic_iv_data
    results = estimate_effect(
        df=df,
        treatment='treatment',
        outcome='outcome',
        instrument='instrument',
        covariates=['covariate'],
        force_statsmodels=True # Force skipping DoWhy
    )

    print("Statsmodels Path Results:", results)
    assert results is not None
    assert 'error' not in results
    assert results['method_used'] == 'statsmodels'
    assert results['effect_estimate'] == pytest.approx(TRUE_EFFECT, abs=0.2)
    assert 'diagnostics' in results
    assert results['diagnostics']['first_stage_f_statistic'] > 10
    assert results['diagnostics']['is_instrument_weak'] is False
    assert results['diagnostics']['overid_test_applicable'] is False

def test_estimate_effect_missing_column():
    """Tests error handling for missing columns."""
    df = pd.DataFrame({'outcome': [1, 2], 'instrument': [3, 4]})
    results = estimate_effect(
        df=df,
        treatment='treatment', # Missing
        outcome='outcome',
        instrument='instrument',
        covariates=[]
    )
    assert 'error' in results
    assert "Missing required columns" in results['error']

def test_estimate_effect_no_instrument():
    """Tests error handling when no instrument is provided."""
    df = pd.DataFrame({'outcome': [1, 2], 'treatment': [3, 4]})
    results = estimate_effect(
        df=df,
        treatment='treatment',
        outcome='outcome',
        instrument=[], # Empty
        covariates=[]
    )
    assert 'error' in results
    assert "Instrument variable(s) must be provided" in results['error']

# TODO: Add tests for:
# - Cases where DoWhy fails and fallback *should* occur
# - Overidentification test results when applicable (using >1 instrument in synthetic data)
# - More complex graph structures if needed
# - Handling of NaNs 