import numpy as np
import pandas as pd

from cais.methods.tmle import estimate_tmle
from cais.methods.partial_identification.bounds import manski_bounds, balke_pearl_bounds


def test_tmle_binary_ate():
    rng = np.random.default_rng(12)
    n = 400
    x = rng.normal(size=n)
    a = rng.binomial(1, 1/(1+np.exp(-.4*x)))
    p = 1/(1+np.exp(-(-.3 + .8*a + .2*x)))
    y = rng.binomial(1, p)
    r = estimate_tmle(pd.DataFrame({'a':a,'y':y,'x':x}), 'a','y',['x'], n_splits=3)
    assert r['method_used'].startswith('cross-fitted TMLE')
    assert 0 <= r['effect_estimate'] <= 1


def test_manski_returns_identified_set():
    d = pd.DataFrame({'a':[0,0,1,1], 'y':[0.,1.,1.,1.]})
    r = manski_bounds(d, 'a', 'y', 0, 1)
    assert r['effect_estimate'] is None
    assert r['identified_set'][0] <= r['identified_set'][1]


def test_balke_pearl_returns_sharp_iv_interval():
    d = pd.DataFrame({'z':[0,0,1,1], 'a':[0,1,0,1], 'y':[0,1,0,1]})
    r = balke_pearl_bounds(d, 'z', 'a', 'y')
    assert r['effect_estimate'] is None
    assert r['identified_set'][0] <= r['identified_set'][1]
