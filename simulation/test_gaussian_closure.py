"""Independent small-dimensional identities for the signed quadratic diagnostic."""
import itertools
import numpy as np
from gaussian_closure_common import weighted_u, cell_statistics


def run():
    rng = np.random.default_rng(2026110201); checked = 0
    for n in [4, 5, 8]:
        x = rng.normal(size=(n, 5))+np.arange(5)*.7
        for sign in [np.ones(5), np.array([1., -1., 1., -1., 1.])]:
            K = (x*sign)@x.T
            for counts in [np.ones(n), rng.integers(1, 5, n), np.r_[np.ones(3), np.zeros(n-3)]]:
                pairs = list(itertools.permutations(range(n), 2)); quads = list(itertools.permutations(range(n), 4))
                w2 = np.array([counts[i]*counts[j] for i, j in pairs])
                w4 = np.array([counts[i]*counts[j]*counts[k]*counts[l] for i, j, k, l in quads])
                expected = np.nan if w4.sum() == 0 else 2*(
                    w2@np.array([K[i,j]**2 for i,j in pairs])/w2.sum()-
                    w4@np.array([K[i,j]*K[k,l] for i,j,k,l in quads])/w4.sum())
                np.testing.assert_allclose(weighted_u(K, counts)[0], expected, rtol=1e-10, atol=1e-10)
                checked += 1
            c = np.ones((n, 1)); _, var, _, plug = cell_statistics(K, c)
            m = x.mean(0); C = np.cov(x.T, ddof=0); A = np.diag(sign)
            dense = 2*np.trace(A@C@A@C)+4*m@A@C@A@m
            np.testing.assert_allclose(plug[0]*n/(n-1), dense, rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(var[0], np.var(np.diag(K)), rtol=1e-12, atol=1e-12)
    # Analytical population closure for noncentral signed Gaussian vectors.
    x = rng.normal(size=(2000, 5))@np.diag([.4,.7,1.,1.3,1.6])+np.arange(5)*.15
    sign = np.array([1.,-1.,1.,-1.,1.]); K = (x*sign)@x.T
    C = np.diag(np.array([.4,.7,1.,1.3,1.6])**2); m = np.arange(5)*.15; A = np.diag(sign)
    truth = 2*np.trace(A@C@A@C)+4*m@A@C@A@m
    estimate = weighted_u(K, np.ones(len(x)))[0]
    assert abs(estimate/truth-1) < .12
    return {'brute_force_weighted_pair_quad_checks': checked, 'dense_covariance_checks': 6,
            'gaussian_population_variance': truth, 'gaussian_U_estimate': estimate,
            'three_distinct_IDs_unestimable': True}


if __name__ == '__main__': print(run())
