"""Finite-window self-term controls independent of ROOT event data."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diagnose_nonstationary_moments import prefix_kernels,window_variance,self_moments
from diagnose_fluctuations import WINDOWS,fluctuation_metrics


class MomentChecks(unittest.TestCase):
    def test_prefix_formula_matches_direct_shifted_impulses(self):
        t=np.arange(2496);wave=np.exp(-t/45.)-np.exp(-t/3.);impulses=np.array([wave,wave*.7])
        shifts=np.array([0,200,800,2400,2600]);phases=np.array([0,1,0,1,0])
        for width in WINDOWS:
            p,p2=prefix_kernels(impulses,width);pred=window_variance(p,p2,phases,shifts,500,1500,width//2)
            expected=[]
            filt=-np.ones(width)/width;filt[width//2]+=1
            for phase,shift in zip(phases,shifts):
                complete=np.convolve(impulses[phase],filt,'full')
                indices=np.arange(500,1500)-shift+width//2
                values=np.zeros(1000);good=(indices>=0)&(indices<len(complete));values[good]=complete[indices[good]]
                expected.append(values.var(ddof=1))
            np.testing.assert_allclose(pred,expected,rtol=1e-10,atol=1e-16)

    def test_single_avalanche_self_term_equals_complete_signal_variance(self):
        t=np.arange(2496);impulse=np.exp(-t/45.)-np.exp(-t/3.);kernels=[prefix_kernels(impulse[None,:],w) for w in WINDOWS]
        shift=430;wave=np.r_[np.zeros(shift),impulse[:2496-shift]]*1.7
        metrics=fluctuation_metrics(wave[None,:],[0.],[400.])[0]
        prediction=self_moments(np.array([shift*2.]),np.array([1.7]),400.,kernels,1.)
        np.testing.assert_allclose(prediction,metrics[:,2]-metrics[:,1],rtol=1e-8,atol=1e-15)


if __name__=='__main__':unittest.main(verbosity=2)
