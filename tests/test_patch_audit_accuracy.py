import numpy as np
import run
from tests.support import parcel

def test_default_formulations_match_an_independent_tighter_solve(tmp_path,monkeypatch):
    filename=parcel(tmp_path,times=[0.,.1,.11,.5,.51,.9,1.],ss=[0.,.1,-.5,-.5,.5,.5,-.5])
    options=dict(avg_SS=0.,smooth=False,ignore_Tp=True,dt=.01)
    normal={mode:run.one_particle(130e-9,1.,2160.,1e6,filename,odesystem_type=mode,**options)
            for mode in ('linear','lny','lnDp')}
    solver=run.solve_ivp
    def tight(*args,**kwargs):
        kwargs.update(rtol=1e-9,atol=1e-11,max_step=.001)
        return solver(*args,**kwargs)
    monkeypatch.setattr(run,'solve_ivp',tight)
    reference=run.one_particle(130e-9,1.,2160.,1e6,filename,odesystem_type='lny',**options)
    for mode,trace in normal.items():
        error=np.max(np.abs(trace['Dp']/reference['Dp']-1.))
        assert error<1e-3,(mode,error)

def test_inactive_log_temperature_does_not_overflow_during_jacobian_probes():
    import microphysics
    args=(0.,.014,.01,293.15,101325.,130e-9,1.,1e6,.3,2160.)
    reference=microphysics.one_particle_lny(np.log([2e-6,293.15]),*args,ignore_Tp=True)
    for inactive in [1000.,-1000.]:
        actual=microphysics.one_particle_lny(np.array([np.log(2e-6),inactive]),*args,ignore_Tp=True)
        np.testing.assert_allclose(actual,reference)
