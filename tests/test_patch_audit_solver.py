from types import SimpleNamespace
import numpy as np
import pytest
import run
from tests.support import parcel

@pytest.mark.parametrize('mode,diameter,temperature',[
    ('linear',-1e-6,293.15),('linear',1e-6,-1.),
    ('linear',65e-9,293.15),('lny',1000.,np.log(293.15))])
def test_success_flag_does_not_accept_invalid_physical_states(tmp_path,monkeypatch,mode,diameter,temperature):
    filename=parcel(tmp_path);calls=[]
    def invalid(*args,**kwargs):
        calls.append(kwargs);times=kwargs['t_eval']
        values=np.tile(np.array([diameter,temperature])[:,None],(1,len(times)))
        return SimpleNamespace(success=True,message='success',t=times,y=values)
    monkeypatch.setattr(run,'solve_ivp',invalid)
    with pytest.raises(RuntimeError,match='nonphysical'):
        run.one_particle(130e-9,1.,2160.,1e6,filename,run_as_avg=True,smooth=False,odesystem_type=mode)
    assert len(calls)==3

def test_explicit_solver_accuracy_controls_reach_the_integrator(tmp_path,monkeypatch):
    filename=parcel(tmp_path);calls=[]
    def constant(rhs,span,initial,**kwargs):
        calls.append(kwargs);times=kwargs['t_eval']
        return SimpleNamespace(success=True,message='success',t=times,
                               y=np.tile(initial[:,None],(1,len(times))))
    monkeypatch.setattr(run,'solve_ivp',constant)
    run.one_particle(130e-9,1.,2160.,1e6,filename,run_as_avg=True,smooth=False,
                     rtol=3e-7,atol=[1e-13,1e-7])
    assert calls[0]['rtol']==3e-7
    np.testing.assert_array_equal(calls[0]['atol'],[1e-13,1e-7])
