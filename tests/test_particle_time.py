from types import SimpleNamespace
import numpy as np
import pytest
import run
from timeutils import output_times
from tests.support import parcel

def test_absolute_source_times_become_elapsed_particle_age(tmp_path):
    filename=parcel(tmp_path,times=(10.,10.5,11.))
    result=run.one_particle(130e-9,1.,2160.,1e6,filename,run_as_avg=True,
                            smooth=False,odesystem_type='lny',dt=.5)
    np.testing.assert_array_equal(result['t'],[0.,.5,1.])

def test_output_grid_preserves_endpoint_without_stretching_dt():
    np.testing.assert_array_equal(output_times(.2,1.),[0.,.2])
    np.testing.assert_allclose(output_times(1.3,.5),[0.,.5,1.,1.3])
    with pytest.raises(ValueError):output_times(1.,0.)

def test_failed_results_are_retried_and_raise_instead_of_exiting(tmp_path,monkeypatch):
    filename=parcel(tmp_path);calls=[]
    def fail(*args,**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(success=False,message='step size failed',t=np.array([0.]),y=np.ones((2,1)))
    monkeypatch.setattr(run,'solve_ivp',fail)
    with pytest.raises(RuntimeError,match='step size failed'):
        run.one_particle(130e-9,1.,2160.,1e6,filename,run_as_avg=True,smooth=False)
    assert len(calls)==3

def test_original_exception_is_retained_as_cause(tmp_path,monkeypatch):
    filename=parcel(tmp_path)
    def fail(*args,**kwargs):raise ValueError('original numerical failure')
    monkeypatch.setattr(run,'solve_ivp',fail)
    with pytest.raises(RuntimeError,match='original numerical failure') as error:
        run.one_particle(130e-9,1.,2160.,1e6,filename,run_as_avg=True,smooth=False)
    assert isinstance(error.value.__cause__,ValueError)
