import numpy as np
import pytest
import run
from tests.support import parcel,record
from timeutils import trace_mean

def test_smoothing_flag_does_not_add_noise(tmp_path):
    filename=parcel(tmp_path,times=np.linspace(0.,1.,200))
    a=run.environmental_trace_funs(filename,smooth=False)
    b=run.environmental_trace_funs(filename,smooth=False)
    np.testing.assert_array_equal(a[1]([0.,.5,1.]),[293.15]*3)
    np.testing.assert_array_equal(a[1]([0.,.5,1.]),b[1]([0.,.5,1.]))

def test_time_mean_and_legacy_sample_mean_are_explicit(tmp_path):
    filename=parcel(tmp_path,times=[0.,1.,100.],ss=[-.01,.03,.03])
    values=run.environmental_trace_funs(filename,smooth=False,avg_SS=0.)[2]([0.,1.,100.])
    assert trace_mean([0.,1.,100.],values)==pytest.approx(0.,abs=1e-15)
    legacy=run.environmental_trace_funs(filename,smooth=False,mean_definition='sample')[2]([0.,1.,100.])
    assert np.mean(legacy)==pytest.approx(0.,abs=1e-15)

def test_noise_is_reproducible_when_explicitly_requested(tmp_path):
    filename=parcel(tmp_path)
    a=run.environmental_trace_funs(filename,smooth=False,temperature_noise_fraction=.01,rng=np.random.default_rng(1))
    b=run.environmental_trace_funs(filename,smooth=False,temperature_noise_fraction=.01,rng=np.random.default_rng(1))
    np.testing.assert_array_equal(a[1]([0.,.5,1.]),b[1]([0.,.5,1.]))

def test_paired_calculations_share_one_environment(tmp_path,monkeypatch):
    source=tmp_path/'source';out=tmp_path/'out';out.mkdir(parents=True)
    parcel(source);seen=[]
    def simulate(*args,**kwargs):
        seen.append(kwargs['environment'])
        return record()['traces']
    monkeypatch.setattr(run,'one_particle',simulate)
    run.many_trajectories(130e-9,1.,2160.,1e6,smooth=False,
        temperature_noise_fraction=.01,rng=np.random.default_rng(1),
        parcel_trace_dir=str(source)+'/',all_particle_trace_dir=str(out))
    assert len(seen)==2 and seen[0] is seen[1]

def test_duration_crop_precedes_mean_calculation(tmp_path):
    filename=parcel(tmp_path,times=[0.,1.,10.],ss=[0.,.01,.9])
    environment=run.environmental_trace_funs(filename,smooth=False,avg_SS=.02,duration=1.)
    assert environment[3]==1.
    assert trace_mean([0.,1.],environment[2]([0.,1.]))==pytest.approx(.02)
