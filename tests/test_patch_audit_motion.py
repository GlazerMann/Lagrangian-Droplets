import pickle
import numpy as np
import pytest
from netCDF4 import Dataset
import parcels
from tests.support import les

def prepare(source,times,velocity):
    les(source,times)
    for index,time in enumerate(times):
        with Dataset(str(source/('PiChamber_huji_19K_trj_32_%010d.nc'%index)),'a') as ds:
            x=np.asarray(ds['x'][:])[None,None,None,:]
            ds['TABS'][:]=285.+2*x
            ds['U'][:]=velocity(time,x)

def generate(source,out,times,duration):
    parcels.trajectories(1,duration,LES_dir=str(source),parcel_trace_dir=str(out),
                         timesteps=range(len(times)),rng=np.random.default_rng(3))
    return pickle.loads((out/'parcel_traces_000000.pkl').read_bytes())

def test_time_dependent_velocity_and_spatial_thermodynamics(tmp_path):
    source=tmp_path/'LES';times=[100.,100.25,100.5,100.75,101.]
    prepare(source,times,lambda time,x:time-100.)
    trace=generate(source,tmp_path/'out',times,1.)
    initial=np.random.default_rng(3).uniform(0.,2.)
    expected=initial+.5*(trace['time']-100.)**2
    np.testing.assert_allclose(trace['x'],expected,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(trace['T'],285.+2*expected,rtol=1e-12)
    np.testing.assert_allclose(trace['SS'],parcels.SuperSaturation(.014,trace['T']),rtol=1e-12)

def test_spatial_velocity_interpolation_matches_exponential_solution(tmp_path):
    source=tmp_path/'LES';times=[0.,.25,.5,.75,1.]
    prepare(source,times,lambda time,x:.5*x)
    trace=generate(source,tmp_path/'out',times,1.)
    expected=np.random.default_rng(3).uniform(0.,2.)*np.exp(.5*trace['time'])
    np.testing.assert_allclose(trace['x'],expected,rtol=1e-6)

def test_moving_parcel_crosses_multiple_reflective_boundaries(tmp_path):
    source=tmp_path/'LES';times=[0.,.25,.5]
    prepare(source,times,lambda time,x:10.)
    trace=generate(source,tmp_path/'out',times,.5)
    unfolded=np.random.default_rng(3).uniform(0.,2.)+10.*trace['time']
    phase=np.mod(unfolded,4.);expected=np.where(phase<=2.,phase,4.-phase)
    np.testing.assert_allclose(trace['x'],expected,atol=1e-10)
    assert np.all((trace['x']>=0.)&(trace['x']<=2.))
    np.testing.assert_allclose(trace['T'],285.+2*expected,rtol=1e-12)

def test_equal_exposures_remain_equal_when_absolute_clocks_cross_a_binary_scale(tmp_path):
    from timeutils import load_environment
    source=tmp_path/'LES';out=tmp_path/'out';les(source,[1000.,1000.1,1060.2])
    parcels.trajectories(4,60.,LES_dir=str(source),parcel_trace_dir=str(out),
                         timesteps=range(3),sample_dt=10.,max_step=10.)
    for filename in out.glob('parcel_traces_*.pkl'):
        trace=pickle.loads(filename.read_bytes())
        assert trace['duration_seconds']==60.
        assert trace['elapsed_time'][-1]==60.
        assert load_environment(str(filename))[0][-1]==60.
