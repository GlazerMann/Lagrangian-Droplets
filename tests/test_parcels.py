import pickle
import numpy as np
import pytest
import parcels
from tests.support import les

def test_one_parcel_nonzero_time_origin_and_duration(tmp_path):
    source=tmp_path/'LES';out=tmp_path/'out';les(source,[100.,100.25,100.5,100.75,101.])
    parcels.trajectories(1,.5,LES_dir=str(source),parcel_trace_dir=str(out),timesteps=range(5))
    trace=pickle.loads((out/'parcel_traces_000000.pkl').read_bytes())
    assert trace['time'][0]==100. and trace['time'][-1]==100.5
    assert trace['elapsed_time'][-1]==.5

def test_unavailable_duration_is_rejected_before_writing(tmp_path):
    source=tmp_path/'LES';out=tmp_path/'out';les(source,[0.,.5,1.])
    with pytest.raises(ValueError,match='cover'):
        parcels.trajectories(1,2.,LES_dir=str(source),parcel_trace_dir=str(out),timesteps=range(3))
    assert not out.exists()

def test_stationary_parcel_retains_resolved_supersaturation_peak(tmp_path):
    source=tmp_path/'LES';out=tmp_path/'out';les(source,[0.,.25,.5,.75,1.],peak=2)
    parcels.trajectories(1,1.,LES_dir=str(source),parcel_trace_dir=str(out),timesteps=range(5))
    trace=pickle.loads((out/'parcel_traces_000000.pkl').read_bytes())
    assert .5 in trace['time']
    assert np.max(trace['SS'])==pytest.approx(parcels.SuperSaturation(.02,293.15))
    assert np.max(np.diff(trace['time']))<=.1+1e-14

def test_boundary_mapping_is_pure_and_handles_multiple_crossings():
    grid=np.array([0.,1.]);position=np.array([3.2,-2.2,.5]);before=position.copy()
    actual=parcels.RBC(position,grid,grid,grid)
    np.testing.assert_array_equal(position,before)
    np.testing.assert_allclose(actual,[.8,.2,.5])
    position.setflags(write=False)
    parcels.find_Velocity(0.,position,grid,grid,grid,lambda q:1.,lambda q:0.,lambda q:0.)
