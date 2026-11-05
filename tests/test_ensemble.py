import copy
import pickle
from pathlib import Path
import numpy as np
import pytest
import run
from tests.support import parcel,record

def test_independent_returned_and_saved_histories(tmp_path,monkeypatch):
    source=tmp_path/'source';out=tmp_path/'out';out.mkdir(parents=True)
    for i in [0,1]:parcel(source,i)
    def simulate(*args,**kwargs):
        i=int(Path(args[4]).stem[-6:])
        return record(diameter=(i+1)*1e-6)['traces']
    monkeypatch.setattr(run,'one_particle',simulate)
    data=run.many_trajectories(130e-9,1.,2160.,1e6,add_ignore_Tp=False,
          parcel_trace_dir=str(source)+'/',all_particle_trace_dir=str(out))
    assert data[0] is not data[1]
    assert data[0]['traces']['Dp'][0]==1e-6
    assert data[1]['traces']['Dp'][0]==2e-6
    data[1]['traces']['Dp'][0]=9e-6
    assert data[0]['traces']['Dp'][0]==1e-6
    files=sorted(out.glob('avgSS_*/*.pkl'))
    assert [pickle.loads(p.read_bytes())['traces']['Dp'][0] for p in files]==[1e-6,2e-6]

@pytest.mark.parametrize('times',[(0.,1.),(0.,10.,20.)])
def test_incompatible_time_grids_are_rejected(times):
    with pytest.raises(ValueError,match='time grid'):
        run.unravel_particle_traces({0:record((0.,1.,2.)),1:record(times)})

def test_single_sample_keeps_actual_timestamp():
    values=run.unravel_particle_trace(record((5.,)))
    np.testing.assert_array_equal(values[0],[5.])
