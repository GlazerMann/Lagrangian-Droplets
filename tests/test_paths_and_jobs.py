import json
from pathlib import Path
import pickle
import runpy
import numpy as np
import pytest
import run
from tracepaths import supersaturation_key,particle_directory
from tests.support import record,parcel

def test_keys_distinguish_scientific_inputs_and_normalize_roundoff():
    assert supersaturation_key(.0011)!=supersaturation_key(.0019)
    assert supersaturation_key(np.linspace(-.04,.05,19)[7])==supersaturation_key(-.005)
    with pytest.raises(ValueError,match='precision'):
        supersaturation_key(.00110000000001)

def test_distinct_mean_runs_keep_both_output_files(tmp_path,monkeypatch):
    source=tmp_path/'source';parcel(source);output=tmp_path/'output'
    monkeypatch.setattr(run,'one_particle',lambda *args,**kwargs:record(ss=kwargs['avg_SS'])['traces'])
    run.main([.0011,.0019],130e-9,1.,2160.,1e6,.5,parcel_trace_dir=str(source),all_particle_trace_dir=str(output))
    files=sorted(output.glob('avgSS_*/particle_*.pkl'))
    assert len(files)==2
    assert sorted(pickle.loads(p.read_bytes())['traces']['SS'][0] for p in files)==[.0011,.0019]

def test_full_particle_id_and_metadata_filtering(tmp_path):
    (tmp_path/'particle_000100000.pkl').write_bytes(pickle.dumps(record()))
    (tmp_path/'supersaturation.json').write_text('{}')
    assert set(run.read_particle_traces(str(tmp_path)))=={100000}

def test_generated_job_preserves_inputs_and_runs_both_controls(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path); calls=[]
    run.split_to_nodes([.0011],3,stop_time=12.,dry_diameter=250e-9,submit=False)
    monkeypatch.setattr(run,'list_parcel_ids',lambda path:[0,1,2,3,4])
    monkeypatch.setattr(run,'main',lambda **kwargs:calls.append(kwargs))
    runpy.run_path('main_node0000.py')
    assert len(calls)==2
    assert [c['constant_SS'] for c in calls]==[False,True]
    for call in calls:
        assert call['dry_diameter']==250e-9
        assert call['duration']==12.
        assert call['parcel_ids']==[0,1,2]
