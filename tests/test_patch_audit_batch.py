import json
from pathlib import Path
import runpy
import pytest
import run
from tests.support import parcel,record
from tracepaths import particle_directory

def test_path_objects_survive_generated_job_serialization(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);calls=[]
    run.split_to_nodes([.01],1,parcel_trace_dir=tmp_path/'source',
        all_particle_trace_dir=tmp_path/'output',all_particle_trace_dir_uniformS=tmp_path/'uniform')
    monkeypatch.setattr(run,'list_parcel_ids',lambda path:[0])
    monkeypatch.setattr(run,'main',lambda **kwargs:calls.append(kwargs))
    runpy.run_path('main_node0000.py')
    assert len(calls)==2 and calls[0]['parcel_trace_dir']==str(tmp_path/'source')

def test_generator_supersaturation_input_is_not_consumed_twice(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert len(run.split_to_nodes((value for value in [.01,.02]),1))==2

def test_job_refuses_to_silently_reduce_the_requested_population(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);calls=[]
    run.split_to_nodes([.01],3)
    monkeypatch.setattr(run,'list_parcel_ids',lambda path:[0,1])
    monkeypatch.setattr(run,'main',lambda **kwargs:calls.append(kwargs))
    with pytest.raises(ValueError,match='only 2'):
        runpy.run_path('main_node0000.py')
    assert not calls

def simulate(root,source,**kwargs):
    return run.many_trajectories(130e-9,1.,2160.,1e6,ignore_Tp=True,smooth=False,
        add_ignore_Tp=False,parcel_trace_dir=str(source),all_particle_trace_dir=str(root),**kwargs)

def test_old_files_cannot_change_a_new_population_selection(tmp_path,monkeypatch):
    source=tmp_path/'source';out=tmp_path/'out'
    for index in [0,1]:parcel(source,index)
    monkeypatch.setattr(run,'one_particle',lambda *args,**kwargs:record()['traces'])
    simulate(out,source)
    with pytest.raises(ValueError,match='outside this run selection'):
        simulate(out,source,part_ii=[0])
    assert set(run.read_particle_traces(particle_directory(str(out),0.)))=={0,1}

def test_empty_source_selection_is_a_failure(tmp_path):
    source=tmp_path/'source';source.mkdir();out=tmp_path/'out'
    with pytest.raises(ValueError,match='No parcels'):
        simulate(out,source)
    assert not out.exists()

def test_partial_rerun_is_unreadable_until_successful_recovery(tmp_path,monkeypatch):
    source=tmp_path/'source';out=tmp_path/'out'
    for index in [0,1]:parcel(source,index)
    good=lambda *args,**kwargs:record()['traces']
    monkeypatch.setattr(run,'one_particle',good);simulate(out,source)
    def fail_second(*args,**kwargs):
        if Path(args[4]).stem.endswith('000001'):raise RuntimeError('injected failure')
        return record()['traces']
    monkeypatch.setattr(run,'one_particle',fail_second)
    with pytest.raises(RuntimeError,match='injected failure'):simulate(out,source)
    directory=particle_directory(str(out),0.)
    assert json.loads((Path(directory)/'run_in_progress.json').read_text())['status']=='in_progress'
    with pytest.raises(RuntimeError,match='incomplete'):run.read_particle_traces(directory)
    with pytest.raises(RuntimeError,match='incomplete'):run.read_particle_trace(0,directory)
    monkeypatch.setattr(run,'one_particle',good);simulate(out,source)
    assert set(run.read_particle_traces(directory))=={0,1}

def test_duplicate_output_keys_are_rejected_before_generating_jobs(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError,match='distinct'):run.split_to_nodes([.01,.01],1)
    assert not list(tmp_path.glob('main_node*.py'))

def test_invalid_job_parameters_do_not_create_broken_scripts(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError,match='Invalid scientific'):run.split_to_nodes([.01],1,dt=float('nan'))
    assert not list(tmp_path.glob('main_node*.py'))
