from types import SimpleNamespace
import pickle
import pytest
import parcels
import run
from timeutils import load_environment
from tests.support import les

def prepare(tmp_path):
    source=tmp_path/'LES';out=tmp_path/'parcels';les(source,[0.,.1,.2])
    return source,out

def generate(source,out,count=2,**kwargs):
    parcels.trajectories(count,.1,LES_dir=source,parcel_trace_dir=out,timesteps=range(3),**kwargs)

def test_smaller_parcel_regeneration_cannot_leave_old_population_members(tmp_path):
    source,out=prepare(tmp_path);generate(source,out)
    before=(out/'parcel_traces_000000.pkl').read_bytes()
    with pytest.raises(ValueError,match='outside this run population'):generate(source,out,1)
    assert (out/'parcel_traces_000000.pkl').read_bytes()==before
    assert run.list_parcel_ids(out)==[0,1]

def test_interrupted_parcel_run_blocks_all_readers_and_recovers(tmp_path,monkeypatch):
    source,out=prepare(tmp_path);generate(source,out)
    solver=parcels.solve_ivp;calls=[]
    def fail_second(*args,**kwargs):
        calls.append(1)
        if len(calls)==2:return SimpleNamespace(success=False,message='injected failure',t=[])
        return solver(*args,**kwargs)
    monkeypatch.setattr(parcels,'solve_ivp',fail_second)
    with pytest.raises(RuntimeError,match='injected failure'):generate(source,out)
    for reader in (lambda:run.list_parcel_ids(out),lambda:run.read_parcel_trace(0,out),
                   lambda:load_environment(out/'parcel_traces_000000.pkl')):
        with pytest.raises(RuntimeError,match='incomplete'):reader()
    with pytest.raises(ValueError,match='complete earlier'):generate(source,out,num_start=1)
    monkeypatch.setattr(parcels,'solve_ivp',solver);generate(source,out)
    assert run.list_parcel_ids(out)==[0,1]
    assert not (out/'parcel_run_in_progress.json').exists()
    assert pickle.loads((out/'parcel_traces_000001.pkl').read_bytes())['duration_seconds']==.1

def test_restart_requires_skipped_population_members_to_exist(tmp_path):
    source,out=prepare(tmp_path)
    with pytest.raises(ValueError,match='complete earlier'):generate(source,out,num_start=1)
    assert not out.exists()
