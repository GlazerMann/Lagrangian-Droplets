import copy
import pickle
import numpy as np
import pytest
import run
import process
from tests.support import record,parcel

def test_linear_and_log_outputs_have_the_same_physical_units(tmp_path):
    filename=parcel(tmp_path)
    traces=[run.one_particle(130e-9,1.,2160.,1e6,filename,run_as_avg=True,
             smooth=False,odesystem_type=mode,dt=.5) for mode in ('linear','lny')]
    for trace in traces:
        assert trace['state_representation']=='physical'
        assert trace['schema_version']==2
        assert 130e-9<trace['Dp'][0]<1e-5
        assert trace['Tp'][0]==pytest.approx(293.15)
    assert traces[0]['Dp'][0]==pytest.approx(traces[1]['Dp'][0])

def test_legacy_trace_requires_explicit_representation(tmp_path):
    old=record();old['traces'].pop('state_representation');old['traces'].pop('schema_version')
    old['traces']['Dp']=np.log(old['traces']['Dp']);old['traces']['Tp']=np.log(old['traces']['Tp'])
    (tmp_path/'particle_000000000.pkl').write_bytes(pickle.dumps(old))
    with pytest.raises(ValueError,match='legacy'):
        run.read_particle_trace(0,str(tmp_path)+'/')
    actual=run.read_particle_trace(0,str(tmp_path)+'/',legacy_representation='lny')
    assert actual['traces']['Dp'][0]==pytest.approx(2e-6)
    assert actual['traces']['Tp'][0]==pytest.approx(293.15)

def test_plot_uses_ambient_kelvin_and_counts_instantaneous_threshold_once(monkeypatch):
    data={0:record()}; captured=[]
    monkeypatch.setattr(run,'read_particle_traces',lambda *args,**kwargs:data)
    def settling(n,t,d,T,S,**kwargs):
        captured.append((d.copy(),T.copy()))
        return np.ones(len(t)),np.zeros(len(t))
    monkeypatch.setattr(process,'N_vs_t__settling',settling)
    monkeypatch.setattr(process.plt,'show',lambda:None)
    monkeypatch.setattr('matplotlib.figure.Figure.savefig',lambda *args,**kwargs:None)
    fig,ax,lines=process.plot_activated_fraction_mean_SS(np.array([.01]),'unused')
    np.testing.assert_allclose(captured[0][0],2e-6)
    np.testing.assert_allclose(captured[0][1],293.15)
    assert lines[1].get_ydata()[0]==0.
    process.plt.close(fig)

def test_particle_illustration_threshold_uses_shifted_environment_once(monkeypatch):
    normal={0:record(),1:record()};uniform={0:record()}
    monkeypatch.setattr(run,'read_particle_traces',lambda path,*args:uniform if 'uniformS' in path else normal)
    monkeypatch.setattr(process.plt,'show',lambda:None)
    monkeypatch.setattr('matplotlib.figure.Figure.savefig',lambda *args,**kwargs:None)
    fig,ax,lines=process.plot_activated_fraction_mean_SS__with_particles(np.array([.01]),'unused/',
                ss_plot=[.01],N_particles=2,r_closest=0.)
    assert lines[1].get_ydata()[0]==0.
    process.plt.close(fig)
