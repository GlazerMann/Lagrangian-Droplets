import numpy as np
import run
import parcels
import process
from tracepaths import particle_directory
from tests.support import les

def test_synthetic_les_to_physical_traces_and_activation_plot(tmp_path,monkeypatch):
    source=tmp_path/'LES';parcel_root=tmp_path/'parcels';output=tmp_path/'particle_traces'
    les(source,[100.,100.25,100.5,100.75,101.],peak=2)
    parcels.trajectories(2,.5,LES_dir=str(source),parcel_trace_dir=str(parcel_root),timesteps=range(5))
    ensemble=run.main([.01],130e-9,1.,2160.,1e6,.1,parcel_trace_dir=str(parcel_root),
                     all_particle_trace_dir=str(output),duration=.5)[0]
    assert ensemble[0] is not ensemble[1]
    assert not np.array_equal(ensemble[0]['traces']['Dp'],ensemble[1]['traces']['Dp'])
    saved=run.read_particle_traces(particle_directory(str(output),.01))
    times,diameters,temperatures,*_=run.unravel_particle_traces(saved)
    assert times[0]==0. and times[-1]==.5
    assert np.all(diameters>130e-9) and np.all(diameters<1e-3)
    np.testing.assert_allclose(temperatures,293.15)
    monkeypatch.setattr(process.plt,'show',lambda:None)
    monkeypatch.setattr('matplotlib.figure.Figure.savefig',lambda *args,**kwargs:None)
    fig,ax,lines=process.plot_activated_fraction_mean_SS(np.array([.01]),str(tmp_path))
    assert np.all(np.isfinite(lines[0].get_ydata()))
    assert 0.<=lines[0].get_ydata()[-1]<=1.
    process.plt.close('all')

def test_absolute_clock_endpoint_roundoff_does_not_reject_valid_duration(tmp_path):
    source=tmp_path/'LES';parcel_root=tmp_path/'parcels';output=tmp_path/'particles'
    les(source,[.1,.2,.3]) # .1 + .2 is slightly greater than .3 in binary floating point
    parcels.trajectories(1,.2,LES_dir=str(source),parcel_trace_dir=str(parcel_root),timesteps=range(3))
    result=run.main([0.],130e-9,1.,2160.,1e6,.1,parcel_trace_dir=str(parcel_root),
                    all_particle_trace_dir=str(output),duration=.2)[0][0]['traces']
    np.testing.assert_array_equal(result['t'],[0.,.1,.2])
