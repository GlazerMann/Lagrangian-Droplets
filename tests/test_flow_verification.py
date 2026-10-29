from types import SimpleNamespace
import pickle
import numpy as np
import pytest
from netCDF4 import Dataset
import parcels
from tests.support import les

def filename(root,index):return root/('PiChamber_huji_19K_trj_32_%010d.nc'%index)

def generate(source,out,steps,duration,**kwargs):
    parcels.trajectories(1,duration,LES_dir=source,parcel_trace_dir=out,timesteps=steps,
                         rng=np.random.default_rng(3),**kwargs)
    return pickle.loads((out/'parcel_traces_000000.pkl').read_bytes())

@pytest.mark.parametrize('descending',[False,True])
def test_coupled_3d_advection_and_scalars_on_unequal_grids(tmp_path,descending):
    source=tmp_path/'LES';source.mkdir();out=tmp_path/'out'
    grids=[np.array([-1.,0.,.5,2.]),np.array([10.,10.3,11.]),np.array([-4.,-2.,-1.,1.,3.])]
    if descending:grids=[g[::-1] for g in grids]
    for index,times in enumerate(([100.,100.2],[100.4,100.6])):
        with Dataset(filename(source,index),'w') as ds:
            for name,grid in zip(('x','y','z'),grids):
                ds.createDimension(name,len(grid));ds.createVariable(name,'f8',(name,))[:]=grid
            ds.createDimension('time',len(times));ds.createVariable('time','f8',('time',))[:]=times
            t,z,y,x=np.meshgrid(times,grids[2],grids[1],grids[0],indexing='ij')
            fields=dict(U=.12*(y-10.5),V=.08*(z+1.),W=np.full(x.shape,.03),
                        TABS=280.+.2*x+.3*y+.4*z+.5*(t-100.),
                        QV=14.+.1*x+.2*y+.3*z+.4*(t-100.))
            for name,values in fields.items():ds.createVariable(name,'f8',('time','z','y','x'))[:]=values
    trace=generate(source,out,range(2),.5,sample_dt=.05)
    rng=np.random.default_rng(3);x0,y0,z0=[rng.uniform(min(g),max(g)) for g in grids]
    t=trace['elapsed_time'];z=z0+.03*t
    y=y0+.08*(z0+1)*t+.5*.08*.03*t**2
    x=x0+.12*(y0-10.5)*t+.5*.12*.08*(z0+1)*t**2+.12*.08*.03*t**3/6
    for actual,expected in zip((trace['x'],trace['y'],trace['z']),(x,y,z)):
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-11)
    expected_T=280.+.2*x+.3*y+.4*z+.5*t
    expected_q=(14.+.1*x+.2*y+.3*z+.4*t)/1000.
    np.testing.assert_allclose(trace['T'],expected_T,rtol=1e-12)
    np.testing.assert_allclose(trace['mixing ratio'],expected_q,rtol=1e-12)
    np.testing.assert_allclose(trace['SS'],parcels.SuperSaturation(expected_q,expected_T),rtol=1e-12)

def test_solid_body_rotation_preserves_radius_and_phase(tmp_path):
    source=tmp_path/'LES';out=tmp_path/'out';les(source,[0.,1.5,3.])
    for index in range(3):
        with Dataset(filename(source,index),'a') as ds:
            x=np.asarray(ds['x'][:])[None,None,None,:];y=np.asarray(ds['y'][:])[None,None,:,None]
            ds['U'][:]=-20*(y-1);ds['V'][:]=20*(x-1)
    trace=generate(source,out,range(3),3.)
    orbit=(trace['x']-1)+1j*(trace['y']-1)
    expected=orbit[0]*np.exp(20j*trace['elapsed_time'])
    assert np.max(np.abs(np.abs(orbit)/abs(orbit[0])-1))<1e-4
    assert np.max(np.abs(orbit-expected))<1e-4

@pytest.mark.parametrize('field,value',[('QV','masked'),('TABS','masked'),('U','masked'),
    ('QV',-1.),('TABS',-1.),('W',float('nan'))])
def test_bad_source_fields_are_rejected_before_writing(tmp_path,field,value):
    source=tmp_path/'LES';out=tmp_path/'out';les(source,[0.,.1])
    for index in range(2):
        with Dataset(filename(source,index),'a') as ds:
            variable=ds[field]
            variable[:]=np.ma.masked_all(variable.shape) if value=='masked' else value
    with pytest.raises(ValueError,match='LES'):generate(source,out,range(2),.1)
    assert not out.exists()

def test_equal_sized_axes_do_not_hide_wrong_dimension_order(tmp_path):
    source=tmp_path/'LES';out=tmp_path/'out';les(source,[0.,.1])
    for index in range(2):
        with Dataset(filename(source,index),'a') as ds:
            ds.renameVariable('TABS','old_TABS')
            ds.createVariable('TABS','f8',('time','x','y','z'))[:]=293.15
    with pytest.raises(ValueError,match='dimensions'):generate(source,out,range(2),.1)
    assert not out.exists()

@pytest.mark.parametrize('options',[{'P':0.},{'rtol':0.},{'atol':[-1.,1.,1.]}])
def test_invalid_flow_configuration_fails_before_outputs(tmp_path,options):
    source=tmp_path/'LES';out=tmp_path/'out';les(source,[0.,.1])
    with pytest.raises(ValueError):generate(source,out,range(2),.1,**options)
    assert not out.exists()

@pytest.mark.parametrize('bad_output',['nonfinite','wrong_time'])
def test_success_status_cannot_accept_invalid_flow_output(tmp_path,monkeypatch,bad_output):
    source=tmp_path/'LES';out=tmp_path/'out';les(source,[0.,.1])
    def invalid(rhs,span,initial,**kwargs):
        t=kwargs['t_eval'].copy();y=np.tile(initial[:,None],(1,len(t)))
        if bad_output=='nonfinite':y[0,-1]=np.nan
        else:t[-1]-=.01
        return SimpleNamespace(success=True,message='injected invalid output',t=t,y=y)
    monkeypatch.setattr(parcels,'solve_ivp',invalid)
    with pytest.raises(RuntimeError,match='injected invalid output'):generate(source,out,range(2),.1)
    assert (out/'parcel_run_in_progress.json').exists()
    assert not list(out.glob('parcel_traces_*.pkl'))

def test_explicit_flow_tolerances_reach_solver(tmp_path,monkeypatch):
    source=tmp_path/'LES';out=tmp_path/'out';les(source,[0.,.1]);calls=[]
    solver=parcels.solve_ivp
    def capture(*args,**kwargs):calls.append(kwargs);return solver(*args,**kwargs)
    monkeypatch.setattr(parcels,'solve_ivp',capture)
    generate(source,out,range(2),.1,rtol=2e-7,atol=[1e-10,2e-10,3e-10])
    assert calls[0]['rtol']==2e-7
    np.testing.assert_array_equal(calls[0]['atol'],[1e-10,2e-10,3e-10])
