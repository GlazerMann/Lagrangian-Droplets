from pathlib import Path
import pickle
import numpy as np

def record(times=(0.,1.), diameter=2e-6, ss=.01):
    t=np.asarray(times,dtype=float);n=len(t)
    return {'traces': {'t':t,'Dp':np.full(n,diameter),'Tp':np.full(n,293.15),
            'T':np.full(n,293.15),'mixing_ratio':np.full(n,.014),'SS':np.full(n,ss),
            'state_representation':'physical','schema_version':2},
            'props': {'Ddry':130e-9,'kappa':1.,'density':2160.,'N':1e6,
                      'Dp_crit':1e-6,'SS_crit':.015}}

def parcel(folder,index=0,times=(0.,.5,1.),temperature=293.15,ss=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    t=np.asarray(times,dtype=float)
    data={'time':t,'T':np.full(len(t),temperature),'SS':np.zeros(len(t)) if ss is None else np.asarray(ss),
          'mixing ratio':np.full(len(t),.014)}
    filename=folder/('parcel_traces_%06d.pkl'%index)
    filename.write_bytes(pickle.dumps(data))
    return str(filename)

def les(folder,times,peak=None):
    from netCDF4 import Dataset
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    grid=np.array([0.,1.,2.])
    for i,t in enumerate(times):
        with Dataset(str(folder/('PiChamber_huji_19K_trj_32_%010d.nc'%i)),'w') as ds:
            for dim,size in [('time',1),('x',3),('y',3),('z',3)]: ds.createDimension(dim,size)
            for dim in ['x','y','z']:ds.createVariable(dim,'f8',(dim,))[:]=grid
            ds.createVariable('time','f8',('time',))[:]=[t]
            for name,value in [('U',0.),('V',0.),('W',0.),('TABS',293.15),('QV',20. if i==peak else 14.)]:
                ds.createVariable(name,'f8',('time','z','y','x'))[:]=value
