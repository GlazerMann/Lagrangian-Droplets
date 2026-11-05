import os
import numpy as np
import pytest
from netCDF4 import Dataset
from verification import helper

def test_statistics_include_all_files_and_save_the_final_accumulation(tmp_path):
    all_samples=[]
    for index in range(101):
        value=0. if index==0 else .02;all_samples.extend([value]*11)
        with Dataset(str(tmp_path/('traces_%06d.nc'%index)),'w') as ds:
            ds.createDimension('time',11);ds.createVariable('s_env','f8',('time',))[:]=value
    helper.store_s_stats(str(tmp_path))
    actual=np.loadtxt(tmp_path/'s_stats.txt')
    np.testing.assert_allclose(actual,[np.mean(all_samples),np.std(all_samples)])
    assert np.loadtxt(tmp_path/'log10s_stats.txt')[0]>0.

def test_small_directories_are_not_silently_ignored(tmp_path):
    with Dataset(str(tmp_path/'traces_000003.nc'),'w') as ds:
        ds.createDimension('time',11);ds.createVariable('s_env','f8',('time',))[:]=.03
    helper.store_s_stats(str(tmp_path))
    assert np.loadtxt(tmp_path/'s_stats.txt')[0]==pytest.approx(.03)

def test_comparison_column_does_not_depend_on_directory_order(tmp_path,monkeypatch):
    (tmp_path/'s_stats.txt').write_text('0\n0\n')
    with Dataset(str(tmp_path/'traces_000000.nc'),'w') as ds:
        ds.createDimension('time',2);ds.createDimension('s_mean',1)
        for key,values in {'time':[0.,1.],'s_env':[0.,0.],'Dp':[1e-6,1e-6],
                           'T_env':[293.15,293.15],'Ddry':[130e-9,130e-9],'kappa':[1.,1.]}.items():
            ds.createVariable(key,'f8',('time',))[:]=values
        ds.createVariable('s_offset','f8',('s_mean',))[:]=[0.]
        ds.createVariable('Dp_predict','f8',('time','s_mean'))[:]=np.full((2,1),1e-6)
    real=os.listdir
    monkeypatch.setattr(helper.os,'listdir',lambda path:['s_stats.txt','traces_000000.nc'] if os.path.abspath(path)==str(tmp_path) else real(path))
    monkeypatch.setattr(helper.plt,'show',lambda:None)
    monkeypatch.setattr('matplotlib.figure.Figure.savefig',lambda *args,**kwargs:None)
    critical=helper.dsd_figs(str(tmp_path)+'/',.5,plot_trajectories=False,fig_overdir=str(tmp_path/'figures')+'/')
    assert len(critical)==1 and critical[0]>130e-9
    helper.plt.close('all')

def test_verification_bulk_diffusivity_has_correct_one_atmosphere_limit():
    assert helper.dv_cont(273.,101325.)==pytest.approx(.211e-4)
