import runpy
import process
from tests.support import record

def test_plot_driver_has_a_defined_output_root_and_explicit_arguments(tmp_path,monkeypatch):
    seen=[]
    monkeypatch.setattr(process,'plot_some_trajectories',lambda *args,**kwargs:seen.append((args,kwargs)))
    monkeypatch.setattr(process,'plot_activated_fraction_mean_SS',lambda *args,**kwargs:seen.append((args,kwargs)))
    monkeypatch.setattr('sys.argv',['main_make-figs.py','--output-dir',str(tmp_path),'--plot-dir',str(tmp_path/'figs'),'--parcel-ids','0'])
    namespace=runpy.run_path('main_make-figs.py');namespace['main']()
    assert len(seen)==2
    assert seen[0][0][0]==[0] and seen[0][0][1]==str(tmp_path)

def test_one_trajectory_plot_accepts_a_scalar_axes_result(tmp_path,monkeypatch):
    monkeypatch.setattr(process.run,'read_particle_trace',lambda *args,**kwargs:record())
    monkeypatch.setattr('matplotlib.figure.Figure.savefig',lambda *args,**kwargs:None)
    fig,*axes=process.plot_some_trajectories([0],str(tmp_path),plot_dir=str(tmp_path)+'/')
    assert len(axes)==2
    process.plt.close(fig)
