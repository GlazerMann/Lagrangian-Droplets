import numpy as np
import pytest
import microphysics as m
import parcels

@pytest.mark.parametrize('P',[70000.,100000.,101325.])
@pytest.mark.parametrize('SS',[-.5,0.,.02])
def test_mixing_ratio_round_trip_uses_matching_pressure(P,SS):
    q=m.MixingRatio(SS,293.15,P)
    assert m.SuperSaturationRatio(q,293.15,P)==pytest.approx(SS,abs=1e-14)
    assert parcels.SuperSaturation(q,293.15,P)==pytest.approx(SS,abs=1e-14)

def test_initial_mixing_ratio_uses_actual_vapor_pressure_in_denominator():
    _,q,_=m.equilibrate_h2o(np.array([130e-9]),np.array([1.]),np.array([1e6]),293.15,-.5,101325.)
    e=.5*m.thermo.es(20.)
    assert q==pytest.approx(m.c.epsilon*e/(101325.-e))

def test_thermal_callback_receives_mass_pressure_and_accommodation(monkeypatch):
    captured=[]
    def thermal(t,q,Dd,kappa,rho,T,S,accom,**kwargs):
        captured.append((q.copy(),accom,kwargs['p']))
        return np.array([1e-16,2.])
    monkeypatch.setattr(m,'get_odesystem_droplet',thermal)
    args=(0.,.014,.01,293.15,70000.,130e-9,1.,1e6,.4,2160.)
    y=np.array([1e-6,293.15])
    m.one_particle(y,*args)
    m.one_particle_lny(np.log(y),*args)
    m.one_particle_lnDp(np.array([np.log(y[0]),y[1]]),*args)
    for state,accom,P in captured:
        assert state[0]==pytest.approx(5.249331775498865e-16)
        assert accom==.4 and P==70000.

@pytest.mark.parametrize('ignore',[True,False])
def test_growth_representations_obey_the_chain_rule(ignore):
    args=(0.,.014,.01,293.15,101325.,130e-9,1.,1e6,.3,2160.)
    state=np.array([2e-6,293.15]);before=state.copy()
    physical=m.one_particle(state,*args,ignore_Tp=ignore)
    logged=m.one_particle_lny(np.log(state),*args,ignore_Tp=ignore)*state
    mixed=m.one_particle_lnDp(np.array([np.log(state[0]),state[1]]),*args,ignore_Tp=ignore)
    mixed[0]*=state[0]
    # exp(log(T)) differs by rounding; vapor-pressure subtraction amplifies it.
    np.testing.assert_allclose(logged,physical,rtol=1e-10,atol=1e-15)
    np.testing.assert_allclose(mixed,physical,rtol=1e-10,atol=1e-15)
    np.testing.assert_array_equal(state,before)

def test_diffusivity_normalizes_one_atmosphere_exactly_and_is_used_by_growth(monkeypatch):
    T=293.15;r=1e-6;accom=.3
    bulk=1e-4*.211*(T/273.)**1.94
    expected=bulk/(1+bulk/(accom*r)*np.sqrt(2*np.pi*m.c.Mw/(m.c.R*T)))
    assert m.dv(T,r,101325.,accom)==pytest.approx(expected)
    def forbidden(*args,**kwargs):raise AssertionError('Inherited dependency diffusion was used')
    monkeypatch.setattr(m.thermo,'dv',forbidden)
    result=m.one_particle(np.array([2*r,T]),0.,.014,.01,T,101325.,130e-9,1.,1e6,accom,2160.,ignore_Tp=True)
    assert np.all(np.isfinite(result))

def test_all_thermal_representations_integrate_to_consistent_physical_states(tmp_path):
    import run
    from tests.support import parcel
    filename=parcel(tmp_path,times=(0.,.005,.01))
    endpoints=[]
    for mode in ('linear','lny','lnDp'):
        trace=run.one_particle(130e-9,1.,2160.,1e6,filename,avg_SS=.01,
            ignore_Tp=False,run_as_avg=True,smooth=False,odesystem_type=mode,dt=.005)
        endpoints.append([trace['Dp'][-1],trace['Tp'][-1]])
    for endpoint in endpoints[1:]:
        np.testing.assert_allclose(endpoint,endpoints[0],rtol=1e-3)
