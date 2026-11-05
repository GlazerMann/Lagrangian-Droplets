import numpy as np
import pytest
import microphysics as m

@pytest.mark.parametrize('temperature',[273.15,293.15,313.15])
def test_kohler_curve_matches_volume_activity_and_kelvin_equation(temperature):
    dry=130e-9;wet=500e-9;kappa=.6
    dry_volume=np.pi/6*dry**3
    water_volume=np.pi/6*(wet**3-dry**3)
    activity=water_volume/(water_volume+kappa*dry_volume)
    kelvin=np.exp(4*m.c.Mw*m.thermo.sigma_w(temperature)/(m.c.R*temperature*m.c.rho_w*wet))
    expected=activity*kelvin-1.
    assert m.thermo.Seq(wet/2,dry/2,temperature,kappa)==pytest.approx(expected,abs=1e-14)

@pytest.mark.parametrize('SS,P',[(-1.01,101325.),(100.,101325.),(0.,1000.),(0.,-1.),(float('nan'),101325.)])
def test_humidity_conversion_rejects_impossible_vapor_states(SS,P):
    with pytest.raises(ValueError):m.MixingRatio(SS,293.15,P)

def test_dry_vapor_limit_has_zero_water_mass():
    assert m.MixingRatio(-1.,293.15,101325.)==0.

@pytest.mark.parametrize('diameters,kappas,numbers',[
    ([130e-9],[0.],[1e6]),([130e-9,200e-9],[1.],[1e6,1e6]),
    ([130e-9],[1.],[-1.]),([-130e-9],[1.],[1e6])])
def test_equilibrium_rejects_unsupported_or_inconsistent_populations(diameters,kappas,numbers):
    with pytest.raises(ValueError,match='Equilibrium requires'):
        m.equilibrate_h2o(diameters,kappas,numbers,293.15,0.,101325.)

@pytest.mark.parametrize('rho',[800.,1000.,2160.])
def test_diameter_mass_inverse_and_added_water_mass(rho):
    dry=130e-9;wet=1e-6
    mass=m.mass_from_diameter(wet,dry,rho)
    dry_mass=np.pi/6*rho*dry**3
    assert mass-dry_mass==pytest.approx(np.pi/6*1000.*(wet**3-dry**3))
    recovered=((6*mass/np.pi-(rho-1000.)*dry**3)/1000.)**(1/3)
    assert recovered==pytest.approx(wet,rel=1e-14)

@pytest.mark.parametrize('SS',[-.5,0.])
def test_initialized_wet_particle_is_a_stationary_quasi_steady_solution(SS):
    T=293.15;dry=130e-9
    wet,q,_=m.equilibrate_h2o([dry],[1.],[1e6],T,SS,101325.)
    derivative=m.one_particle(np.array([wet[0],T]),0.,q,SS,T,101325.,dry,1.,1e6,.3,2160.,ignore_Tp=True)
    assert abs(derivative[0])<1e-15

def test_humid_heat_capacity_uses_moist_mass_fractions_and_si_units():
    T=293.15;P=101325.;S=1.
    vapor_pressure=m.saturation_vapor_pressure(T)
    q=(vapor_pressure*m.M_h2o)/(vapor_pressure*m.M_h2o+(P-vapor_pressure)*m.M_dry_air)
    dry=m.air_heat_capacity(0.,T,p=P)
    actual=m.air_heat_capacity(S,T,p=P)
    assert actual==pytest.approx((1-q)*dry+q*1860.)
    assert actual>dry+10.
