import numpy as np
import pytest
import process

def test_drag_has_no_discontinuity_at_reynolds_one():
    D=80e-6;T=293.15
    velocity=process.get_dynamic_viscosity_of_air(T)/(process.get_air_density(T,1.)*D)
    below=process.get_drag_coefficient(velocity*(1-1e-8),D,T,1.)
    above=process.get_drag_coefficient(velocity*(1+1e-8),D,T,1.)
    assert below==pytest.approx(above,rel=1e-7)

@pytest.mark.parametrize('P',[70000.,101325.,120000.])
def test_settling_sweep_crosses_old_root_gap_and_satisfies_force_balance(P):
    T=293.15;rho=process.get_air_density(T,1.,p=P);mu=process.get_dynamic_viscosity_of_air(T)
    diameters=np.r_[np.geomspace(1e-8,1e-3,80),80e-6]
    for D in diameters:
        v=float(process.get_terminal_velocity(D,1000.,T,1.,p=P).item())
        Re=rho*v*D/mu
        drag=3*np.pi*mu*D*v*(1+.15*Re**.687)
        gravity=np.pi/6*D**3*1000.*process.c.g
        assert drag==pytest.approx(gravity,rel=1e-8,abs=1e-30)

def test_terminal_speed_converges_to_stokes_limit():
    D=1e-9;T=293.15
    actual=float(process.get_terminal_velocity(D,1000.,T,1.).item())
    limit=1000.*process.c.g*D**2/(18*process.get_dynamic_viscosity_of_air(T))
    assert actual==pytest.approx(limit,rel=1e-8,abs=0.)
