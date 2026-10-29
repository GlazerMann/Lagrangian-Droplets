import numpy as np
import pytest
import process

@pytest.mark.parametrize('P',[70000.,101325.,120000.])
def test_low_reynolds_speed_uses_consistent_pressure(P):
    D=2e-6;T=293.15
    actual=float(process.get_terminal_velocity(D,1000.,T,1.,p=P).item())
    from scipy.optimize import brentq
    mu=process.get_dynamic_viscosity_of_air(T)
    rho=process.get_air_density(T,1.,p=P)
    stokes=1000.*process.c.g*D**2/(18*mu)
    # Independent scalar force balance, rather than assuming finite-Re drag
    # is exactly the zero-Re Stokes limit.
    reference=brentq(lambda v:v*(1+.15*(rho*v*D/mu)**.687)-stokes,0.,stokes,xtol=1e-18)
    assert actual==pytest.approx(reference,rel=1e-8)

def test_settling_survival_uses_every_actual_interval(monkeypatch):
    monkeypatch.setattr(process,'get_terminal_velocity',lambda *args,**kwargs:np.array([.1]))
    result,speed=process.N_vs_t__settling(1.,np.array([0.,1.,10.]),np.full(3,2e-6),np.full(3,293.15),np.ones(3))
    assert result[-1]==pytest.approx(np.exp(-1.))
    single,_=process.N_vs_t__settling(1.,np.array([0.]),np.array([2e-6]),np.array([293.15]),np.ones(1))
    assert single[0]==1.

def test_failed_terminal_velocity_solve_is_not_accepted(monkeypatch):
    monkeypatch.setattr(process,'fsolve',lambda *args,**kwargs:(np.array([1.]),{},5,'no progress'))
    with pytest.raises(RuntimeError,match='no progress'):
        process.get_terminal_velocity(2e-6,1000.,293.15,1.)
