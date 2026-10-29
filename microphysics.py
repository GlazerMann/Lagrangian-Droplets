#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 19:06:25 2022

@author: Laura Fierce

Many of the functions in this file were pulled from Pyrcel: https://pyrcel.readthedocs.io/en/latest/
"""


from pyrcel import constants as c
from pyrcel import thermo
from scipy.optimize import bisect
import numpy as np

rho_h2o = 1000.     # density of water, kg/m^3
sig_h2o = 71.97/1000.  # surface tenstion of water, J/m^2
Cp_h2o_vapor = 1860.    # approximate near-room-temperature vapor heat capacity, J/(kg*K)
Cp_h2o = 4184. # heat capacity of liquid water, J/(kg*K)
k_h2o_vapor = 25.95/1e3 # thermal conductivity of water vapor, W/(m*K)
Lv = 2450.9*1e3 # enthalpy of vaporization of water vapor?, J/kg
D_inf = 0.242/100**2


# molecular weight of dry air, H2O, and CO2:
M_dry_air = 28.9647/1000. # kg/mol
M_h2o = 18./1000. # kg/mol
M_co2 = 44./1000. # kg/mol





def SuperSaturationRatio(mixing_ratio, temperature, P=100000.):
    p = P/100. # Atmospheric pressure (hPa); legacy default retained
    To = temperature-273.15 #Convert to degrees C
    #Saturated vapor pressure (T)
    psat = 6.1094*np.exp(17.625*To/(243.04+To)) #(hPa)
    #Actual Vapor Pressure
    pv = (mixing_ratio)*p/(0.622+(mixing_ratio)) # (hPa)
    #Supersaturation
    SS = pv/psat-1
    return SS

def MixingRatio(saturation_ratio,T,P):
    saturation_ratio=np.asarray(saturation_ratio,dtype=float)
    T=np.asarray(T,dtype=float)
    P=np.asarray(P,dtype=float)
    if (np.any(~np.isfinite(T)) or np.any(T<=0) or np.any(~np.isfinite(P))
            or np.any(P<=0) or np.any(~np.isfinite(saturation_ratio))):
        raise ValueError('Mixing ratio requires finite temperature, pressure, and supersaturation')
    p = P/100 # Atmospheric pressure (hPa)
    #R = 8.314    # Gas constant
    To = T-273.15 #Convert to degrees C
    #Actual Vapor Pressure
    
    # Saturated vapor pressure (T)
    psat = 6.1094*np.exp(17.625*To/(243.04+To)) #(hPa)
    # Vapor pressure at saturation_ratio
    pv = psat*(saturation_ratio+1.)
    if np.any(pv<0) or np.any(pv>=p) or np.any(~np.isfinite(pv)):
        raise ValueError('Vapor pressure must satisfy 0 <= vapor pressure < total pressure')
    mixing_ratio = c.epsilon*pv/(p-pv)
    
    return mixing_ratio

def equilibrate_h2o(dry_diameters, kappas, Ns, temp, SS, P):
    
    dry_diameters=np.asarray(dry_diameters,dtype=float)
    kappas=np.asarray(kappas,dtype=float)
    Ns=np.asarray(Ns,dtype=float)
    if (dry_diameters.ndim!=1 or not len(dry_diameters)
            or kappas.shape!=dry_diameters.shape or Ns.shape!=dry_diameters.shape
            or np.any(~np.isfinite(dry_diameters)) or np.any(dry_diameters<=0)
            or np.any(~np.isfinite(kappas)) or np.any(kappas<=0)
            or np.any(~np.isfinite(Ns)) or np.any(Ns<0)):
        raise ValueError('Equilibrium requires matching finite populations, positive diameters and kappa, and nonnegative number')
    if (not all(np.isfinite(value) for value in (temp,SS,P)) or temp<=0 or P<=0):
        raise ValueError('Equilibrium requires finite positive temperature and pressure')
    vapor_pressure=(SS+1.)*thermo.es(temp-273.15)
    if not 0<=vapor_pressure<P:
        raise ValueError('Vapor pressure must satisfy 0 <= vapor pressure < total pressure')
    dry_radii = dry_diameters/2
    f = lambda r, dry_radius, kappa: thermo.Seq(r, dry_radius, temp, kappa) - SS
    diameters = []
    for dry_radius, kappa in zip(dry_radii, kappas):
        # Locate the critical value (f(r_crit) > 0), and use bisection from it and
        # r_dry (f(r_dry) < 0) to find equilibrium wet particle radius for a given S0
        r_b, SS_crit =  thermo.kohler_crit(temp, dry_radius, kappa)
        r_a = dry_radius
        radius = bisect(f, r_a, r_b, args=(dry_radius, kappa), xtol=1e-30, maxiter=500)
        diameters.append(2.*radius)
    
    diameters = np.array(diameters) # diameter of particles in m
    
    # intial parcel thermodynamic properties 
    vapor_pressure = (SS + 1.0) * thermo.es(temp - 273.15)
    mixing_ratio = c.epsilon * vapor_pressure / (P - vapor_pressure) # kg vapor/kg dry air
    
    mass_water = (   # kg liquid water per m^3
        lambda radius, dry_radius, Ni: (4.0 * np.pi / 3.0)
        * c.rho_w
        * Ni
        * (radius ** 3 - dry_radius ** 3)
        )
    liquid_water_content = np.sum(     # kg liqiud water per m^3
        [
            mass_water(radius, dry_radius, Ni)
            for radius, dry_radius, Ni in zip(np.array(diameters)/2., dry_diameters/2., Ns)
        ])

    liquid_water_content /= thermo.rho_air(temp, P, 0.0) # kg liqiud water per kg air
    return diameters, mixing_ratio, liquid_water_content
    
'''

# @nb.njit()
# @auxcc.export("oneway", "f8[:](f8[:],f8,f8,f8,f8,f8,f8[:],f8[:],f8[:],f8,f8)")
def oneway(y, t, wv, S, T, P, Ddrys, kappas, Ns, accom, rho_aero=1000.):
    """ Calculates the instantaneous time-derivative of the parcel model system.
    Given a current state vector `y` of the parcel model, computes the tendency
    of each term including thermodynamic (pressure, temperature, etc) and aerosol
    terms. The basic aerosol properties used in the model must be passed along
    with the state vector (i.e. if being used as the callback function in an ODE
    solver).

    Parameters
    ----------
    y : array_like
        Current state of the parcel model system,
            * y[0] = altitude, m
            * y[1] = Pressure, Pa
            * y[2] = temperature, K
            * y[3] = water vapor mass mixing ratio, kg/kg
            * y[4] = cloud liquid water mass mixing ratio, kg/kg
            * y[5] = cloud ice water mass mixing ratio, kg/kg
            * y[6] = parcel supersaturation
            * y[7:] = particle diameter, m
    t : float
        Current simulation time, in seconds.
    nr : Integer
        Number of aerosol radii being tracked.
    D_drys : array_like
        Array recording aerosol dry diameters, m.
    Nis : array_like
        Array recording aerosol number concentrations, 1/(m**3).
    V : float
        Updraft velocity, m/s.
    kappas : array_like
        Array recording aerosol hygroscopicities.
    accom : float, optional (default=:const:`constants.ac`)
        Condensation coefficient.
    Returns
    -------
    x : array_like
        Array of shape (``nr``+7, ) containing the evaluated parcel model
        instaneous derivative.
    Notes
    -----
    This function is implemented using numba; it does not need to be just-in-
    time compiled in order ot function correctly, but it is set up ahead of time
    so that the internal loop over each bin growth term is parallelized.
    """

    Ds = y[:int(len(y)/2)]
    Tps = y[int(len(y)/2):]

    T_c = T - 273.15  # convert temperature to Celsius
    pv_sat = es(T_c)  # saturation vapor pressure
    Tv = (1.0 + 0.61 * wv) * T

    ## Compute air densities from current state
    rho_air = P / c.Rd / Tv
    #: TODO - port to parcel.py

    # drs_dt = np.empty(shape=(nr), dtype=DTYPE)
    dDs_dt = np.empty_like(Ds)
    dTps_dt = np.empty_like(Ds)
    
    for i in nb.prange(len(Ds)):
        D = Ds[i]
        Tp = Tps[i]
        Ddry = Ddrys[i]
        kappa = kappas[i]
        r_dry = Ddry/2.
        r = D/2.
        ## Non-continuum diffusivity/thermal conductivity of air near
        ## near particle
        dv_r = dv(T, r, P, accom)
        ka_r = ka(T, r, rho_air)

        ## Condensation coefficient
        G_a = (c.rho_w * c.R * T) / (pv_sat * dv_r * c.Mw)
        G_b = (c.L * c.rho_w * ((c.L * c.Mw / (c.R * T)) - 1.0)) / (ka_r * T)
        G = 1.0 / (G_a + G_b)

        ## Difference between ambient and particle equilibrium supersaturation
        Seq_r = Seq(r, r_dry, T, kappa)
        delta_S = S - Seq_r
        
        dr_dt = (G / r) * delta_S
        
#         ## Size and liquid water tendencies
#         dr_dt = (G / r) * delta_S
#         dTp_dt = 0.

#        if D>Ddry:
        Tp = T # assume the droplet temperature is the same as the air temperature
        mp = get_mass_from_diameter(Ddry,D,rho_aero)
        
        q = np.array([mp,Tp])
        dmp_dt,dTp_dt = get_odesystem_evaporate(t,q,Ddry,kappa,rho_aero,T,S+1.,udiff_fun=lambda t: 0., p=101325.)
        dr_dt = dmp_dt / (4.0 * PI * c.rho_w * r * r)
        dDs_dt[i] = 2.*dr_dt
        dTps_dt[i] = dTp_dt
        
    dydt = np.hstack([dDs_dt,dTps_dt])
    
    return dydt
'''
def one_particle(y, t, mixing_ratio, SS, T, P, dry_diameter, kappa, N, accom, rho_aero, ignore_Tp=False):
    T_c = T - 273.15  # convert temperature to Celsius
    pv_sat = thermo.es(T_c)  # saturation vapor pressure, Pa

    ## Compute air densities from current state
    Tv = (1.0 + 0.61 * mixing_ratio) * T 
    rho_air = P / (c.Rd * Tv)
    
    Dp = y[0]
    Tp = y[1]
    dry_radius = dry_diameter/2.
    Rp = Dp/2.
    
    if ignore_Tp:
        ## Non-continuum diffusivity/thermal conductivity of air near
        ## near particle, m^2/s
        dv_r = dv(T, Rp, P, accom) # Non-continuum diffusivity of water vapor, m^2/s
        ka_r = thermo.ka(T, rho_air, Rp) # Non-continuum thermal conductivity of air near particle, J/(s*m*K)
        
        ## Condensation coefficient, m^2/s
        G_a = (c.rho_w * c.R * T) / (pv_sat * dv_r * c.Mw) # s/m^2
        G_b = (c.L * c.rho_w * ((c.L * c.Mw / (c.R * T)) - 1.0)) / (ka_r * T) # s/m^2
        G = 1.0 / (G_a + G_b) # m^2/s
    
        ## Difference between ambient and particle equilibrium supersaturation
        Seq_r = thermo.Seq(Rp, dry_radius, T, kappa)
        delta_SS = SS - Seq_r
        dr_dt = (G / Rp) * delta_SS # m/s
        
        dTp_dt = 0. # K/s
    else:
        q = np.array([mass_from_diameter(Dp, dry_diameter, rho_aero), Tp])
        dmp_dt, dTp_dt = get_odesystem_droplet(t, q, dry_diameter, kappa, rho_aero, T, SS+1., accom, udiff_fun=lambda t: 0., p=P)
        dr_dt = dmp_dt / (4.0 * np.pi * c.rho_w * Rp * Rp) # m/s 
        
    dDp_dt = 2.*dr_dt # m/s
    dydt = np.hstack([dDp_dt, dTp_dt])
    return dydt
    
def one_particle_lny(lny, t, mixing_ratio, SS, T, P, dry_diameter, kappa, N, accom, rho_aero, ignore_Tp=False):
    # The inactive temperature coordinate may receive arbitrarily large
    # finite-difference probes from an implicit solver; it must stay inactive.
    with np.errstate(over='raise', invalid='raise'):
        state = np.array([np.exp(lny[0]), T]) if ignore_Tp else np.exp(lny)
    derivative = one_particle(state, t, mixing_ratio, SS, T, P, dry_diameter,
                              kappa, N, accom, rho_aero, ignore_Tp=ignore_Tp)
    return derivative / state

def one_particle_lnDp(lny, t, mixing_ratio, SS, T, P, dry_diameter, kappa, N, accom, rho_aero, ignore_Tp=False):
    state = np.array(lny, dtype=float, copy=True)
    with np.errstate(over='raise', invalid='raise'):
        state[0] = np.exp(state[0])
    derivative = one_particle(state, t, mixing_ratio, SS, T, P, dry_diameter,
                              kappa, N, accom, rho_aero, ignore_Tp=ignore_Tp)
    derivative[0] /= state[0]
    return derivative
    
## Auxiliary, single-value calculations with GIL released for derivative
## calculations
def sigma_w(T):
    """See :func:`pyrcel.thermo.sigma_w` for full documentation
    """
    return 0.0761 - (1.55e-4) * (T - 273.15)

def ka(T, r, rho):
    """See :func:`pyrcel.thermo.ka` for full documentation
    """
    ka_cont = 1e-3 * (4.39 + 0.071 * T)
    denom = 1.0 + (ka_cont / (c.at * r * rho * c.Cp)) * np.sqrt(
        (2 * np.pi * c.Ma) / (c.R * T)
    )
    return ka_cont / denom

def dv(T, r, P, accom):
    """See :func:`pyrcel.thermo.dv` for full documentation
    """
    P_atm = P / 101325.0  # Pa -> atm
    dv_cont = 1e-4 * (0.211 / P_atm) * ((T / 273.0) ** 1.94)
    denom = 1.0 + (dv_cont / (accom * r)) * np.sqrt(
        (2 * np.pi * c.Mw) / (c.R * T)
    )
    return dv_cont / denom

def es(T):
    """See :func:`pyrcel.thermo.es` for full documentation
    """
    return 611.2 * np.exp(17.67 * T / (T + 243.5))

def Seq(r, r_dry, T, kappa):
    """See :func:`pyrcel.thermo.Seq` for full documentation.
    """
    A = (2.0 * c.Mw * sigma_w(T)) / (c.R * T * c.rho_w * r)
    B = 1.0
    if kappa > 0.0:
        B = (r ** 3 - (r_dry ** 3)) / (r ** 3 - (r_dry ** 3) * (1.0 - kappa))
    return np.exp(A) * B - 1.0

def get_odesystem_droplet(t, q, Dd, tkappa, rho_aero, Tv, Sv, accom, udiff_fun=lambda t: 0., p=101325.):
    mp = q[0]
    Tp = q[1]
    Dp = ((mp*6/np.pi - Dd**3*(rho_aero - rho_h2o))/rho_h2o)**(1./3.)
    
    lamb = 1.6 # constant between 1.6 and 2
    # Stable continuous limit as Tp approaches Tv. Subtracting powers of
    # nearly identical temperatures otherwise changes equivalent log states.
    delta = (Tp - Tv) / Tv
    exponent = 2. - lamb
    C_T = 1. if delta == 0. else exponent*delta / np.expm1(exponent*np.log1p(delta))
    
    p_d = droplet_vapor_pressure(Dp, Dd, tkappa, Tp) # droplet vapor pressure, Pa
    p_v = vapor_pressure(Sv, Tv) # vapor pressure, Pa
    udiff = udiff_fun(t) # not sure, might be in m/s, always zero
    Re = particle_Re(udiff, Dp, Tv, Sv, p=p) # Reynolds number, unitless
    Sc = particle_Sc(Dp, Tv, Sv, p=p) # Schmidt number, unitless
    Pr = particle_Pr(Tv, Sv, accom, Dp, p=p) # Prandtl number, unitless
    Sh = 1+0.38*Re**(1/2)*Sc**(1/3) # Sherwood number, unitless, chen et al.
    dmdt = 2*np.pi*p*Dp*M_h2o*D_inf*C_T*Sh/(c.R*Tv)*np.log((p-p_d)/(p-p_v)) # 
    
    #print(Sv, p_d, p_v, udiff, Re, Sc, Pr, Sh, dmdt)
    
    if Dp < Dd and dmdt < 0:
        dmdt = 0. 
    if Dp < Dd:
        Dp = Dd
    
    Nu = 1 + 0.3*Re**(1/2)*Pr**(1/3) # Chen et al. 
    #Kg = air_thermal_conductivity(Sv, Tv, accom, Dp, x_co2=410e-6, p=101325.)
    dTdt = (np.pi*Dp**2*k_h2o_vapor*(Tv-Tp)/(Dp/2)*Nu+Lv*(dmdt))/(mp*Cp_h2o)
    return np.array([dmdt, dTdt])

def droplet_vapor_pressure(D, Dd, tkappa, Td):
    p_sat = saturation_vapor_pressure(Td) # Pa
    S_d = droplet_saturation_ratio(D, Dd, tkappa, Td)
    p_d = S_d*p_sat
    return p_d # Pa

def saturation_vapor_pressure(Tv_K):
    Tv = Tv_K-273.15
    p_sat = 611.21*np.exp((18.678 - Tv/234.5)*(Tv/(257.14+Tv)))
    return p_sat # Pa

def droplet_saturation_ratio(D, Dd, tkappa,Td):
    S_d = ((D**3 - Dd**3)/(D**3 - Dd**3*(1-tkappa)))*np.exp(4*sig_h2o*M_h2o/(c.R*Td*rho_h2o*D))
    return S_d

def vapor_pressure(S, Tv):
    p_sat = saturation_vapor_pressure(Tv)
    p_v = S*p_sat
    return p_v # Pa

def particle_Re(u_diff, Dp, Tv, S, x_co2=410e-6, p=101325.):
    rho_air = air_density(Tv, S, x_co2=x_co2, p=p) # kg/m^3
    mu = air_dynamic_viscosity(Tv) # kg/(m*s)
    nu = mu/rho_air # m^2/s
    Re = abs(u_diff)*Dp/nu # udiff needs to be in m/s for this to be unitless
    return Re


def air_density(Tv, S, x_co2=410e-6,p=101325.):
    x_h2o = h2o_mixing_ratio(S, Tv, p=p) # mol h2o per mol air
    rho_air = (M_dry_air*(1-x_h2o-x_co2) + x_h2o*M_h2o + x_co2*M_co2)*p/(c.R*Tv) # kg/m^3
    return rho_air
    
def h2o_mixing_ratio(S, Tv, p=101325.):
    p_v = vapor_pressure(S,Tv) # Pa
    mixing_ratio = p_v/p # mol h2o per mol air
    return mixing_ratio

def air_dynamic_viscosity(Tv):
    # http://www-mdp.eng.cam.ac.uk/web/library/enginfo/aerothermal_dvd_only/aero/fprops/propsoffluids/node5.html
    mu0 = 17.15e-6 #should be 17.15E-4?
    T0 = 273.15
    mu = mu0*(Tv/T0)**0.7 # kg/(m*s)
    return mu

def particle_Sc(Dp, Tv, S, x_co2=410e-6, p=101325.):
    rho_air = air_density(Tv, S, x_co2=x_co2, p=p) # kg/m^3
    mu = air_dynamic_viscosity(Tv) # kg/(m*s)
    D_AB = water_vapor_diffusion_coefficient(Tv) # m^2/s
    Sc = mu/(rho_air*D_AB) # unitless
    return Sc

def water_vapor_diffusion_coefficient(Tv):
    # https://www.researchgate.net/post/Binary_diffusion_coefficients_for_water_vapour_in_air_at_normal_pressure
    diffusion_coeff = 0.211/100**2 * (Tv/273.15)**1.94 # Pruppacher and Klett
    return diffusion_coeff # m^2/s

def particle_Pr(Tv, S, accom, Dp, p=101325.):
    Cp_v = air_heat_capacity(S, Tv, p=p) # J/(kg*K)
    mu = air_dynamic_viscosity(Tv)  # kg/(m*s)
    #Kg = air_thermal_conductivity(S, Tv, accom, Dp, x_co2=410e-6, p=101325.) # J/(m*s*K)
    Pr = Cp_v*mu/k_h2o_vapor # unitless
    return Pr

def air_heat_capacity(S, Tv, p=101325.):
    Cp_dry_air = 1030.5 - 0.11975*Tv + 3.9734e-4*Tv**2 # J/(kg*K), http://www.mhtlab.uwaterloo.ca/pdf_reports/mhtl_G01.pdf
    H = specific_humidity(S, Tv, p=p)
    # H is a mass fraction of moist air, so mix both heat capacities on
    # the same mass basis and in J/(kg*K). The legacy dry-air fit is retained.
    Cp = (1.-H)*Cp_dry_air + H*Cp_h2o_vapor
    return Cp

def specific_humidity(S, Tv, p=101325.):
    p_sat = saturation_vapor_pressure(Tv)
    p_h2o = p_sat*S
    specific_humidity = p_h2o*M_h2o/(p_h2o*M_h2o + (p-p_h2o)*M_dry_air) # kg water vapor per kg air 
    return specific_humidity

#def air_thermal_conductivity(S, Tv, accom, Dp, x_co2=410e-6, p=101325.):
#    ka = 1E-3*(4.39+0.0071*Tv) # J/(m*s*K)
#    rho_air = air_density(Tv, S, x_co2=410e-6, p=101325.) # kg/m^3
#    Cp = air_heat_capacity(S, Tv) # J/(kg*K)
#    x_h2o = h2o_mixing_ratio(S, Tv, p=p) # mol h2o per mol air
#    M_air = M_dry_air*(1-x_co2-x_h2o)+M_h2o*x_h2o+M_co2*x_co2 # kg/mol
#    denom = 1+(((2*ka)/(accom*Dp*rho_air*Cp))*np.sqrt((2*np.pi*M_air)/(c.r*Tv))) # unitless
#    Kg = ka/denom # J/(m*s*K)
#    return Kg


def mass_from_diameter(diameter, dry_diameter, aerosol_density):
    """Total solute-plus-water mass in kg for the thermal ODE state."""
    return np.pi/6. * (rho_h2o*diameter**3 +
                       (aerosol_density-rho_h2o)*dry_diameter**3)
