#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 01:18:29 2023

@author: fier887
"""

import numpy as np
import os
from netCDF4 import Dataset
try:
    from . import constants as c
except ImportError: # execution as a standalone script
    import constants as c
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np

import os
import matplotlib.pyplot as plt

from scipy.optimize import fminbound
from scipy.optimize import bisect

from scipy.integrate import solve_ivp

from scipy.stats import norm
from scipy.stats import skewnorm
from scipy.stats import skewcauchy
from scipy.stats import gamma
from scipy.stats import gumbel_r

from scipy.io import loadmat

from netCDF4 import Dataset


def dsd_figs(trajectories_dir,time,
             plot_trajectories=True,
             Dp_edges_logscl=np.logspace(-7,-4,101),
             Dp_edges=np.linspace(0.,40e-6,101),
             dns_lab='DNS',approx_kohler=False,
             offline_lab='offline',
             fig_overdir='figs/',
             dpi = 1000):


    lnDp_edges = np.log(Dp_edges_logscl)

    idx = trajectories_dir[:-1].rfind('/')
    case_str = trajectories_dir[(idx+1):-1]
    if not os.path.exists(fig_overdir):
        os.mkdir(fig_overdir)

    fig_dir = fig_overdir + case_str + '/'
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)


    dsd_fig_dir = fig_dir + 'Dp_vs_t/'
    if not os.path.exists(dsd_fig_dir):
        os.mkdir(dsd_fig_dir)

    logdsd_fig_dir = fig_dir + 'lnDp_vs_t/'
    if not os.path.exists(logdsd_fig_dir):
        os.mkdir(logdsd_fig_dir)

    dlnDp = lnDp_edges[1] - lnDp_edges[0]
    dDp = Dp_edges[1] - Dp_edges[0]
    # lnDp_edges = np.log(Dp_edges)
    dNdlnD_ref = np.zeros(len(lnDp_edges)-1)
    dNdlnD_predict = np.zeros(len(lnDp_edges)-1)
    dNdD_ref = np.zeros(len(Dp_edges)-1)
    dNdD_predict = np.zeros(len(Dp_edges)-1)
    D_crit_all = []
    filenames = sorted(name for name in os.listdir(trajectories_dir)
                       if name.startswith('traces_') and name.endswith('.nc'))
    for ff,filename in enumerate(filenames):
        if filename.startswith('trace'):
            ncfilename = trajectories_dir + filename
            ncfile = Dataset(ncfilename,mode='r',format='NETCDF4_CLASSIC')

            matches = np.flatnonzero(ncfile['s_offset'][:] == 0.)
            if len(matches) != 1:
                ncfile.close()
                raise ValueError('Each verification trace must contain one zero supersaturation offset')
            idx_ss = int(matches[0])
            if not any(ncfile['s_env'][:]>10.):#np.mean(ncfile['s_env'][:])<1.:
                if plot_trajectories:
                    hln_ref = plt.plot(ncfile['time'][:],1e6*ncfile['Dp'][:],label=dns_lab,linewidth=2.);
                    hln_predict = plt.plot(ncfile['time'][:],1e6*ncfile['Dp_predict'][:,idx_ss],label=offline_lab,linewidth=2.);
                    plt.xlabel('time [s]')
                    plt.ylabel('diameter [$\mu$m]')
                    plt.legend()
                    plt.title(filename)
                    plt.savefig(dsd_fig_dir + str(ff).zfill(6) + '.pdf',dpi=dpi)
                    plt.yscale('log')
                    plt.xlim([min(ncfile['time'][:]),max(ncfile['time'][:])])
                    plt.savefig(logdsd_fig_dir + str(ff).zfill(6) + '.pdf',dpi=dpi)

                    plt.close()
                if (max(ncfile['time'][:])>time) and (min(ncfile['time'][:])<time):
                    Dp_ref = np.interp(time,ncfile['time'][:],ncfile['Dp'][:])
                    Dp_predict = np.interp(time,ncfile['time'][:],ncfile['Dp_predict'][:,idx_ss])
                    one_hist,edges = np.histogram(np.log(Dp_ref),bins=lnDp_edges)
                    dNdlnD_ref += one_hist/dlnDp
                    # hist_ref += one_hist
                    one_hist,edges = np.histogram(np.log(Dp_predict),bins=lnDp_edges)
                    # hist_predict += one_hist
                    dNdlnD_predict += one_hist/dlnDp

                    one_hist,edges = np.histogram(Dp_ref,bins=Dp_edges)
                    dNdD_ref += one_hist/(1e6*dDp)
                    # hist_ref += one_hist
                    one_hist,edges = np.histogram(Dp_predict,bins=Dp_edges)
                    # hist_predict += one_hist
                    dNdD_predict += one_hist/(1e6*dDp)

                    T = np.interp(time,ncfile['time'][:],ncfile['T_env'][:])
                    r_dry = np.interp(time,ncfile['time'][:],ncfile['Ddry'][:])/2.
                    kappa = np.interp(time,ncfile['time'][:],ncfile['kappa'][:])
                    r_crit, s_crit = kohler_crit(T, r_dry, kappa, approx=approx_kohler)
                    D_crit = 2.*r_crit
                    D_crit_all.append(D_crit)

                lnDp_mids = (lnDp_edges[1]-lnDp_edges[0])/2. + lnDp_edges[:-1]
                Dp_mids = (Dp_edges[1]-Dp_edges[0])/2. + Dp_edges[:-1]
            ncfile.close()
        if np.remainder(ff,100) == 0.:
            print(ff + 1, '/', len(filenames))
    hln2_ref = plt.plot(1e6*Dp_mids,dNdD_ref,label=dns_lab);
    hln2_predict = plt.plot(1e6*Dp_mids,dNdD_predict,label=offline_lab);
    plt.xlabel('diameter [$\mu$m]')
    plt.ylabel('number density [m$^{-3}\mu$m$^{-1}$]')
    plt.legend()
    plt.xlim([1e6*min(Dp_mids),1e6*max(Dp_mids)])
    plt.savefig(fig_dir + 'dsd.pdf',dpi=dpi)
    plt.show()

    hln2_ref = plt.plot(1e6*np.exp(lnDp_mids),dNdlnD_ref,label=dns_lab);
    hln2_predict = plt.plot(1e6*np.exp(lnDp_mids),dNdlnD_predict,label=offline_lab);
    plt.xlabel('diameter [$\mu$m]')
    plt.ylabel('number density [m$^{-3}$]')
    plt.legend()
    plt.xlim([1e6*min(np.exp(lnDp_mids)),1e6*max(np.exp(lnDp_mids))])
    plt.xscale('log');
    plt.savefig(fig_dir + 'dsd_log.pdf',dpi=dpi)

    return D_crit_all

def store_DNS_trajectories(dns_filename,trajectories_overdir,P0=101325.,add_verification=True):

    if not os.path.exists(trajectories_overdir):
        os.mkdir(trajectories_overdir)

    one_case = int(dns_filename[dns_filename.find('.mat')-1])
    trajectories_dir = trajectories_overdir + 'case' + str(one_case) + '/'

    if not os.path.exists(trajectories_dir):
        os.mkdir(trajectories_dir)
    filenames = os.listdir(trajectories_dir)
    for filename in filenames:
        os.remove(trajectories_dir + filename)

    try:
        import mat73 # optional, required only for HDF5 MATLAB inputs
        data_dict = mat73.loadmat(dns_filename)
    except:
        data_dict = loadmat(dns_filename)

    for aa in range(len(data_dict['trajectories'])):
        ncfilename = trajectories_dir + 'traces_' + str(aa).zfill(6)+'.nc'
        try: ncfile.close()  # just to be safe, make sure dataset is not already open.
        except: pass
        data_one_trajectory = data_dict['trajectories'][aa][0]
        if len(data_one_trajectory.shape) == 1:
            data_one_trajectory = data_one_trajectory.reshape(1,-1)

        Ntimes = len(data_one_trajectory[:,1])

        ncfile = Dataset(ncfilename,mode='w',format='NETCDF4_CLASSIC')

        ncfile.createDimension('time', Ntimes)

        time = ncfile.createVariable('time', np.float64, ('time',))
        time.units = 'seconds'
        time.longname = 'simulation time'
        time[:] = data_one_trajectory[:,1]

        x = ncfile.createVariable('x', np.float64, ('time',))
        x.units = 'm'
        x.longname = 'x'
        x[:] = data_one_trajectory[:,2]

        y = ncfile.createVariable('y', np.float64, ('time',))
        y.units = 'm'
        y.longname = 'y'
        y[:] = data_one_trajectory[:,3]

        z = ncfile.createVariable('z', np.float64, ('time',))
        z.units = 'm'
        z.longname = 'z'
        z[:] = data_one_trajectory[:,4]

        q = ncfile.createVariable('q', np.float64, ('time',))
        q.units = '(kg water vapor)/(kg dry air)'
        q.longname = 'water vapor mixing ratio of the fluid'
        q[:] = data_one_trajectory[:,11]

        q_star = ncfile.createVariable('q_star', np.float64, ('time',))
        q_star.units = '(kg water vapor)/(kg dry air)'
        q_star.longname = 'equilibrium water vapor mixing ratio at the droplet surface'
        q_star[:] = data_one_trajectory[:,12]

        T_env = ncfile.createVariable('T_env', np.float64, ('time',))
        T_env.units = 'K'
        T_env.longname = 'fluid temperature'
        T_env[:] = data_one_trajectory[:,10]

        s_env = ncfile.createVariable('s_env', np.float64, ('time',))
        s_env.units = '%'
        s_env.longname = 'water vapor supersaturation'
        s_env[:] = data_one_trajectory[:,13]/100.-1.

        P_env = ncfile.createVariable('P_env', np.float64, ('time',))
        P_env.units = 'Pa'
        P_env.longname = 'pressure'
        P_env[:] = P0*np.ones(len(data_one_trajectory[:,1]))


        if add_verification:
            Dp = ncfile.createVariable('Dp', np.float64, ('time',))
            Dp.units = 'm'
            Dp.longname = 'particle wet diameter'
            Dp[:] = 2.*data_one_trajectory[:,8]

            Tp = ncfile.createVariable('Tp', np.float64, ('time',))
            Tp.units = 'K'
            Tp.longname = 'particle tempearture'
            Tp[:] = data_one_trajectory[:,9]
        ncfile.close()


def drive_many_trajectories(trajectories_dir,part_Ddry,part_kappa,s_offsets=np.array([0.]),read_verification=False,max_step=0.1,ignore_Tp=True,ignore_feedback=True,method='LSODA'):

    s_stats = np.loadtxt(trajectories_dir + 's_stats.txt')
    s_mean_ref = s_stats[0]

    filenames = os.listdir(trajectories_dir)
    for filename in filenames:
        if filename.startswith('traces_'):
            ncfilename = trajectories_dir + filename
            ts,s_envs,q_envs,T_envs,P_envs,Dps,Tps = read_one_trajectory(ncfilename,read_verification=True,read_q=True)
            Dps_predict = np.zeros([len(ts),len(s_offsets)])
            for ss,ds_mean in enumerate(s_offsets):
                Dps_predict[:,ss] = drive_one_particle(ts,s_envs+ds_mean,T_envs,P_envs,Ddry=part_Ddry,kappa=part_kappa,max_step=max_step,ignore_Tp=ignore_Tp,method=method)

            try: ncfile.close()  # just to be safe, make sure dataset is not already open.
            except: pass

            try:
                ncfile = Dataset(ncfilename,mode='a',format='NETCDF4_CLASSIC')
                if 's_mean' not in ncfile.dimensions:
                    ncfile.createDimension('s_mean', len(s_offsets))
                if 's_mean' not in ncfile.variables:
                    s_mean = ncfile.createVariable('s_mean', np.float64, ('s_mean',))
                else:
                    s_mean = ncfile['s_mean']
                s_mean.units = '%'
                s_mean.longname = 'mean environmental supersaturation'
                s_mean[:] = s_offsets + s_mean_ref

                if 's_offset' not in ncfile.variables:
                    s_offset = ncfile.createVariable('s_offset', np.float64, ('s_mean',))
                else:
                    s_offset = ncfile['s_offset']
                s_offset.units = '%'
                s_offset.longname = 'imposed difference between mean environmental supersaturation in DNS and this trajectory'
                s_offset[:] = s_offsets

                if 'Ddry' not in ncfile.variables:
                    Ddry = ncfile.createVariable('Ddry', np.float64, ('time',))
                else:
                    Ddry = ncfile['Ddry']
                Ddry.units = 'm'
                Ddry.longname = 'dry diameter'
                Ddry[:] = part_Ddry*np.ones(len(ncfile['time'][:]))

                if 'kappa' not in ncfile.variables:
                    kappa = ncfile.createVariable('kappa', np.float64, ('time',))
                else:
                    kappa = ncfile['kappa']
                kappa.units = 'unitless'
                kappa.longname = 'hygroscopicity parameter'
                kappa[:] = part_kappa*np.ones(len(ncfile['time'][:]))

                if 'Dp_predict' not in ncfile.variables:
                    Dp_predict = ncfile.createVariable('Dp_predict', np.float64, ('time','s_mean'))
                else:
                    Dp_predict = ncfile['Dp_predict']
                Dp_predict.units = 'm'
                Dp_predict.longname = 'wet diameter predicted from one-way coupling'
                Dp_predict[:] = Dps_predict

                ncfile.close()
            except:
                print('ncfilename = ' + ncfilename + ' is not opening')

def kohler_crit(T, r_dry, kappa, approx=False):
    """ Critical radius and supersaturation of an aerosol particle.
    The critical size of an aerosol particle corresponds to the maximum equilibrium
    supersaturation achieved on its Kohler curve. If a particle grows beyond this
    size, then it is said to "activate", and will continue to freely grow even
    if the environmental supersaturation decreases.
    This function computes the critical size and and corresponding supersaturation
    for a given aerosol particle. Typically, it will analyze :func:`Seq` for the
    given particle and numerically compute its inflection point. However, if the
    ``approx`` flag is passed, then it will compute the analytical critical point
    for the approximated kappa-Kohler equation.
    Parameters
    ----------
    T : float
        ambient air temperature, K
    r_dry : float
        dry particle radius, m
    kappa : float
        particle hygroscopicity parameter
    approx : boolean, optional (default=False)
        use the approximate kappa-kohler equation
    Returns
    -------
    (r_crit, s_crit) : tuple of floats
        Tuple of :math:`(r_\\text{crit},\, S_\\text{crit})`, the critical radius (m)
        and supersaturation of the aerosol droplet.
    See Also
    --------
    Seq : equilibrium supersaturation calculation
    """
    neg_seq = lambda r: -1.0 * seq(r, r_dry, T, kappa)
    out = fminbound(
        neg_seq, r_dry, r_dry * 1e4, xtol=1e-10, full_output=True, disp=0
    )
    r_crit, s_crit = out[:2]
    s_crit *= -1.0  # multiply by -1 to undo negative flag for Seq

    return r_crit, s_crit

def sigma_w(T):
    """ Surface tension of water for a given temperature.
    .. math::
        \\begin{equation}
        \sigma_w = 0.0761 - 1.55\\times 10^{-4}(T - 273.15)
        \end{equation}
    Parameters
    ----------
    T : float
        ambient air temperature, degrees K
    Returns
    -------
    float
        :math:`\sigma_w(T)` in J/m^2
    """
    return 0.0761 - 1.55e-4 * (T - 273.15)

def seq(r, r_dry, T, kappa):
    """ κ-Kohler theory equilibrium saturation over aerosol.
    Calculates the equilibrium supersaturation (relative to 100% RH) over an
    aerosol particle of given dry/wet radius and of specified hygroscopicity
    bathed in gas at a particular temperature
    Following the technique of [PK2007], classical
    Kohler theory can be modified to account for the hygroscopicity of an aerosol
    particle using a single parameter, :math:`\kappa`. The modified theory predicts
    that the supersaturation with respect to a given aerosol particle is,
    .. math::
        S_\\text{eq} &= a_w \exp \\left( \\frac{2\sigma_{w} M_w}{RT\\rho_w r} \\right)\\\\
        a_w &= \\left(1 + \kappa\\left(\\frac{r_d}{r}^3\\right) \\right)^{-1}
    with the relevant thermodynamic properties of water defined elsewhere in this
    module, :math:`r_d` is the particle dry radius (``r_dry``), :math:`r` is the
    radius of the droplet containing the particle (``r``), :math:`T` is the temperature
    of the environment (``T``), and :math:`\kappa` is the hygroscopicity parameter
    of the particle (``kappa``).
    Parameters
    ----------
    r : float
        droplet radius, m
    r_dry : float
        dry particle radius, m
    T : float
        ambient air temperature, K
    kappa: float
        particle hygroscopicity parameter
    Returns
    -------
    float
        :math:`S_\\text{eq}` for the given aerosol/droplet system
    References
    ----------
    .. [PK2007] Petters, M. D., and S. M. Kreidenweis. "A Single Parameter
        Representation of Hygroscopic Growth and Cloud Condensation Nucleus
        Activity." Atmospheric Chemistry and Physics 7.8 (2007): 1961-1971
    See Also
    --------
    Seq_approx : compute equilibrium supersaturation using an approximation
    kohler_crit : compute critical radius and equilibrium supersaturation
    """
    A = (2.0 * c.Mw * sigma_w(T)) / (c.R * T * c.rho_w * r)
    B = (r ** 3 - (r_dry ** 3)) / (r ** 3 - (r_dry ** 3) * (1.0 - kappa))
    s = np.exp(A) * B - 1.0
    return s

def store_s_stats(trajectories_dir):
    """Write final sample statistics for every eligible trace file.

    Retains the original eligibility thresholds and equal-sample weighting.
    """
    filenames = sorted(name for name in os.listdir(trajectories_dir)
                       if name.startswith('traces_') and name.endswith('.nc'))
    samples = []
    for filename in filenames:
        with Dataset(os.path.join(trajectories_dir,filename),mode='r') as dataset:
            values = np.asarray(dataset['s_env'][:],dtype=float)
        if np.mean(values)<5. and np.all(values<20.) and len(values)>10:
            samples.append(values)
    if not samples:
        raise ValueError('No eligible verification supersaturation samples')
    values = np.concatenate(samples)
    if np.any(~np.isfinite(values)) or np.any(values<=-1.):
        raise ValueError('Verification supersaturation must be finite and greater than -1')
    np.savetxt(os.path.join(trajectories_dir,'s_stats.txt'),[np.mean(values),np.std(values)])
    logarithms = np.log10(1.+values)
    np.savetxt(os.path.join(trajectories_dir,'log10s_stats.txt'),[np.mean(logarithms),np.std(logarithms)])

def read_one_trajectory(trajectory_filename,read_verification=False,read_q=False):
    try: ncfile.close()  # just to be safe, make sure dataset is not already open.
    except: pass
    ncfile = Dataset(trajectory_filename,mode='r',format='NETCDF4_CLASSIC')
    ts = ncfile['time'][:]
    q_envs = ncfile['q'][:]
    s_envs = ncfile['s_env'][:]
    T_envs = ncfile['T_env'][:]
    P_envs = ncfile['P_env'][:]

    if read_verification:
        Dps = ncfile['Dp'][:]
        Tps = ncfile['Tp'][:]

    ncfile.close()
    if read_q:
        if read_verification:
            return ts,s_envs,q_envs,T_envs,P_envs,Dps,Tps
        else:
            return ts,s_envs,q_envs,T_envs,P_envs
    else:
        if read_verification:
            return ts,s_envs,T_envs,P_envs,Dps,Tps
        else:
            return ts,s_envs,T_envs,P_envs

def drive_one_particle(ts,s_envs,T_envs,P_envs,
                       Ddry=110e-9,kappa=1.,max_step=0.1,rho_dry=2160.,ignore_Tp=True,method='LSODA'):
    r_dry = Ddry/2.
    r0 = equilibrate_Dp(Ddry, kappa, T_envs[0], s_envs[0])/2.
    rs = np.zeros(len(ts))

    rs[0] = r0
    for tt in range(len(ts)-1):
        t0 = ts[tt]
        r0 = rs[tt]
        t = ts[tt+1]
        dlnrdt = lambda t, lnr: get_dlnrdt(lnr,r_dry,kappa,s_envs[tt],T_envs[tt],P_envs[tt])
        output = solve_ivp(
            dlnrdt,[t0,t],np.log(np.array([r0])),
            t_eval=np.array([t]),method=method,max_step=max_step)

        if not output.success or output.y.shape != (1,1) or not np.all(np.isfinite(output.y)):
            raise RuntimeError('Verification particle integration failed: %s' % output.message)
        rs[tt+1] = float(np.exp(output.y[0,0]))
    return 2.*rs

def equilibrate_Dp(Ddry, kappa, T_env, s_env):

    f = lambda r, r_dry, kap: seq(r, r_dry, T_env, kappa) - s_env
    r_crit,s_crit = kohler_crit(T_env, Ddry/2., kappa)
    if s_env>0.:
        Dp = 2.*r_crit
    else:
        r_a = Ddry/2.
        r_b = r_crit
        Dp = 2.*bisect(f, r_a, r_b, args=(Ddry/2., kappa), xtol=1e-30, maxiter=500)
    return Dp


def get_dlnrdt(lnr,r_dry,kappa,s_env,T_env,P_env):
    r = np.exp(lnr)
    pv_sat = es(T_env - 273.15)
    seq_i = seq(r, r_dry, T_env, kappa)

    rho_air = P_env / (c.Rd * T_env)  # current air density accounting for humidity
    # rho_air_dry = (P - e) / c.Rd / T  # dry air density

    dv_r = dv(T_env, r, P_env, c.ac)
    ka_r = ka(T_env, rho_air, r)

    G_a = (c.rho_w * c.R * T_env) / (pv_sat * dv_r * c.Mw)
    G_b = (c.L * c.rho_w * ((c.L * c.Mw / (c.R * T_env)) - 1.0)) / (ka_r * T_env)
    G = 1.0 / (G_a + G_b)

    dlnrdt = (G / r**2.) * (s_env - seq_i)
    return dlnrdt

def es(T_c):
    """ Calculates the saturation vapor pressure over water for a given temperature.
    Uses an empirical fit [Bolton1980], which is accurate to :math:`0.1\%` over the
    temperature range :math:`-30^oC \leq T \leq 35^oC`,
    .. math::
        \\begin{equation}
        e_s(T) = 611.2 \exp\left(\\frac{17.67T}{T + 243.5}\\right) \\tag{RY1989, 2.17}
        \end{equation}
    where :math:`e_s` is in Pa and :math:`T` is in degrees C.
    Parameters
    ----------
    T_c : float
        ambient air temperature, degrees C
    Returns
    -------
    float
        :math:`e_s(T)` in Pa
    References
    ----------
    .. [Bolton1980] Bolton, David. "The Computation of Equivalent Potential
       Temperature". Monthly Weather Review 108.8 (1980): 1046-1053
    .. [RY1989] Rogers, R. R., and M. K. Yau. A Short Course in Cloud Physics.
       Burlington, MA: Butterworth Heinemann, 1989.
    """
    return 611.2 * np.exp(17.67 * T_c / (T_c + 243.5))

def dv_cont(T, P):
    """ Diffusivity of water vapor in air, neglecting non-continuum effects.
    See :func:`dv` for details.
    Parameters
    ----------
    T : float
        ambient temperature of air surrounding droplets, K
    P : float
        ambient pressure of surrounding air, Pa
    Returns
    -------
    float
        :math:`D_v(T, P)` in m^2/s
    See Also
    --------
    dv : includes correction for non-continuum effects
    """
    P_atm = P / 101325.0  # Pa -> atm
    return 1e-4 * (0.211 / P_atm) * ((T / 273.0) ** 1.94)


def dv(T, r, P, accom=c.ac):
    """ Diffusivity of water vapor in air, modified for non-continuum effects.
    The diffusivity of water vapor in air as a function of temperature and pressure
    is given by
    .. math::
        \\begin{equation}
        D_v = 10^{-4}\\frac{0.211}{P}\left(\\frac{T}{273}\\right)^{1.94}
              \\tag{SP2006, 17.61}
        \end{equation}
    where :math:`P` is in atm [SP2006]. Aerosols much smaller than the mean free path
    of the  air surrounding them (:math:`K_n >> 1`) perturb the flow around them
    moreso than larger particles, which affects this value. We account for corrections
    to :math:`D_v` in the non-continuum regime via the parameterization
    .. math::
        \\begin{equation}
        D'_v = \\frac{D_v}{1+ \\frac{D_v}{\\alpha_c r}
                \left(\\frac{2\pi M_w}{RT}\\right)^{1/2}} \\tag{SP2006, 17.62}
        \end{equation}
    where :math:`\\alpha_c` is the condensation coefficient (:const:`constants.ac`).
    Parameters
    ----------
    T : float
        ambient temperature of air surrounding droplets, K
    r : float
        radius of aerosol/droplet, m
    P : float
        ambient pressure of surrounding air, Pa
    accom : float, optional (default=:const:`constants.ac`)
        condensation coefficient
    Returns
    -------
    float
        :math:`D'_v(T, r, P)` in m^2/s
    References
    ----------
    .. [SP2006] Seinfeld, John H, and Spyros N Pandis. Atmospheric Chemistry
       and Physics: From Air Pollution to Climate Change. Vol. 2nd. Wiley, 2006.
    See Also
    --------
    dv_cont : neglecting correction for non-continuum effects
    """
    dv_t = dv_cont(T, P)
    denom = 1.0 + (dv_t / (accom * r)) * np.sqrt((2.0 * np.pi * c.Mw) / (c.R * T))
    return dv_t / denom

def ka_cont(T):
    """ Thermal conductivity of air, neglecting non-continuum effects.
    See :func:`ka` for details.
    Parameters
    ----------
    T :
        ambient air temperature surrounding droplet, K
    Returns
    -------
    float
        :math:`k_a(T)` in J/m/s/K
    See Also
    --------
    ka : includes correction for non-continuum effects.
    """
    return 1e-3 * (4.39 + 0.071 * T)


def ka(T, rho, r):
    """ Thermal conductivity of air, modified for non-continuum effects.
    The thermal conductivity of air is given by
    .. math::
        \\begin{equation}
        k_a = 10^{-3}(4.39 + 0.071T) \\tag{SP2006, 17.71}
        \end{equation}
    Modification to account for non-continuum effects (small aerosol/droplet
    size) yields the equation
    .. math::
        \\begin{equation}
        k'_a = \\frac{k_a}{1 + \\frac{k_a}{\\alpha_t r_p \\rho C_p}
               \\frac{2\pi M_a}{RT}^{1/2}} \\tag{SP2006, 17.72}
        \end{equation}
    where :math:`\\alpha_t` is a thermal accommodation coefficient
    (:const:`constants.at`).
    Parameters
    ----------
    T : float
        ambient air temperature, K
    rho : float
        ambient air density, kg/m^3
    r : float
        droplet radius, m
    Returns
    -------
    float
        :math:`k'_a(T, \\rho, r)` in J/m/s/K
    References
    ----------
    .. [SP2006] Seinfeld, John H, and Spyros N Pandis. Atmospheric Chemistry
       and Physics: From Air Pollution to Climate Change. Vol. 2nd. Wiley, 2006.
    See Also
    --------
    ka_cont : neglecting correction for non-continuum effects
    """
    ka_t = ka_cont(T)
    denom = 1.0 + (ka_t / (c.at * r * rho * c.Cp)) * np.sqrt(
        (2.0 * np.pi * c.Ma) / (c.R * T)
    )
    return ka_t / denom

def sigma_w(T):
    """ Surface tension of water for a given temperature.
    .. math::
        \\begin{equation}
        \sigma_w = 0.0761 - 1.55\\times 10^{-4}(T - 273.15)
        \end{equation}
    Parameters
    ----------
    T : float
        ambient air temperature, degrees K
    Returns
    -------
    float
        :math:`\sigma_w(T)` in J/m^2
    """
    return 0.0761 - 1.55e-4 * (T - 273.15)
