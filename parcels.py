#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 19:06:25 2022

@author:  Jesse Anderson and Laura Fierce

"""

import numpy as np
import pickle
import json
import re
import tempfile
import os
from scipy.integrate import solve_ivp
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
import time, random, sys
import operator
from timeutils import output_times, relative_times
import matplotlib.pyplot as plt

def trajectories(num_trajectories, stop_time=0, num_start=0, LES_dir='OUT_3D/',
                 parcel_trace_dir='parcel_traces/', timesteps=range(90000,180000,3601),
                 sample_dt=0.1, max_step=0.1, P=100000., rng=None, rtol=1e-6, atol=1e-9):
    """Integrate absolute LES time and save resolved environmental histories.

    stop_time is a duration in seconds; zero retains the legacy until-end mode.
    sample_dt and LES snapshot times determine environmental output sampling,
    independently of the velocity solver's local error estimates. LES time
    coordinates must already be in seconds. Returned parcel time stays absolute;
    particle-growth readers explicitly convert it to age.
    """
    count, first_id = operator.index(num_trajectories), operator.index(num_start)
    if count < 1 or first_id < 0 or first_id >= count:
        raise ValueError('Require num_trajectories >= 1 and 0 <= num_start < count')
    output_times(1., sample_dt)
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError('max_step must be finite and positive')
    if not np.isfinite(stop_time) or stop_time < 0:
        raise ValueError('stop_time is a finite nonnegative duration')
    if not np.isfinite(P) or P<=0:
        raise ValueError('LES pressure must be finite and positive')
    if not np.isfinite(rtol) or rtol<=0:
        raise ValueError('Flow rtol must be finite and positive')
    atol=np.asarray(atol,dtype=float)
    if atol.shape not in ((),(3,)) or np.any(~np.isfinite(atol)) or np.any(atol<=0):
        raise ValueError('Flow atol must be a positive finite scalar or three-component vector')
    fields = {name: [] for name in ('U', 'V', 'W', 'TABS', 'QV')}
    time_parts = []
    grids = None
    for step in timesteps:
        filename = os.path.join(LES_dir, 'PiChamber_huji_19K_trj_32_%010d.nc' % int(step))
        with Dataset(filename) as dataset:
            current = tuple(read_les_variable(dataset,name,(name,)) for name in ('x','y','z'))
            if any(len(grid)<2 or not (np.all(np.diff(grid)>0) or np.all(np.diff(grid)<0)) for grid in current):
                raise ValueError('LES coordinate grids require at least two strictly monotone points')
            if grids is None:
                grids = current
            elif any(not np.array_equal(a,b) for a,b in zip(grids,current)):
                raise ValueError('LES spatial grids differ between snapshots')
            times = read_les_variable(dataset,'time',('time',))
            time_parts.append(times)
            for name in fields:
                values = read_les_variable(dataset,name,('time','z','y','x'))
                if values.shape != (len(times),len(current[2]),len(current[1]),len(current[0])):
                    raise ValueError('LES values do not match time and spatial grids')
                if (name=='TABS' and np.any(values<=0)) or (name=='QV' and np.any(values<0)):
                    raise ValueError('LES requires positive Kelvin temperature and nonnegative water mixing ratio')
                fields[name].append(values)
    if not time_parts:
        raise ValueError('No LES snapshots were selected')
    time_grid = np.concatenate(time_parts)
    relative_times(time_grid)
    if len(time_grid) < 2:
        raise ValueError('At least two LES times are required')
    interpolators = {}
    for name, parts in fields.items():
        values = np.concatenate(parts, axis=0)
        if name == 'QV':
            values = values/1000. # LES g/kg -> kg/kg
        interpolators[name] = RegularGridInterpolator((*grids,time_grid), values.transpose(3,2,1,0))
    spacing = (time_grid[1]-time_grid[0])/max(count-1,1)
    starts = time_grid[0] + spacing*np.arange(first_id,count)
    ends = starts+stop_time if stop_time else np.full(starts.shape,time_grid[-1])
    clock_tolerance = 32*np.finfo(float).eps*max(1.,abs(time_grid[0]),abs(time_grid[-1]))
    if np.any(ends > time_grid[-1]+clock_tolerance):
        raise ValueError('LES history does not cover the requested duration for every parcel')
    ends = np.minimum(ends, time_grid[-1]) # clamp only permitted floating-point endpoint roundoff
    existing={int(match.group(1)) for name in os.listdir(parcel_trace_dir)
              for match in [re.fullmatch(r'parcel_traces_(\d+)\.pkl',name)] if match} if os.path.isdir(parcel_trace_dir) else set()
    if existing.difference(range(count)):
        raise ValueError('Existing parcels are outside this run population; use a fresh parcel directory')
    incomplete=os.path.join(parcel_trace_dir,'parcel_run_in_progress.json')
    if first_id and (os.path.exists(incomplete) or set(range(first_id)).difference(existing)):
        raise ValueError('Restart requires complete earlier parcels; rerun from num_start=0 after an incomplete run')
    os.makedirs(parcel_trace_dir, exist_ok=True)
    with open(incomplete,'w') as stream:
        json.dump({'status':'in_progress','parcel_ids':list(range(count))},stream)
    rng = np.random.default_rng() if rng is None else rng
    for parcel_id, start, end in zip(range(first_id,count),starts,ends):
        initial = np.array([rng.uniform(np.min(grid),np.max(grid)) for grid in grids])
        sample_times = np.unique(np.concatenate([
            np.clip(start+output_times(end-start,sample_dt),start,end),
            time_grid[(time_grid>start)&(time_grid<end)], [start,end]]))
        if end == start:
            positions = initial[None,:]
        else:
            rhs = lambda t,position: find_Velocity(t,position,*grids,
                    interpolators['U'],interpolators['V'],interpolators['W'])
            solution = solve_ivp(rhs,[start,end],initial,t_eval=sample_times,max_step=max_step,
                                 rtol=rtol,atol=atol)
            if (not solution.success or not np.array_equal(solution.t,sample_times)
                    or np.asarray(solution.y).shape!=(3,len(sample_times))
                    or not np.all(np.isfinite(solution.y))):
                raise RuntimeError('Parcel %s integration failed: %s' % (parcel_id,solution.message))
            positions = RBC(solution.y.T,*grids)
        queries = np.column_stack([positions,sample_times])
        temperature = interpolators['TABS'](queries)
        mixing_ratio = interpolators['QV'](queries)
        supersaturation = SuperSaturation(mixing_ratio,temperature,P=P)
        if (not np.all(np.isfinite(temperature)) or np.any(temperature<=0)
                or not np.all(np.isfinite(mixing_ratio)) or np.any(mixing_ratio<0)
                or not np.all(np.isfinite(supersaturation))):
            raise RuntimeError('Interpolated LES environment is nonphysical or nonfinite')
        trajectory = dict(x=positions[:,0],y=positions[:,1],z=positions[:,2],
                          time=sample_times,elapsed_time=sample_times-start,
                          T=temperature,SS=supersaturation,
                          pressure_Pa=P,time_units='seconds',
                          duration_seconds=float(stop_time) if stop_time else float(end-start))
        trajectory['elapsed_time'][-1] = trajectory['duration_seconds']
        trajectory['mixing ratio'] = mixing_ratio
        filename = os.path.join(parcel_trace_dir,'parcel_traces_%06d.pkl' % parcel_id)
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(mode='wb',dir=parcel_trace_dir,delete=False) as stream:
                temporary=stream.name
                pickle.dump(trajectory,stream)
            os.replace(temporary,filename)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
    os.unlink(incomplete)


# ## Function that handles periodic boundary conditions
def RBC(position, X_grid, Y_grid, Z_grid):
    """Return reflected coordinates without altering the solver's state.

    This retains the existing mirrored lookup convention. It is not a change
    to a periodic wall model or an additional particle-inertia model.
    """
    result = np.array(position, dtype=float, copy=True)
    if result.shape[-1] != 3:
        raise ValueError('Positions must have three coordinates')
    for axis, grid in enumerate((X_grid, Y_grid, Z_grid)):
        lower, upper = float(np.min(grid)), float(np.max(grid))
        width = upper-lower
        if width <= 0:
            raise ValueError('Reflective coordinate grids must have positive width')
        folded = np.mod(result[..., axis]-lower, 2*width)
        result[..., axis] = lower + np.where(folded <= width, folded, 2*width-folded)
    return result
    


# ## Function that interpolates the velocity vectors 
def find_Velocity(t, position, X_grid, Y_grid, Z_grid, interpolate_U, interpolate_V, interpolate_W): 
    
    ## Reflective boundary conditions, keep the parcel in the box
    position = RBC(position, X_grid, Y_grid, Z_grid)
    
    x = position[0]
    y = position[1]
    z = position[2]

    dxdt = interpolate_U([x, y, z, t])  # interpolates the x wind component at x, y, z, t
    dydt = interpolate_V([x, y, z, t])  # interpolates the y wind component at x, y, z, t
    dzdt = interpolate_W([x, y, z, t])  # interpolates the z wind component at x, y, z, t
    velocity = np.array([dxdt, dydt, dzdt]) # velocity of parcel
    velocity = np.reshape(velocity,3)
    return velocity


# # Define the Calculation of the supersatration, (Magnus Approximation)
def SuperSaturation(mixing_ratio,temperature,P=100000.):
    P_atm = P/100. # Atmospheric pressure (hPa); legacy default retained
    T = temperature-273.15 #Convert to degrees C
    #Saturation vapor pressure (T)
    saturation_pressure = 6.1094*np.exp(17.625*T/(243.04+T)) #(hPa)
    #Actual Vapor Pressure
    vapor_pressure = (mixing_ratio)*P_atm/(0.622+(mixing_ratio)) # (hPa)
    #Supersaturation
    SS = vapor_pressure/saturation_pressure-1
    return SS


def read_les_variable(dataset,name,dimensions):
    """Preserve netCDF missing-value semantics before constructing interpolants."""
    variable=dataset[name]
    if variable.dimensions!=dimensions:
        raise ValueError('LES %s dimensions must be %s, got %s' % (name,dimensions,variable.dimensions))
    raw=variable[:]
    if np.any(np.ma.getmaskarray(raw)):
        raise ValueError('LES %s contains missing or masked values' % name)
    values=np.asarray(raw,dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError('LES %s contains nonfinite values' % name)
    return values
