#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 19:08:29 2022

@author: Laura Fierce

"""

import microphysics
import pickle, os, sys
import copy
import re
import subprocess
import tempfile
import json
from tracepaths import supersaturation_key, particle_directory
import numpy as np
from scipy.integrate import solve_ivp
from pyrcel.thermo import kohler_crit
from scipy.signal import savgol_filter
import warnings
from timeutils import output_times, relative_times, load_environment, trace_mean, require_complete_parcels


def make_directories(avg_SS, parcel_trace_dir='parcel_traces/', all_particle_trace_dir='particle_traces/',
                     all_particle_trace_dir_uniformS='particle_traces_uniformS/', plot_dir='figures/'):
    avg_SS = list(avg_SS)
    # Validate every scientific key before creating or overwriting any output.
    for value in avg_SS:
        supersaturation_key(value)
    for path in (parcel_trace_dir,all_particle_trace_dir,all_particle_trace_dir_uniformS,plot_dir):
        os.makedirs(path,exist_ok=True)
    for value in avg_SS:
        particle_directory(all_particle_trace_dir,value,create=True)
        particle_directory(all_particle_trace_dir_uniformS,value,create=True)
    


def main(avg_SS, dry_diameter, kappa, density, N, dt, parcel_trace_dir='parcel_traces/',all_particle_trace_dir = 'particle_traces/', constant_SS = False, odesystem_type='lny', ignore_Tp=True, odesolver='BDF', temperature_noise_fraction=0., rng=None, mean_definition='time', duration=None, parcel_ids=None, P0=101325., accom=0.3, rtol=1e-6, atol=None):
    
    print('\n')
    avg_SS = list(avg_SS)
    if len({supersaturation_key(value) for value in avg_SS}) != len(avg_SS):
        raise ValueError('Each run needs a distinct supersaturation output key')
    for value in avg_SS:
        supersaturation_key(value)
    print('Making particle traces...')
    print('\n')
    
    all_particle_traces = []
    for aa, SS in enumerate(avg_SS):
        print('SS offet = ', SS)
        particle_traces = many_trajectories(
                dry_diameter, kappa, density, N,
                odesystem_type=odesystem_type,
                avg_SS = SS,
                P0=P0, accom=accom, dt=dt,
                parcel_trace_dir = parcel_trace_dir, 
                ignore_Tp=ignore_Tp, smooth=False,
                all_particle_trace_dir=all_particle_trace_dir,
                run_as_avg = constant_SS, odesolver=odesolver, temperature_noise_fraction=temperature_noise_fraction, rng=rng, mean_definition=mean_definition, duration=duration, part_ii=parcel_ids, rtol=rtol, atol=atol)
        all_particle_traces.append(particle_traces)
        print('\n')
        
    return all_particle_traces

def split_to_nodes(avg_SS, number_of_parcels, stop_time=60., dt=0.5,
        parcel_trace_dir='parcel_traces/', all_particle_trace_dir='particle_traces/',
        all_particle_trace_dir_uniformS='particle_traces_uniformS/', N=1e6,
        dry_diameter=0.130e-6, kappa=1.0, density=2160., odesystem_type='lny', submit=False):
    """Write complete turbulent/uniform jobs; submit only when explicitly requested."""
    avg_SS = list(avg_SS)
    if len({supersaturation_key(value) for value in avg_SS}) != len(avg_SS):
        raise ValueError('Each job needs a distinct supersaturation output key')
    for value in avg_SS:
        supersaturation_key(value)
    if int(number_of_parcels)!=number_of_parcels or number_of_parcels<1:
        raise ValueError('number_of_parcels must be a positive integer')
    numeric = (stop_time, dt, N, dry_diameter, kappa, density)
    if (not all(np.isfinite(value) for value in numeric) or stop_time < 0 or dt <= 0
            or N < 0 or dry_diameter <= 0 or kappa <= 0 or density <= 0):
        raise ValueError('Invalid scientific parameters for generated jobs')
    parcel_trace_dir = os.fspath(parcel_trace_dir)
    all_particle_trace_dir = os.fspath(all_particle_trace_dir)
    all_particle_trace_dir_uniformS = os.fspath(all_particle_trace_dir_uniformS)
    scripts = []
    for index, value in enumerate(avg_SS):
        config = dict(avg_SS=[float(value)],number_of_parcels=int(number_of_parcels),
            duration=float(stop_time),dt=float(dt),dry_diameter=float(dry_diameter),
            kappa=float(kappa),density=float(density),N=float(N),odesystem_type=odesystem_type,
            parcel_trace_dir=parcel_trace_dir,all_particle_trace_dir=all_particle_trace_dir,
            all_particle_trace_dir_uniformS=all_particle_trace_dir_uniformS)
        main_filename = 'main_node%04d.py' % index
        code_lines = ['import run', 'config = '+repr(config),
            "parcel_ids = run.select_parcel_ids(config['parcel_trace_dir'],config.pop('number_of_parcels'))",
            "duration = config.pop('duration') or None",
            "uniform_root = config.pop('all_particle_trace_dir_uniformS')",
            "arguments = dict(config, parcel_ids=parcel_ids, duration=duration, ignore_Tp=True)",
            "run.main(**arguments, constant_SS=False)",
            "arguments['all_particle_trace_dir'] = uniform_root",
            "run.main(**arguments, constant_SS=True)"]
        code = '\n'.join(code_lines)+'\n'
        with open(main_filename,'w') as stream:
            stream.write(code)
        run_filename = 'run_node%04d.sh' % index
        lines = ['#!/bin/tcsh', '#SBATCH -A sooty2', '#SBATCH -p slurm',
                 '#SBATCH -t 2:00:00', '#SBATCH -N 1',
                 '#SBATCH -o node%04d.out' % index, '#SBATCH -e node%04d.err' % index,
                 '#SBATCH -J node%04d.exe.default' % index,
                 'module load gcc/5.2.0', 'module load netcdf', 'unset PYTHONHOME',
                 'module load python/3.7.2', 'python '+main_filename]
        with open(run_filename,'w') as stream:
            stream.write('\n'.join(lines)+'\n')
        os.chmod(run_filename,0o755)
        if submit:
            subprocess.run(['sbatch',run_filename],check=True)
        scripts.append(run_filename)
    return scripts
        

def many_trajectories(dry_diameter, kappa, density, N, avg_SS=0.,
                       P0=101325., accom=0.3, part_ii=None, dt=1.,
                       ignore_Tp=False, smooth=True, parcel_trace_dir='parcel_traces/',
                       all_particle_trace_dir='particle_traces/', odesystem_type='linear',
                       run_as_avg=False, add_ignore_Tp=True, odesolver='BDF',
                       temperature_noise_fraction=0., rng=None, mean_definition='time', duration=None, rtol=1e-6, atol=None):
    particle_trace_dir = particle_directory(all_particle_trace_dir, avg_SS)
    parcel_trace_nums = list_parcel_ids(parcel_trace_dir)
    if part_ii is not None:
        requested = set(np.atleast_1d(part_ii).tolist())
        if requested.difference(parcel_trace_nums):
            raise ValueError('Requested parcel IDs are not available')
        parcel_trace_nums = [i for i in parcel_trace_nums if i in requested]
    if not parcel_trace_nums:
        raise ValueError('No parcels were selected for this run')
    expected_files = {parcel_trace_nums[0]} if run_as_avg else set(parcel_trace_nums)
    existing_files = set()
    if os.path.isdir(particle_trace_dir):
        existing_files = {int(match.group(1)) for filename in os.listdir(particle_trace_dir)
                          for match in [re.fullmatch(r'particle_(\d+)\.pkl', filename)] if match}
    if existing_files.difference(expected_files):
        raise ValueError('Existing particle files are outside this run selection; use a fresh output directory')
    particle_trace_dir = particle_directory(all_particle_trace_dir, avg_SS, create=True)
    incomplete = os.path.join(particle_trace_dir, 'run_in_progress.json')
    with open(incomplete, 'w') as stream:
        json.dump({'status': 'in_progress', 'parcel_ids': parcel_trace_nums}, stream)
    particle_traces = {}
    uniform_trace = None
    for parcel_num in parcel_trace_nums:
        if run_as_avg and uniform_trace is not None:
            particle_traces[parcel_num] = copy.deepcopy(uniform_trace)
            continue
        filename = get_parcel_filename(parcel_num, parcel_trace_dir)
        environment = particle_environment(filename, P0, avg_SS, smooth, run_as_avg,
                    temperature_noise_fraction, rng, mean_definition, duration)
        arguments = dict(avg_SS=avg_SS, P0=P0, accom=accom, dt=dt, smooth=smooth,
                         run_as_avg=run_as_avg, odesystem_type=odesystem_type,
                         odesolver=odesolver, environment=environment, rtol=rtol, atol=atol)
        trace = one_particle(dry_diameter, kappa, density, N, filename,
                             ignore_Tp=ignore_Tp, **arguments)
        record = {'traces': trace}
        if add_ignore_Tp:
            record['traces_ignoreTp'] = one_particle(dry_diameter, kappa, density, N,
                                    filename, ignore_Tp=True, **arguments)
        r_crit, SS_crit = kohler_crit(float(trace_mean(trace['t'], trace['T'], 'time')),
                                    dry_diameter/2., kappa)
        record['props'] = dict(Ddry=dry_diameter, kappa=kappa, density=density, N=N,
                               Dp_crit=2*r_crit, SS_crit=SS_crit)
        write_particle_record(get_particle_filename(parcel_num, particle_trace_dir), record)
        particle_traces[parcel_num] = record
        if run_as_avg:
            uniform_trace = record
    os.unlink(incomplete)
    return particle_traces


def one_particle(dry_diameter, kappa, density, N, parcel_filename, avg_SS=0., P0=101325., accom=0.3, dt=1., ignore_Tp=True, odesystem_type='linear', smooth=True, run_as_avg=False, tau0=False, odesolver='BDF', environment=None, temperature_noise_fraction=0., rng=None, mean_definition='time', duration=None, rtol=1e-6, atol=None):
    if environment is None:
        environment = particle_environment(parcel_filename, P0, avg_SS, smooth, run_as_avg,
                    temperature_noise_fraction, rng, mean_definition, duration)
    mixing_ratio_fun, temp_fun, SS_fun, t_end = environment
    temp_0 = float(temp_fun(0.))
    # Preserve the existing nonactivated initialization policy in supersaturated air.
    SS_0 = min(float(SS_fun(0.)), 0.)
    diameter, _, _ = microphysics.equilibrate_h2o(
        np.array([dry_diameter]), np.array([kappa]), np.array([N]), temp_0, SS_0, P0)
    initial = np.array([diameter[0], temp_0])
    callbacks = {'linear': microphysics.one_particle,
                 'lny': microphysics.one_particle_lny,
                 'lnDp': microphysics.one_particle_lnDp}
    if odesystem_type not in callbacks:
        raise ValueError('Unknown ODE state representation: %s' % odesystem_type)
    q_0 = initial.copy()
    if odesystem_type == 'lny':
        q_0 = np.log(q_0)
    elif odesystem_type == 'lnDp':
        q_0[0] = np.log(q_0[0])
    rhs = lambda t, state: callbacks[odesystem_type](state, t,
        mixing_ratio_fun(t), SS_fun(t), temp_fun(t), P0, dry_diameter,
        kappa, N, accom, density, ignore_Tp=ignore_Tp)
    if not np.isfinite(rtol) or rtol <= 0:
        raise ValueError('rtol must be finite and positive')
    default_atol = {'linear': [1e-12, 1e-6], 'lny': [1e-8, 1e-8], 'lnDp': [1e-8, 1e-6]}
    state_atol = np.asarray(default_atol[odesystem_type] if atol is None else atol, dtype=float)
    if state_atol.shape not in ((), (2,)) or np.any(~np.isfinite(state_atol)) or np.any(state_atol <= 0):
        raise ValueError('atol must be a positive finite scalar or a two-component vector')
    times = output_times(t_end, dt)
    if t_end == 0.:
        states = initial[:, None]
    else:
        failure = None
        message = 'no solution returned'
        for options in ({'max_step': 1e-2}, {'max_step': 1e-3},
                        {'max_step': 1e-3, 'rtol': min(rtol, 1e-10), 'atol': state_atol*1e-3}):
            try:
                solution = solve_ivp(rhs, [0., t_end], q_0, method=odesolver,
                                     t_eval=times, **dict({'rtol': rtol, 'atol': state_atol}, **options))
                values = np.asarray(solution.y)
                if (solution.success and values.shape == (2, len(times))
                        and np.array_equal(solution.t, times) and np.all(np.isfinite(values))):
                    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                        physical = state_to_physical(values, odesystem_type)
                    if (np.all(np.isfinite(physical)) and np.all(physical[0] > 0)
                            and np.all(physical[1] > 0)
                            and np.all(physical[0] >= dry_diameter*(1.-1e-8))):
                        states = physical
                        break
                    message = 'solver returned a nonphysical particle state'
                else:
                    message = str(solution.message)
            except (ValueError, RuntimeError, FloatingPointError) as error:
                failure = error
                message = str(error)
        else:
            raise RuntimeError('Particle integration failed for %s after three attempts: %s'
                               % (parcel_filename, message)) from failure
    return {'t': times, 'Dp': states[0], 'Tp': states[1],
            'mixing_ratio': mixing_ratio_fun(times), 'T': temp_fun(times),
            'SS': SS_fun(times), 'state_representation': 'physical', 'schema_version': 2}




def unravel_particle_traces(particle_traces):
    """Unpack an ensemble only when all samples describe the same time grid.

    Missing logarithmic states must never be padded with zero (a one-meter
    diameter after exponentiation). Resampling/population policy is explicit
    caller work rather than an implicit choice made by this helper.
    """
    if not particle_traces:
        raise ValueError('Cannot unpack an empty particle ensemble')
    records = [dict(record, traces=physical_trace(record['traces']))
               for record in particle_traces.values()]
    ts = np.asarray(records[0]['traces']['t'])
    if ts.ndim != 1 or not len(ts):
        raise ValueError('Particle time grid must be a nonempty vector')
    fields = ('Dp', 'Tp', 'T', 'mixing_ratio', 'SS')
    for record in records:
        trace = record['traces']
        if not np.array_equal(np.asarray(trace['t']), ts):
            raise ValueError('Particle traces have incompatible time grids')
        if any(np.asarray(trace[key]).shape != ts.shape for key in fields):
            raise ValueError('Particle values do not match their time grid')
    arrays = [np.stack([record['traces'][key] for record in records]) for key in fields]
    properties = [np.asarray([record['props'][key] for record in records])
                  for key in ('SS_crit', 'Dp_crit', 'Ddry', 'kappa')]
    return (ts.copy(), *arrays, *properties)

def unravel_particle_trace(particle_trace):
    values = unravel_particle_traces({0: particle_trace})
    return (values[0], *(array[0] for array in values[1:]))

#import matplotlib.pyplot as plt

def get_particle_filename(ii, particle_trace_dir):
    return os.path.join(particle_trace_dir, 'particle_%09d.pkl' % ii)
   
def read_particle_traces(particle_trace_dir, legacy_representation=None):
    require_complete_run(particle_trace_dir)
    result = {}
    for filename in sorted(os.listdir(particle_trace_dir)):
        match = re.fullmatch(r'particle_(\d+)\.pkl', filename)
        if match:
            particle_id = int(match.group(1))
            if particle_id in result:
                raise ValueError('Duplicate particle ID in trace directory')
            result[particle_id] = read_particle_trace(particle_id, particle_trace_dir, legacy_representation)
    return result

def read_particle_trace(ii, particle_trace_dir, legacy_representation=None):
    require_complete_run(particle_trace_dir)
    particle_filename = get_particle_filename(ii, particle_trace_dir)
    with open(particle_filename, 'rb') as stream:
        record = pickle.load(stream)
    record = copy.deepcopy(record)
    for key in ('traces', 'traces_ignoreTp'):
        if key in record:
            record[key] = physical_trace(record[key], legacy_representation)
    return record

def get_parcel_filename(parcel_num, parcel_trace_dir='parcel_traces/'):
    return os.path.join(parcel_trace_dir, 'parcel_traces_%06d.pkl' % parcel_num)
    
def read_parcel_trace(parcel_num,parcel_trace_dir = 'parcel_traces/'):
    require_complete_parcels(parcel_trace_dir)
    parcel_filename = get_parcel_filename(parcel_num,parcel_trace_dir=parcel_trace_dir)#parcel_trace_dir + '/parcel_traces' + str(parcel_num).zfill(4)
    f = open(parcel_filename,'rb')
    trajectory = pickle.load(f)
    return trajectory

def environmental_trace_funs(parcel_filename, P0=101325., avg_SS=0., smooth=True,
                             temperature_noise_fraction=0., rng=None, mean_definition='time', duration=None):
    time_trace, temperature, supersaturation = load_environment(parcel_filename, duration)
    if smooth and len(temperature) >= 3:
        window = min(151, len(temperature) if len(temperature)%2 else len(temperature)-1)
        temperature = savgol_filter(temperature, window, min(2, window-1))
    if not np.isfinite(temperature_noise_fraction) or temperature_noise_fraction < 0:
        raise ValueError('Temperature noise fraction must be finite and nonnegative')
    if temperature_noise_fraction:
        rng = np.random.default_rng() if rng is None else rng
        temperature = temperature - rng.normal(0., temperature_noise_fraction*temperature)
    if np.any(temperature <= 0):
        raise ValueError('Temperature perturbation produced nonpositive Kelvin values')
    supersaturation = supersaturation + avg_SS - trace_mean(time_trace, supersaturation, mean_definition)
    mixing_ratio = microphysics.MixingRatio(supersaturation, temperature, P0)
    return (lambda t: np.interp(t, time_trace, mixing_ratio),
            lambda t: np.interp(t, time_trace, temperature),
            lambda t: np.interp(t, time_trace, supersaturation), float(time_trace[-1]))

def environmental_trace_funs_constant_SS(parcel_filename, P0=101325., avg_SS=0.,
                                         smooth=True, mean_definition='time', duration=None):
    times, temperature, _ = load_environment(parcel_filename, duration)
    mean_temperature = trace_mean(times, temperature, mean_definition)
    values = np.full(len(times), mean_temperature)
    ss = np.full(len(times), avg_SS)
    mixing_ratio = microphysics.MixingRatio(ss, values, P0)
    return (lambda t: np.interp(t, times, mixing_ratio),
            lambda t: np.interp(t, times, values),
            lambda t: np.interp(t, times, ss), float(times[-1]))

def state_to_physical(values, representation):
    values = np.array(values, dtype=float, copy=True)
    if representation == 'lny':
        values = np.exp(values)
    elif representation == 'lnDp':
        values[0] = np.exp(values[0])
    elif representation != 'linear':
        raise ValueError('Unknown state representation: %s' % representation)
    return values


def physical_trace(trace, legacy_representation=None):
    trace = copy.deepcopy(trace)
    representation = trace.get('state_representation')
    if representation == 'physical':
        return trace
    if representation is None:
        if legacy_representation is None:
            raise ValueError('Unmarked legacy trace: pass legacy_representation=lny, lnDp, or linear explicitly')
        representation = legacy_representation
    values = state_to_physical(np.stack([trace['Dp'], trace['Tp']]), representation)
    trace.update(Dp=values[0], Tp=values[1], state_representation='physical', schema_version=2)
    return trace


def particle_environment(filename, P0, avg_SS, smooth, run_as_avg,
                         temperature_noise_fraction=0., rng=None, mean_definition='time', duration=None):
    options = dict(P0=P0, avg_SS=avg_SS, smooth=smooth, mean_definition=mean_definition, duration=duration)
    if run_as_avg:
        return environmental_trace_funs_constant_SS(filename, **options)
    return environmental_trace_funs(filename, temperature_noise_fraction=temperature_noise_fraction,
                                    rng=rng, **options)


def list_parcel_ids(parcel_trace_dir):
    require_complete_parcels(parcel_trace_dir)
    return sorted(int(match.group(1)) for name in os.listdir(parcel_trace_dir)
                  for match in [re.fullmatch(r'parcel_traces_(\d+)\.pkl',name)] if match)


def select_parcel_ids(parcel_trace_dir, count):
    available = list_parcel_ids(parcel_trace_dir)
    if count < 1 or int(count) != count or len(available) < count:
        raise ValueError('Requested %s parcels, but only %s are available' % (count, len(available)))
    return available[:int(count)]


def require_complete_run(particle_trace_dir):
    if os.path.exists(os.path.join(particle_trace_dir, 'run_in_progress.json')):
        raise RuntimeError('Particle output directory contains an incomplete run; rerun successfully first')


def write_particle_record(filename, record):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=os.path.dirname(filename), delete=False) as stream:
            temporary = stream.name
            pickle.dump(record, stream)
        os.replace(temporary, filename)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
