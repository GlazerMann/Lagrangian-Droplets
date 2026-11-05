"""Shared time-coordinate validation for parcel and particle traces."""
import numpy as np

def relative_times(times):
    times=np.asarray(times,dtype=float)
    if times.ndim!=1 or not len(times) or not np.all(np.isfinite(times)):
        raise ValueError('Trace times must be a nonempty finite vector')
    if np.any(np.diff(times)<=0):
        raise ValueError('Trace times must be strictly increasing')
    return times-times[0]

def output_times(duration,dt):
    if not np.isfinite(duration) or duration<0:
        raise ValueError('Duration must be finite and nonnegative')
    if not np.isfinite(dt) or dt<=0:
        raise ValueError('dt must be finite and positive')
    if duration==0:
        return np.array([0.])
    times=np.arange(0.,duration,dt)
    return np.concatenate([times[times<duration],np.array([duration])])

def load_environment(filename,duration=None):
    import pickle
    import os
    require_complete_parcels(os.path.dirname(os.fspath(filename)))
    with open(filename,'rb') as stream:
        trace=pickle.load(stream)
    times=relative_times(trace['time'])
    temperature=np.asarray(trace['T'],dtype=float)
    ss=np.asarray(trace['SS'],dtype=float)
    if temperature.shape!=times.shape or ss.shape!=times.shape:
        raise ValueError('Environment values must match trace times')
    if not np.all(np.isfinite(temperature)) or np.any(temperature<=0) or not np.all(np.isfinite(ss)):
        raise ValueError('Environment must have finite values and positive Kelvin temperature')
    if duration is None:
        duration = trace.get('duration_seconds')
    if duration is not None:
        if duration > 0 and len(times) < 2:
            raise ValueError('Positive duration requires at least two environment samples')
        clock_scale=max(1.,abs(float(trace['time'][0])),abs(float(trace['time'][-1])))
        tolerance=32*np.finfo(float).eps*clock_scale
        if not np.isfinite(duration) or duration<0 or duration>times[-1]+tolerance:
            raise ValueError('Requested duration is outside the available parcel history')
        if duration>times[-1]:
            times[-1]=duration # recover a requested endpoint lost only to clock subtraction
        cropped=np.concatenate([times[times<duration],np.array([float(duration)])])
        temperature=np.interp(cropped,times,temperature)
        ss=np.interp(cropped,times,ss)
        times=cropped
    return times,temperature,ss


def trace_mean(times, values, definition='time'):
    times=np.asarray(times,dtype=float)
    values=np.asarray(values,dtype=float)
    if definition=='sample':
        return float(np.mean(values))
    if definition!='time':
        raise ValueError('Mean definition must be time or sample')
    if len(times)==1:
        return float(values[0])
    return float(np.sum(.5*(values[:-1]+values[1:])*np.diff(times))/(times[-1]-times[0]))


def require_complete_parcels(directory):
    """Readers must not combine completed and incomplete parcel generations."""
    import os
    if os.path.exists(os.path.join(directory,'parcel_run_in_progress.json')):
        raise RuntimeError('Parcel directory contains an incomplete run; rerun from num_start=0')
