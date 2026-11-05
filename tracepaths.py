"""Canonical paths for supersaturation runs; reject lossy input precision."""
from decimal import Decimal, InvalidOperation
import json
import os
import numpy as np

def supersaturation_key(value):
    value=float(value)
    if not np.isfinite(value):
        raise ValueError('Supersaturation must be finite')
    try:
        rounded=Decimal(str(value)).quantize(Decimal('0.000000000001'))
    except InvalidOperation as error:
        raise ValueError('Supersaturation cannot be represented in an output key') from error
    tolerance=32*np.finfo(float).eps*max(abs(value),1e-3)
    if abs(float(rounded)-value)>tolerance:
        raise ValueError('Supersaturation precision exceeds the supported 1e-12 grid')
    text=format(rounded,'f').rstrip('0').rstrip('.')
    if text in ('','-0'):text='0'
    return 'avgSS_'+text

def particle_directory(root,value,create=False):
    path=os.path.join(root,supersaturation_key(value))
    if create:
        os.makedirs(path,exist_ok=True)
        metadata=os.path.join(path,'supersaturation.json')
        if os.path.exists(metadata):
            with open(metadata) as stream:stored=json.load(stream)
            if supersaturation_key(stored['supersaturation'])!=supersaturation_key(value):
                raise ValueError('Output directory contains a different supersaturation')
        else:
            with open(metadata,'w') as stream:
                json.dump({'supersaturation':float(value),'key':supersaturation_key(value)},stream,indent=2)
    return path+os.sep
