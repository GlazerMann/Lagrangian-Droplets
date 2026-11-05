#!/usr/bin/env python3
"""Make figures from physical-unit traces or explicitly identified legacy data."""
import argparse
import os
import numpy as np
import process

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default='.')
    parser.add_argument('--plot-dir',default='figs')
    parser.add_argument('--parcel-ids',nargs='+',type=int,default=[18,1,4])
    parser.add_argument('--legacy-representation',choices=['linear','lny','lnDp'])
    args=parser.parse_args()
    os.makedirs(args.plot_dir,exist_ok=True)
    plot_dir=args.plot_dir+os.sep
    process.plot_some_trajectories(args.parcel_ids,args.output_dir,avg_SS=0.,
        Slims=[-6,6],Dlims=[0.,15.],plot_dir=plot_dir,
        legacy_representation=args.legacy_representation)
    process.plot_activated_fraction_mean_SS(np.linspace(-.04,.03,15),args.output_dir,
        plot_dir=plot_dir,legacy_representation=args.legacy_representation)

if __name__=='__main__':
    main()
