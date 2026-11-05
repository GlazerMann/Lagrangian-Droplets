#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for storing DNS trajectories, driving offline microphysics, and vmaking verification figures

author: Laura Fierce
"""

import numpy as np
import helper

# =============================================================================
# specifications
# =============================================================================
trajectories_overdir = '/Users/fier887/Downloads/DNS_traces_230504__test/'
raw_dns_dir = '/Users/fier887/Downloads/New_cases (7-27-22)/'

Ddry = 110e-9
kappa = 1.2

# =============================================================================
# drive trajectories
# =============================================================================
case = 3
dns_filename = raw_dns_dir + 'case' + str(case) + '.mat'
helper.store_DNS_trajectories(dns_filename,trajectories_overdir,P0=101325.,add_verification=True)

trajectories_dir = trajectories_overdir + 'case' + str(case) + '/'
helper.store_s_stats(trajectories_dir)

s_offsets = np.array([0.])
helper.drive_many_trajectories(
    trajectories_dir,Ddry,kappa,s_offsets=s_offsets,read_verification=True,
    max_step=1.)

# =============================================================================
# make figures
# =============================================================================
trajectories_dir = trajectories_overdir + 'case' + str(case) + '/'
D_crit_all = helper.dsd_figs(trajectories_dir,60*10,plot_trajectories=True)
