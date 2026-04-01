# Enabled to remove warnings for demo purposes.
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pathlib
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec

from lifelines import KaplanMeierFitter
from sksurv.nonparametric import kaplan_meier_estimator
from lifelines.statistics import logrank_test,multivariate_logrank_test
kmf = KaplanMeierFitter()
kmf1 = KaplanMeierFitter()
kmf2 = KaplanMeierFitter()
kmf3 = KaplanMeierFitter()
kmf4 = KaplanMeierFitter()
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test

import statsmodels.api as sm
from lifelines import CoxTimeVaryingFitter
from matplotlib.ticker import FormatStrFormatter
#import fn_utils as utils

import sys
sys.path.append(r'C:\Users\mbsaad\Desktop\Other projects\Covid_vaccine\New_request\Utils_fn')
import fn_utils as utils
import fn_bias_correction as bc

'''-------------------------------------
        Corrected version
--------------------------------------'''
root_dir = pathlib.Path.cwd()
df_data = pd.read_excel(os.path.join(root_dir,"Covid_data.xlsx"),sheet_name='Main')
x_scale = [0, 12, 24, 36, 48, 60]
extra_space = 2

time_col = 'OS'
event_col = 'OS_status'
k_min0 = 4
k_min1 = 4

'''time_col = 'RFS'
event_col = 'RFS_status'
k_min0 = 0
k_min1 = 0'''

cols = ['MRN','any_vax_3m_3m',time_col,event_col]
df_out = df_data[cols]
df_out['Time']  = df_out[time_col] 
df_out['Event']  = df_out[event_col]

df_out['TX_EndDate'] = df_data['RT.End.Date']
df_out['Event_Date'] = df_data['Vaccine_Date']
long_df = bc.build_timevarying_table(df_out)
figg,sm_group_0,sm_group_1 = bc.plot_simon_makuch([],long_df, 'TV-cox'+ time_col,x_scale,extra_space, k_min0=k_min0, k_min1=k_min1)

# ─────── check at-risk counts per group at tail ───────
t0, s0, r0 = sm_group_0
t1, s1, r1 = sm_group_1


print("Not Vaccinated -- last 5 time points:")
for t, r in zip(t0[-5:], r0[-5:]):
    print(f"  t={t:.1f} months,  n_at_risk={r}")

print("\nVaccinated -- last 5 time points:")
for t, r in zip(t1[-5:], r1[-5:]):
    print(f"  t={t:.1f} months,  n_at_risk={r}")
    
    
    

