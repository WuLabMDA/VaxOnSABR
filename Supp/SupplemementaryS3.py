# Enabled to remove warnings for demo purposes.
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pathlib
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec

import sys
sys.path.append(r'C:\Users\mbsaad\Desktop\Other projects\Covid_vaccine\New_request\Utils_fn')
import fn_utils as utils
import fn_bias_correction as bc

'''-------------------------------------
        Corrected version
--------------------------------------'''
root_dir = pathlib.Path.cwd()
df_data = pd.read_excel(os.path.join(root_dir,"Covid_data.xlsx"),sheet_name='Extra')
x_scale = [0, 12, 24, 36, 48]
extra_space = 2

#---figure a-c---
'''time_col = 'RFS'
event_col = 'RFS_status'
k_min0 = 0
k_min1 = 0'''

#---figure d-f---
time_col = 'OS'
event_col = 'OS_status'
k_min0 = 0
k_min1 = 0

#───────All era───────
df_data = utils.censoring(df_data, time_col,event_col,48)

cols = ['MRN','any_vax_3m_3m','IO',time_col,event_col]
df_out = df_data[cols]
df_out['Time']  = df_out[time_col] 
df_out['Event']  = df_out[event_col]

df_out['TX_EndDate'] = df_data['RT.End.Date']
df_out['Event_Date'] = df_data['Vaccine_Date']
long_df = bc.build_timevarying_table(df_out)
figg,sm_group_0,sm_group_1 = bc.plot_simon_makuch([],long_df, 'TV-overall '+ time_col,x_scale,extra_space, k_min0=k_min0, k_min1=k_min1)

# ─────── check at-risk counts per group at tail ───────
t0, s0, r0 = sm_group_0
t1, s1, r1 = sm_group_1


print("Not Vaccinated -- last 5 time points:")
for t, r in zip(t0[-5:], r0[-5:]):
    print(f"  t={t:.1f} months,  n_at_risk={r}")

print("\nVaccinated -- last 5 time points:")
for t, r in zip(t1[-5:], r1[-5:]):
    print(f"  t={t:.1f} months,  n_at_risk={r}")
    
    
#───────Splits by arms ───────
with_io = df_out[df_out['IO']=='Yes'].reset_index(drop=True)
long_df_with = bc.build_timevarying_table(with_io)
figg,sm_group_0,sm_group_1 = bc.plot_simon_makuch([],long_df_with, 'TV-SABR+ICI_arm ' + time_col,  x_scale,extra_space, k_min0=0, k_min1=0)

without_io = df_out[df_out['IO']=='No'].reset_index(drop=True)
long_df_wo = bc.build_timevarying_table(without_io)
figg,sm_group_0,sm_group_1 = bc.plot_simon_makuch([],long_df_wo,'TV-SABR_arm ' + time_col,  x_scale,extra_space, k_min0=0, k_min1=0)

'''-------------------------------------
        Four groups comparison
--------------------------------------'''

long_df = long_df.rename(columns={'id':'MRN'})
df_merge = long_df.merge(df_data[['MRN','IO']], on='MRN', how='left')
df_merge['new_group'] = 'NA'
for i in range(len(df_merge)):
    row = df_merge.iloc[i,:]
    if (row['IO'] == 'No') & (row['vax_tv'] == 0):
        df_merge['new_group'][i] = 'sabr_no_vax'
    if (row['IO'] == 'No') & (row['vax_tv'] == 1):
        df_merge['new_group'][i] = 'sabr_with_vax'
    if (row['IO'] == 'Yes') & (row['vax_tv'] == 0):
        df_merge['new_group'][i] = 'isabr_no_vax'
    if (row['IO'] == 'Yes') & (row['vax_tv'] == 1):
        df_merge['new_group'][i] = 'isabr_with_vax'
        
        
df_merge = df_merge.rename(columns={'MRN':'id'})        
figg,sm_group_0,sm_group_1,sm_group_2,sm_group_3 = bc.plot_simon_makuch('Four',df_merge, 'TV-cox'+ time_col,x_scale,extra_space, k_min0=k_min0, k_min1=k_min1)