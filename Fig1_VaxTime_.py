# Enabled to remove warnings for demo purposes.
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import sys
import pathlib
import os

import sys
sys.path.append(r'C:\Users\mbsaad\Desktop\Other projects\Covid_vaccine\New_request\Utils_fn')
import fn_utils as utils
import fn_bias_correction as bc
import fn_timing as time


root_dir = pathlib.Path.cwd()
df_data = pd.read_excel(os.path.join(root_dir,"Covid_data.xlsx"),sheet_name='Main')


'''--------------'''
#       OS
'''--------------'''
x_scale = [0, 12, 24, 36, 48, 60]
extra_space = 2

'''time_col = 'OS'
event_col = 'OS_status'
k_min0 = 4
k_min1 = 4'''

time_col = 'RFS'
event_col = 'RFS_status'
k_min0 = 0
k_min1 = 0

cols = ['MRN','Vaccine_Time',time_col,event_col]
df_out = df_data[cols]
df_out['Time']  = df_out[time_col] 
df_out['Event']  = df_out[event_col]
df_out['TX_EndDate'] = df_data['RT.End.Date']
df_out['Event_Date'] = df_data['Vaccine_Date']
long_df = bc.build_timevarying_table(df_out)

figg,sm_group_0,sm_group_1,sm_group_2,pairwise = time.plot_simon_makuch('time',long_df, 'TV-cox'+ time_col,x_scale,extra_space, k_min0=k_min0, k_min1=k_min0)
print('----------------')
print("Pre vs No:", pairwise[0])
print('----------------')
print("Post vs No:", pairwise[1])
print('----------------')
print("Pre vs Post:", pairwise[2])



'''--------------'''
#       RFS
'''--------------'''
'''x_scale = [0, 12, 24, 36, 48, 60]
extra_space = 2
cols = ['MRN','Vaccine_Time','RFS', 'RFS_status']
df_rfs = df_data[cols]
df_rfs['Time']  = df_rfs['RFS'] 
df_rfs['Event']  = df_rfs['RFS_status']
df_rfs['TX_EndDate'] = df_data['RT.End.Date']
df_rfs['Event_Date'] = df_data['Vaccine_Date']
long_df = bc.build_timevarying_table(df_rfs)
figg,sm_group_0,sm_group_1,sm_group_2,pairwise = time.plot_simon_makuch('time',long_df, 'TV-cox RFS',x_scale,extra_space, k_min0=0, k_min1=0)
print('----------------')
print("Pre vs No:", pairwise[0])
print('----------------')
print("Post vs No:", pairwise[1])
print('----------------')
print("Pre vs Post:", pairwise[2])'''

