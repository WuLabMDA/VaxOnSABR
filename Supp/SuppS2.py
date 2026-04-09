# Enabled to remove warnings for demo purposes.
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import sys
import pathlib
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys
sys.path.append(r'C:\Users\mbsaad\Desktop\Other projects\Covid_vaccine\New_request\Utils_fn')
import fn_utils as utils
import fn_bias_correction as bc


# ─────────FLu Vaccine────────
root_dir = pathlib.Path.cwd()
df_data = pd.read_excel(os.path.join(root_dir,"Covid_data_v2.xlsx"),sheet_name='Main')
x_scale = [0, 12, 24, 36, 48, 60]
extra_space = 2

'''time_col = 'OS'
event_col = 'OS_status'
k_min0 = 4
k_min1 = 4'''

time_col = 'RFS'
event_col = 'RFS_status'
k_min0 = 4
k_min1 = 4

cols = ['MRN','Had_flu_vaccine',time_col,event_col]
df_out = df_data[cols]
df_out['Time']  = df_out[time_col] 
df_out['Event']  = df_out[event_col]
df_out['TX_EndDate'] = df_data['RT.End.Date']
df_out['Event_Date'] = df_data['Flu_Vaccine_Date']
long_df = bc.build_timevarying_table(df_out)
figg,sm_group_0,sm_group_1 = bc.plot_simon_makuch([],long_df, 'TV-Flu Vaccine '+ time_col,x_scale,extra_space, k_min0=k_min0, k_min1=k_min1)

# ─────────Covid Infection────────
del cols, df_out
cols = ['MRN','Had_covid_infection',time_col,event_col]
df_out = df_data[cols]
df_out['Time']  = df_out[time_col] 
df_out['Event']  = df_out[event_col]
df_out['TX_EndDate'] = df_data['RT.End.Date']
df_out['Event_Date'] = df_data['Covid_infection_Date']
long_df = bc.build_timevarying_table(df_out)
figg,sm_group_0,sm_group_1 = bc.plot_simon_makuch([],long_df, 'TV-Covid_Infection '+ time_col,x_scale,extra_space, k_min0=k_min0, k_min1=k_min1)

