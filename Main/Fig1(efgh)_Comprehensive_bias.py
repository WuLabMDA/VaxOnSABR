# Enabled to remove warnings for demo purposes.
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pathlib
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate
import sys
sys.path.append(r'C:\Users\mbsaad\Desktop\Other projects\Covid_vaccine\New_request\Utils_fn')
import fn_utils as utils
import fn_bias_correction as bc



    

'''-------------------------------------
       Step-1 Find 1:1 Match
--------------------------------------'''
root_dir = pathlib.Path.cwd()
#df_data = pd.read_excel(os.path.join(root_dir,"Covid_data.xlsx"),sheet_name='Main')
df_data = pd.read_excel(os.path.join(root_dir,"Covid_data_v2.xlsx"),sheet_name='Main')

cols = ['MRN','any_vax_3m_3m','Age','Gender','TumorSize','ECOG','Histology','Smoking_history','Race','Had_covid_infection']
base_df = df_data[cols].copy()
base_df = base_df.rename(columns={'any_vax_3m_3m':'Treatment'})
hist_ref = 'ADC'
hist_other = 'Non-' + hist_ref
#base_df = base_df.rename(columns={'TumorSize_ori':'TumorSize','ECOG_ori':'ECOG','Histology_ori':'Histology','Smoking_history_ori':'Smoking_history'})
base_df = utils.mapping(base_df,hist_ref,hist_other)
base_df['Treatment']  = base_df['Treatment'].map({'No':0,'Yes':1})

covars = ['Age','TumorSize',"Gender","ECOG",'Smoking_history'] 
cal = 0.2
matched = utils.PSM(covars,base_df,cal)
utils.plot_ps_distributions(base_df, matched, ps_col="ps", treat_col="Treatment")
smd_before = utils.compute_smd_table(base_df,covars).rename(columns={"SMD": "SMD_before"})
smd_after  = utils.compute_smd_table(matched,covars).rename(columns={"SMD": "SMD_after"})
smd = smd_before.merge(smd_after, on="Variable", how="inner")
print(tabulate(smd, headers='keys', tablefmt='pretty', showindex=False))
utils.loveplot(smd)

del base_df

'''-------------------------------------
        Step-1 Correcting biases 
--------------------------------------'''
##time_col = 'OS'
#event_col = 'OS_status'

time_col = 'RFS'
event_col = 'RFS_status'

# ─────── Time-bias correction ───────
df_data = df_data[df_data['MRN'].isin(matched['MRN'])].reset_index(drop=True)
cols = list(cols) +[time_col,event_col,'RT.End.Date','Vaccine_Date']
base_df = df_data[cols].copy()

base_df = base_df.rename(columns={'any_vax_3m_3m':'Vaccine',time_col:'Time',event_col:'Event',
                                  'RT.End.Date':'TX_EndDate','Vaccine_Date':'Event_Date'})

x_scale = [0, 12, 24, 36, 48, 60]
extra_space = 2
cols = ['MRN','Vaccine','Time', 'Event','TX_EndDate','Event_Date']
df_bias = base_df[cols]
long_df = bc.build_timevarying_table(df_bias)
figg,sm_group_0,sm_group_1 = bc.plot_simon_makuch([],long_df, 'TV-cox' + time_col, x_scale,extra_space,k_min0=4, k_min1=0)
t0, s0, r0 = sm_group_0
t1, s1, r1 = sm_group_1
print("Not Vaccinated -- last 5 time points:")
for t, r in zip(t0[-5:], r0[-5:]):print(f"  t={t:.1f} months,  n_at_risk={r}")

print("\nVaccinated -- last 5 time points:")
for t, r in zip(t1[-5:], r1[-5:]):print(f"  t={t:.1f} months,  n_at_risk={r}")

# ─────── Confounder-bias correction ───────

# -------------
# Preprocess 
# -------------
base_df['Age'] = round(base_df['Age'],0)
median = np.median(base_df['Age'])
base_df['Age'] = np.where(base_df['Age'] <=median, '<=75', '>75')
base_df['Smoking_history'] = np.where(base_df['Smoking_history'] == 'Never', 'No', 'Yes')
base_df['ECOG'] = np.where(base_df['ECOG'] <=1, '0-1', '2-3')
base_df['Histology'] = np.where(base_df['Histology'] == hist_ref,hist_ref,hist_other)
base_df['TumorSize'] = pd.to_numeric(base_df['TumorSize'])
cut_off = 3
base_df['TumorSize'] = np.where(base_df['TumorSize'] <cut_off, '<3cm', '>=3cm')
base_df = base_df.rename(columns={'Smoking_history':'Smoker','Event_Date':'Vaccine_Date'})
#base_df = base_df.drop(columns=['TX_EndDate','Event_Date','Smoking_history'],axis=1)

# --------------------
# Locking reference
# --------------------
base_df["Gender"] = pd.Categorical(base_df["Gender"], categories=["Male","Female"])
base_df["Smoker"] = pd.Categorical(base_df["Smoker"], categories=["Yes","No"])
base_df["Had_covid_infection"] = pd.Categorical(base_df["Had_covid_infection"], categories=["Yes","No"])
base_df["Race"] = pd.Categorical(base_df["Race"], categories=["White","Non_white"])  # adjust to your data
base_df["ECOG"] = pd.Categorical(base_df["ECOG"], categories=["0-1","2-3"])
base_df["Histology"] = pd.Categorical(base_df["Histology"], categories=[hist_ref,hist_other])
base_df["TumorSize"] = pd.Categorical(base_df["TumorSize"], categories=[">=3cm","<3cm"])


# -----------------------------
#  Adjustment & Forest Plot
# -----------------------------
covariates = ["Age","TumorSize","Gender","Smoker",'Race',"Had_covid_infection",'Histology','ECOG']
df_bias = base_df.copy() 
uni_test,multi_test,summ_test = bc.run_correct_sel_bias(df_bias, covariates, pen=0.0)
results = pd.DataFrame({
    "Model": ["Univariate-" + time_col, "Adjusted-" + time_col],
    "p-value": [uni_test['p'][0], multi_test['p'][0]]
})

print(results)

summ_test = summ_test.reset_index(drop=False)
utils.build_forest_plot(summ_test,time_col)

del df_bias



