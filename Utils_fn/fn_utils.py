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
from matplotlib.ticker import FixedLocator, FixedFormatter

from lifelines import KaplanMeierFitter
from sksurv.nonparametric import kaplan_meier_estimator
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines import CoxTimeVaryingFitter
import statsmodels.api as sm
from lifelines import CoxPHFitter

import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate



def create_group(long_df,info):
    
    for i in range(len(long_df)):
        curr_id = long_df['id'][i]
        row = info[info['MRN']==curr_id].reset_index(drop=True)
        if row.IO[0] == 'No':
            long_df['IO'][i] = 0
        else:
            long_df['IO'][i] = 1
            
    for i in range(len(long_df)):
        if (long_df['vax_tv'][i] == 0) and (long_df['IO'][i] == 0):
            long_df['Group'][i] = 'SABR_no_vax'
        elif (long_df['vax_tv'][i] == 1) and (long_df['IO'][i] == 0):
            long_df['Group'][i] = 'SABR_with_vax'
        elif (long_df['vax_tv'][i] == 0) and (long_df['IO'][i] == 1):
            long_df['Group'][i] = 'I-SABR_no_vax'
        elif (long_df['vax_tv'][i] == 1) and (long_df['IO'][i] == 1):
            long_df['Group'][i] = 'I-SABR_with_vax'
            
    return long_df


def censoring(df, time_col, event_col, horizon_months):
    d = df.copy()

    t_orig = d[time_col].astype(float).values
    e_orig = d[event_col].astype(int).values

    # Admin censoring: if original time > horizon, force censor (event=0)
    e_new = np.where(t_orig > horizon_months, 0, e_orig)

    # Cap time at horizon
    t_new = np.minimum(t_orig, horizon_months)

    d[time_col] = t_new
    d[event_col] = e_new
    return d


'''------------------------------'''
#       1-to-1 Matching
'''------------------------------'''
def mapping(temp_df,hist_ref,hist_other):
    #---Age---
    temp_df['Age'] = round(temp_df['Age'],0)
    median = np.median(temp_df['Age'])
    temp_df['Age'] = np.where(temp_df['Age'] <=median, '<=75', '>75')
    temp_df['Age']  = temp_df['Age'].map({'<=75':0,'>75':1})
    #---TumorSize---
    temp_df['TumorSize'] = pd.to_numeric(temp_df['TumorSize'])
    cut_off = 3
    temp_df['TumorSize'] = np.where(temp_df['TumorSize'] <cut_off, '<3cm', '>=3cm')
    temp_df['TumorSize']  = temp_df['TumorSize'].map({'<3cm':0,'>=3cm':1})
    #---ECOG---
    temp_df['ECOG'] = np.where(temp_df['ECOG'] <=1, '0-1', '2-3')
    temp_df['ECOG']  = temp_df['ECOG'].map({'0-1':0,'2-3':1})
    #---Histology---
    temp_df['Histology'] = np.where(temp_df['Histology'] ==hist_ref, hist_ref,hist_other)
    temp_df['Histology']  = temp_df['Histology'].map({hist_other:0,hist_ref:1})
    #---Smoker---
    temp_df['Smoking_history'] = np.where(temp_df['Smoking_history'] == 'Never', 'No', 'Yes')
    temp_df['Smoking_history']  = temp_df['Smoking_history'].map({'No':0,'Yes':1})
    #---Gender---
    temp_df['Gender']  = temp_df['Gender'].map({'Female':0,'Male':1})
    #---Race---
    temp_df['Race']  = temp_df['Race'].map({'Non_white':0,'White':1})
    return temp_df
        
def PSM(covars,temp_df,cal):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(temp_df[covars], temp_df["Treatment"])
    ps = lr.predict_proba(temp_df[covars])[:, 1]
    temp_df["ps"] = ps
    eps = 1e-6 # Logit transform (for caliper)
    logit_ps = np.log(ps.clip(eps, 1 - eps) / (1 - ps.clip(eps, 1 - eps)))
    temp_df["logit_ps"] = logit_ps
    
    #--greedy mathc (no replacement) on logit ps---
    treated = temp_df[temp_df["Treatment"] == 1].copy()
    control = temp_df[temp_df["Treatment"] == 0].copy()
    caliper = cal * temp_df["logit_ps"].std(ddof=1)
    
    control_pool = control.copy()
    control_pool["matched"] = False
    pairs = []
    
    for i, row in treated.sort_values("logit_ps").iterrows():
        diffs = np.abs(control_pool.loc[~control_pool["matched"], "logit_ps"] - row["logit_ps"])
        if diffs.empty:
            continue
        j = diffs.idxmin()
        if diffs.loc[j] <= caliper:
            control_pool.at[j, "matched"] = True
            pairs.append((i, j))
    m_t = temp_df.loc[[i for (i, j) in pairs]].copy()
    m_c = temp_df.loc[[j for (i, j) in pairs]].copy()
    m_t["pair_id"] = range(len(m_t))
    m_c["pair_id"] = range(len(m_c))
    matched = pd.concat([m_t, m_c], axis=0, ignore_index=True)
    print(f"Matched pairs: {len(m_t)} (out of {len(treated)})")
    return matched

def plot_ps_distributions(before_df, after_df, ps_col="ps", treat_col="Treatment"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, df, title in zip(axes, [before_df, after_df], ["Before matching", "After matching"]):
        t = df[df[treat_col] == 1][ps_col].dropna().astype(float)
        c = df[df[treat_col] == 0][ps_col].dropna().astype(float)

        ax.hist(c, bins=20, alpha=0.6, density=True, label="Control")
        ax.hist(t, bins=20, alpha=0.6, density=True, label="Treated")

        ax.set_title(title)
        ax.set_xlabel("Propensity score")
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    plt.show()
    
def loveplot(smd):
    smd["abs_before"] = smd["SMD_before"].abs()
    smd["abs_after"]  = smd["SMD_after"].abs()
    smd = smd.sort_values("abs_before", ascending=True).reset_index(drop=True)
    y = np.arange(len(smd))
    
    plt.figure(figsize=(7, max(3, 0.35 * len(smd) + 1)))
    plt.scatter(smd["abs_before"], y, label="Before")
    plt.scatter(smd["abs_after"],  y, label="After")
    
    plt.axvline(0.1, linestyle="--")  # common threshold
    plt.yticks(y, smd["Variable"])
    plt.xlabel("Absolute SMD")
    plt.title("Covariate balance (Love plot)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def smd_cont(x_t, x_c):
    mt, mc = np.mean(x_t), np.mean(x_c)
    st, sc = np.var(x_t, ddof=1), np.var(x_c, ddof=1)
    sp = np.sqrt((st + sc) / 2)
    return 0.0 if sp == 0 else (mt - mc) / sp

def smd_cat(x_t, x_c):
    # for categorical vars (Histology)
    cats = sorted(set(x_t).union(set(x_c)))
    smd_sum = 0
    for c in cats:
        pt = np.mean(x_t == c)
        pc = np.mean(x_c == c)
        p = (pt + pc) / 2
        smd_sum += (pt - pc)**2 / (p*(1 - p) + 1e-8)
    return np.sqrt(0.5 * smd_sum)

def compute_smd_table(df_in,covars):
    Tmask = df_in["Treatment"] == 1
    Cmask = df_in["Treatment"] == 0
    out = []
    for v in covars:
        if v == "Histology":
            val = smd_cat(df_in.loc[Tmask, v], df_in.loc[Cmask, v])
        else:
            val = smd_cont(df_in.loc[Tmask, v], df_in.loc[Cmask, v])
        out.append({"Variable": v, "SMD": round(val, 3)})
    return pd.DataFrame(out)



'''------------------------------'''
#       Forest plot
'''------------------------------'''

def categorize_row(row):
    if row['p'] < 0.05:
        if row['exp(coef)'] < 1:
            return 1  # Protective (green) - show first
        else:
            return 3  # Harmful (red) - show last
    else:
        return 2  # Non-significant (gray) - show middle
    
def build_forest_plot(df,endp):
    df['sort_category'] = df.apply(categorize_row, axis=1)
    df = df.sort_values(['sort_category', 'p'], ascending=[True, True])
    df = df.iloc[::-1].reset_index(drop=True) # reverse plot (first item appear at top)

    print("==========================")
    print("Order of variables (top to bottom):")
    print("==========================")
    
    for i, row in df.iterrows():
        print(f"{i+1}. {row['covariate']} - p={row['p']:.4f}, HR={row['exp(coef)']:.2f}, CI=({row['exp(coef) lower 95%']:.2f}-{row['exp(coef) upper 95%']:.2f})")

    data = {'variable': df['covariate'].tolist(),'hr': df['exp(coef)'].tolist(),'lower': df['exp(coef) lower 95%'].tolist(),
            'upper': df['exp(coef) upper 95%'].tolist(),'p': df['p'].tolist()}
    
    #---Set publicationb quality---
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['figure.dpi'] = 600
    
    #----------Dynamically determine x-axis limits based on data---------
    min_ci = min(data['lower'])
    max_ci = max(data['upper'])
    
    #---Add alternating background colors FIRST------------
    fig, ax = plt.subplots(figsize=(11, 6))
    n = len(data['variable'])
    y_positions = np.arange(n)
    for i in range(n):
        if i % 2 == 0:  # even position get gray
            rect = Rectangle(
                (0.001, i - 0.5), # (x,y)
                1000, # width
                1, # height
                facecolor='#f0f0f0', # light gray
                edgecolor='none',
                zorder=0) # behind everything
            ax.add_patch(rect)
    #------Determine colors based on significance-------
    colors = []
    for i in range(n):
        if data['p'][i] < 0.05:
            if data['hr'][i] < 1:
                #colors.append('#27AE60')  # Green - protective
                colors.append('#4169E1')
            else:
                colors.append('#E74C3C') # Red - harmful
        else:
            colors.append('#95A5A6') # Gray - not significant
            
    #------Plot confidence intervals--------------
    for i in range(n):
        ax.plot([data['lower'][i], data['upper'][i]], [i, i],color=colors[i], linewidth=3, solid_capstyle='butt', zorder=2)
        cap_height = 0.15
        ax.plot([data['lower'][i], data['lower'][i]], [i-cap_height, i+cap_height],color=colors[i], linewidth=3, solid_capstyle='butt', zorder=2)
        ax.plot([data['upper'][i], data['upper'][i]], [i-cap_height, i+cap_height],color=colors[i], linewidth=3, solid_capstyle='butt', zorder=2)
    
    #----Plot HR points as diamonds------
    for i in range(n):
        ax.scatter(data['hr'][i], i, marker='D', s=120,color=colors[i], edgecolors='#2c3e50', linewidths=1.5, zorder=3)
    ax.axvline(x=1, color='#2c3e50', linestyle='--', linewidth=2, zorder=1) # reference line HR=1
    
    #----dynamic xlim based on actual data range------
    ax.set_xscale('log')  # log scale
    min_ci = min(data['lower'])
    max_ci = max(data['upper'])
    # add 20% breathing room on each side (in log space)
    x_left  = np.exp(np.log(min_ci) - 0.1 * (np.log(max_ci) - np.log(min_ci)))
    x_right = np.exp(np.log(max_ci) + 0.1 * (np.log(max_ci) - np.log(min_ci)))
    ax.set_xlim(x_left, x_right)
    
    #---choose ticks that fall within the visible (original scale) ---
    all_ticks = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    ticks = [t for t in all_ticks if x_left <= t <= x_right]
    if 1.0 not in ticks:
        ticks.append(1.0)
        ticks = sorted(ticks)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_locator(FixedLocator([]))  # remove minor ticks

    # Set y-axis
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(data['variable'])
    
    # Labels and title
    ax.set_xlabel('Hazard Ratio', fontsize=13, fontweight='bold')
    ax.set_title(endp +' :Time-varying Multivariate Cox',fontsize=14, fontweight='bold', pad=20)
    
    # Add text annotations for "Favors" labels at the top
    #ax.text(x_left * 1.05, n-0.2, 'Favors Lower Hazard',color='#27AE60', fontsize=11, fontweight='bold', ha='left', va='bottom')
    #ax.text(x_right * 0.95, n-0.2, 'Favors Higher Hazard',color='#E74C3C', fontsize=11, fontweight='bold', ha='right', va='bottom')
    
    # Add HR values on the right side
    for i in range(n):
        hr_text = f"{data['hr'][i]:.2f} ({data['lower'][i]:.2f}-{data['upper'][i]:.2f})"
        ax.text(x_right * 1.35, i, hr_text, fontsize=10, va='center', ha='left')
        
    # Add column header for HR values
    ax.text(x_right * 1.35, n-0.5, 'HR (95% CI)', fontsize=11, fontweight='bold', va='center', ha='left')
    
    #----Styling------
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5, zorder=0)
    
    ax.set_facecolor('white') # Set white background for the plot area
    plt.tight_layout()
    plt.show()
                
            
            
    
'''------------------------------'''
#       Uncorrected (Naive)
'''------------------------------'''

def naive_km_metric(T1, T2, E1, E2):

    # ─────── log-rank p-value ───────
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    p_val  = results.p_value
    p_text = f"p = {p_val:.4f}" if p_val >= 0.0001 else "p < 0.0001"

    # ─────── HR using standard CoxPH ───────
    cox_df = pd.DataFrame({
        'T':     pd.concat([T1, T2], ignore_index=True),
        'E':     pd.concat([E1, E2], ignore_index=True),
        'group': [1] * len(T1) + [0] * len(T2)   # 1=Vaccinated, 0=Not vaccinated
    })
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col='T', event_col='E')
    hr    = np.exp(cph.params_['group'])
    hr_lo = np.exp(cph.confidence_intervals_['95% lower-bound']['group'])
    hr_hi = np.exp(cph.confidence_intervals_['95% upper-bound']['group'])
    hr_text = f"HR = {hr:.3f} (95% CI {hr_lo:.3f}–{hr_hi:.3f})"

    return p_text, hr_text


def GT_binary_group(df, text, var,scale,extra_space,k_min=0):

    had_vaccine = df[df[var] == 'Yes'].reset_index(drop=True)
    no_vaccine  = df[df[var] == 'No'].reset_index(drop=True)

    T1 = had_vaccine['Time'].copy()
    E1 = had_vaccine['Event']
    T2 = no_vaccine['Time'].copy()
    E2 = no_vaccine['Event']

    # ─────── handle Time=0 ───────
    eps = 1e-6
    T1[T1 == 0] = eps
    T2[T2 == 0] = eps

    # ─────── create both subplots from the GridSpec ───────
    fig     = plt.figure(figsize=(6, 6))
    gs      = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[4, 1], hspace=0.05)
    ax_km   = fig.add_subplot(gs[0])    # KM curve
    ax_risk = fig.add_subplot(gs[1])    # at-risk table

    ax_km.set_title("Uncorrected " + str(text))

    # ─────── truncating plot ───────
    max_time = scale[-1] + extra_space

    kmf1_fit, tcut1 = plot_km_truncated_by_n_at_risk(
        ax=ax_km, T=T1, E=E1, label="Vaccinated",
        color="royalblue", k_min_at_risk=k_min, max_time=max_time,
        show_censors=True, censor_marker="+", censor_ms=5, censor_mew=1.2, linewidth=2.5)

    kmf2_fit, tcut2 = plot_km_truncated_by_n_at_risk(
        ax=ax_km, T=T2, E=E2, label="Not vaccinated",
        color="Red", k_min_at_risk=k_min, max_time=max_time,
        show_censors=True, censor_marker="+", censor_ms=5, censor_mew=1.2, linewidth=2.5)

    add_at_risk_counts(kmf1_fit, kmf2_fit, ax=ax_km, fontsize=9,
                       rows_to_show=['At risk'], xticks=scale)

    p_text, hr_text = naive_km_metric(T1, T2, E1, E2)

    # ─────── display p-value and HR together in one box ───────
    combined_text = f"{p_text}\n{hr_text}"
    ax_km.text(0.02, 0.15, combined_text, transform=ax_km.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='grey', alpha=0.8))

    # ─────── legends and captions ───────
    ax_km.set_xlabel("Time (months)", fontsize=13)
    ax_km.set_ylabel("Survival probability", fontsize=13)
    ax_km.tick_params(axis='both', labelsize=13.5)
    ax_km.set_xlim(0, max_time)
    #ax_km.set_xticks([0, 12, 24, 36, 48, 60])
    ax_km.set_xticks(scale)
    ax_km.set_ylim(0.0, 1.05)
    #ax_km.legend(loc="upper right", frameon=False)
    ax_km.legend(loc="lower right", frameon=False)
    ax_risk.axis('off')

    plt.tight_layout()
    out_name = f"KM_{text}_{var}.png"
    plt.savefig(out_name, dpi=600, bbox_inches='tight')
    print(f"Saved: {out_name}")
    plt.show()


def plot_km_truncated_by_n_at_risk(ax, T, E, label, color, k_min_at_risk=5, max_time=60,
                                   show_censors=True, censor_marker="+", censor_ms=5,
                                   censor_mew=1.2, linewidth=2):
    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=E, label=label)

    # ─────── find truncation cutoff ───────
    et = kmf.event_table.copy()
    et = et[et.index <= max_time]

    if k_min_at_risk == 0:
        t_cut = float(et.index.max())
    else:
        ok = et["at_risk"] >= k_min_at_risk
        t_cut = float(et.index[ok].max()) if ok.any() else 0.0

    # ─────── build survival function up to max_time ───────
    sf = kmf.survival_function_[label].copy()
    sf = sf[sf.index <= max_time]

    # ─────── truncate: keep only points up to t_cut ───────
    sf_trunc = sf[sf.index <= t_cut].copy()

    # ─────── extend flat tail to last observed time (R survfit style) ───────
    # ax.step(where="post") needs a point AFTER the last event to draw the
    # flat horizontal line. We extend to last observed patient time so the
    # tail is clearly visible — not just a stub.
    last_obs = float(T[T <= max_time].max())
    end = min(last_obs, max_time)
    if end > sf_trunc.index[-1]:
        anchor = pd.Series([sf_trunc.iloc[-1]], index=[end], name=label)
        sf_trunc = pd.concat([sf_trunc, anchor])

    # ─────── draw step curve ───────
    ax.step(sf_trunc.index.values, sf_trunc.values, where="post",
            label=label, linewidth=linewidth, color=color)

    # ─────── censor marks up to t_cut only ───────
    # Deduplicate marks at t_cut to avoid dense cluster artifact
    if show_censors:
        cens = kmf.event_table["censored"]
        cens_times = cens.index[(cens > 0) & (cens.index <= t_cut) & (cens.index <= max_time)]
        if len(cens_times) > 0:
            y = kmf.predict(cens_times)
            ax.plot(cens_times, y, linestyle="None", marker=censor_marker,
                    markersize=censor_ms, markeredgewidth=censor_mew, color=color)

    return kmf, t_cut


'''------------------------------'''
#       Timing
'''------------------------------'''

def calculate_pairwise_hr(df_sub,group_col):

    cph = CoxPHFitter()
    
    # Pre vs No
    df_pre_no = df_sub[df_sub[group_col].isin(['Pre', 'No'])].copy()
    df_pre_no['group'] = (df_pre_no[group_col] == 'Pre').astype(int)
    if len(df_pre_no) > 0 and df_pre_no['Event'].sum() > 0:
        cph.fit(df_pre_no[['Time', 'Event', 'group']], duration_col='Time', event_col='Event')
        ci = cph.confidence_intervals_.loc['group']  # FIX: Use .loc to access the row
        hr_pre_no = {
            'HR': float(cph.hazard_ratios_['group']),
            'CI_low': float(np.exp(ci['95% lower-bound'])),
            'CI_high': float(np.exp(ci['95% upper-bound'])),
            'p_value': float(cph.summary.loc['group', 'p'])
        }
    else:
        hr_pre_no = {'HR': np.nan, 'CI_low': np.nan, 'CI_high': np.nan, 'p_value': np.nan}
    
    # Post vs No
    df_post_no = df_sub[df_sub[group_col].isin(['Post', 'No'])].copy()
    df_post_no['group'] = (df_post_no[group_col] == 'Post').astype(int)
    if len(df_post_no) > 0 and df_post_no['Event'].sum() > 0:
        cph.fit(df_post_no[['Time', 'Event', 'group']], duration_col='Time', event_col='Event')
        ci = cph.confidence_intervals_.loc['group']
        hr_post_no = {
            'HR': float(cph.hazard_ratios_['group']),
            'CI_low': float(np.exp(ci['95% lower-bound'])),
            'CI_high': float(np.exp(ci['95% upper-bound'])),
            'p_value': float(cph.summary.loc['group', 'p'])
        }
    else:
        hr_post_no = {'HR': np.nan, 'CI_low': np.nan, 'CI_high': np.nan, 'p_value': np.nan}
    
    # Pre vs Post
    df_pre_post = df_sub[df_sub[group_col].isin(['Pre', 'Post'])].copy()
    df_pre_post['group'] = (df_pre_post[group_col] == 'Pre').astype(int)
    if len(df_pre_post) > 0 and df_pre_post['Event'].sum() > 0:
        cph.fit(df_pre_post[['Time', 'Event', 'group']], duration_col='Time', event_col='Event')
        ci = cph.confidence_intervals_.loc['group']
        hr_pre_post = {
            'HR': float(cph.hazard_ratios_['group']),
            'CI_low': float(np.exp(ci['95% lower-bound'])),
            'CI_high': float(np.exp(ci['95% upper-bound'])),
            'p_value': float(cph.summary.loc['group', 'p'])
        }
    else:
        hr_pre_post = {'HR': np.nan, 'CI_low': np.nan, 'CI_high': np.nan, 'p_value': np.nan}
    
    return {
        'Pre_vs_No': hr_pre_no,
        'Post_vs_No': hr_post_no,
        'Pre_vs_Post': hr_pre_post
    }

def plot_vax_time(df,text,var,k_min):
    
    pre_tx =df[df[var]=='Pre'].reset_index(drop=True)
    post_tx =df[df[var]=='Post'].reset_index(drop=True)
    no_vaccine =df[df[var]=='No'].reset_index(drop=True)

    T1 = pre_tx['Time'] # time
    E1 = pre_tx['Event']
    
    T2 = post_tx['Time'] # time
    E2 = post_tx['Event']
    
    T3 = no_vaccine['Time'] # time
    E3 = no_vaccine['Event']
    
    # ─────── handle Time=0 ───────
    eps = 1e-6
    T1[T1 == 0] = eps
    T2[T2 == 0] = eps
    T3[T3 == 0] = eps

    # ─────── create both subplots from the GridSpec ───────
    fig = plt.figure(figsize=(6,6))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[4, 1])  # More space for top plot
    ax_km = fig.add_subplot(gs[0])   # KM curve
    #ax_risk = fig.add_subplot(gs[1]) # at-risk table
    
    ax_km.set_title("Uncorrected " + str(text))
    
    
    # ─────── truncating plot ───────
    max_time = 60 
    kmf1_fit, tcut1 = plot_km_truncated_by_n_at_risk(ax=ax_km, T=T1, E=E1, label="Pre_SABR", color="royalblue", k_min_at_risk=k_min, max_time=max_time,
                                                 show_censors=True, censor_marker="+", censor_ms=5, censor_mew=1.2,linewidth=2.5)
    kmf2_fit, tcut2 = plot_km_truncated_by_n_at_risk(ax=ax_km, T=T2, E=E2, label="Post_SABR", color="green",k_min_at_risk=k_min, max_time=max_time,
                                                 show_censors=True, censor_marker="+", censor_ms=5, censor_mew=1.2,linewidth=2.5)
    
    kmf3_fit, tcut3 = plot_km_truncated_by_n_at_risk(ax=ax_km, T=T3, E=E3, label="Not vaccinated", color="red",k_min_at_risk=4, max_time=max_time,
                                                 show_censors=True, censor_marker="+", censor_ms=5, censor_mew=1.2,linewidth=2.5)
    
    add_at_risk_counts(kmf1_fit, kmf2_fit,kmf3_fit, ax=ax_km, fontsize=9,rows_to_show=['At risk'], xticks=[0, 12, 24, 36, 48, 60])
    #ax_risk.axis('off')
    
    # ─────── overall log-rank ───────
    
    T = np.concatenate([T1, T2, T3])
    E = np.concatenate([E1, E2, E3])
    groups = (["Pre_SABR"] * len(T1) +["Post_SABR"] * len(T2) +["Not_vaccinated"] * len(T3))
    results = multivariate_logrank_test(T, groups, E)
    
    # ─────── Pairwise log-rank ───────
    # T1 is Pre, T2 is Post, T3 No vax
    lr_pre = logrank_test(T1, T3, event_observed_A=E1, event_observed_B=E3)
    lr_post = logrank_test(T2, T3, event_observed_A=E2, event_observed_B=E3)
    lr_pre_post = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    
    # ─────── Pairwise HR ───────
    hr_results = calculate_pairwise_hr(df, var)
    
    # ─────── legends and captions ───────
    ax_km.set_xlabel("Time (months)", fontsize=13)
    ax_km.set_ylabel("Survival probability", fontsize=13)
    ax_km.tick_params(axis='both', labelsize=13.5)
    ax_km.set_xlim(0, max_time)
    ax_km.set_xticks([0, 12, 24, 36, 48, 60])
    ax_km.set_ylim(0.0, 1.05)
    ax_km.legend(loc="lower left", frameon=False)
 
    
    plt.tight_layout()
    out_name = f"KM_{text}_{var}.png"
    plt.savefig(out_name, dpi=600, bbox_inches='tight')
    print(f"Saved: {out_name}")
    plt.show()

    # Dynamically get axis limits
    '''x_max = ax_km.get_xlim()[1]
    y_max = ax_km.get_ylim()[1]
    ax_km.text(x=0.2 * x_max, y=y_max - 0.3, s=f"$p$-value = {results.p_value:.4f}", fontsize='medium')
    ax_km.set_xticks([0, 12, 24, 36, 48, 60])
    ax_km.tick_params(axis="both", labelsize=14)
    
    ax_km.set_xlabel("Timeline (Months)")
    ax_km.set_ylabel("Survival Probability")
    add_at_risk_counts(kmf1_fit, kmf2_fit, kmf3_fit,ax=ax_km)
    ax_km.legend(loc='lower left', fontsize=12)
    plt.tight_layout()
    plt.show()'''
    
    log_rank = [lr_pre.p_value, lr_post.p_value, lr_pre_post.p_value]
    return log_rank,hr_results