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

from lifelines import KaplanMeierFitter
from sksurv.nonparametric import kaplan_meier_estimator
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines import CoxTimeVaryingFitter
import statsmodels.api as sm
from lifelines import CoxPHFitter

def build_timevarying_table(
    df,
    id_col="MRN",
    t0_col="TX_EndDate",
    time_col="Time",
    event_col="Event",
    vax_date_col="Event_Date",
    time_unit="months"
):
    d = df.copy()

    d[t0_col]       = pd.to_datetime(d[t0_col],       errors="coerce")
    d[vax_date_col] = pd.to_datetime(d[vax_date_col], errors="coerce")

    eps = 1e-6
    d[time_col] = d[time_col].replace(0, eps)

    d["t_vax"] = (d[vax_date_col] - d[t0_col]).dt.days / 30

    rows = []
    for _, r in d.iterrows():
        pid   = r[id_col]
        stop  = float(r[time_col])
        event = int(r[event_col])
        t_vax = r["t_vax"]

        # Case 1: no vaccine, or vaccine after follow-up ends
        if pd.isna(t_vax) or t_vax >= stop:
            rows.append({"id": pid, "start": 0.0, "stop": stop,
                         "event": event, "vax_tv": 0,
                         "vax_time": "No",        # ← new column
                         "interval": 1})
            continue

        # Case 2: vaccine before or at t0 → Pre-SABR
        if t_vax <= 0:
            rows.append({"id": pid, "start": 0.0, "stop": stop,
                         "event": event, "vax_tv": 1,
                         "vax_time": "Pre",        # ← new column
                         "interval": 1})
            continue

        # Case 3: vaccine during follow-up → Post-SABR, split into 2
        # Pre-vaccine interval: still unvaccinated → label "No"
        rows.append({"id": pid, "start": 0.0, "stop": float(t_vax),
                     "event": 0, "vax_tv": 0,
                     "vax_time": "No",             # ← pre-vaccine = No
                     "interval": 1})
        # Post-vaccine interval: now vaccinated post-SABR → label "Post"
        rows.append({"id": pid, "start": float(t_vax), "stop": stop,
                     "event": event, "vax_tv": 1,
                     "vax_time": "Post",           # ← post-vaccine = Post
                     "interval": 2})

    long_df = pd.DataFrame(rows)
    long_df = long_df.sort_values(["id", "start", "stop"]).reset_index(drop=True)
    long_df = long_df[long_df["stop"] > long_df["start"]]
    return long_df

'''==============================================='''
#           Immortal -Bias Correction
'''=============================================='''
def last_observed_time(long_df, group, max_time,var_tv):
    stops = []
    for pid in long_df["id"].unique():
        pat = long_df[long_df["id"] == pid]
        gi  = pat[pat[var_tv] == group]
        if not gi.empty:
            stops.append(float(gi["stop"].max()))
    return min(max(stops), max_time) if stops else max_time

def apply_tcut_with_flat_tail(times, surv, tcut, last_obs,max_time):
    
    """Keep points up to tcut, then extend flat to last_obs so the
    curve ends with a visible horizontal line matching R survfit style."""

    mask   = times <= tcut
    t_plot = times[mask]
    s_plot = surv[mask].astype(float)

    if len(t_plot) == 0:
        return times, surv.astype(float)

    # ─────── extend flat line to last observed time ───────
    # This makes the flat tail clearly visible (not just a stub)
    end = min(last_obs, max_time)
    if end > t_plot[-1]:
        t_plot = np.append(t_plot, end)
        s_plot = np.append(s_plot, s_plot[-1])

    return t_plot, s_plot
    
def get_censor_marks(long_df, group, times, surv_plot, tcut, last_obs,var_tv):
    cens_times = []

    for pid in long_df["id"].unique():
        pat = long_df[long_df["id"] == pid]

        # ─────── get all intervals belonging to this group ───────
        group_intervals = pat[pat[var_tv] == group]
        if group_intervals.empty:
            continue

        # ─────── the last interval in this group for this patient ───────
        last_interval = group_intervals.iloc[-1]
        interval_stop  = float(last_interval["stop"])
        interval_event = int(last_interval["event"])

        # ─────── censored in this group means: event=0 in last group interval ───────
        # IMPORTANT: do NOT use pat["event"].max() — that would wrongly exclude
        # Case 3 patients who were unvaccinated (group 0, event=0) but later
        # switched to vaccinated and died (group 1, event=1). Those patients
        # ARE censored out of group 0 and deserve a censor mark there.
        if interval_event == 1:
            continue

        # ─────── show at actual stop time (on flat tail) up to last_obs ───────
        # Do NOT clamp to tcut — patients censored after tcut should show
        # at their real times on the flat horizontal tail, just like R survfit.
        if interval_stop <= last_obs:
            cens_times.append(interval_stop)

    # ─────── look up survival value at each censor time ───────
    cens_x = []
    cens_y = []
    for ct in cens_times:
        idx = np.searchsorted(times, ct, side='right') - 1
        idx = np.clip(idx, 0, len(surv_plot) - 1)
        sv  = surv_plot[idx]
        if not np.isnan(sv):
            cens_x.append(ct)
            cens_y.append(sv)

    return cens_x, cens_y

def ctv_metrics(long_df,var_tv):
    ctvf = CoxTimeVaryingFitter()
    ctvf.fit(long_df, id_col='id', start_col='start', stop_col='stop',
             event_col='event', formula=var_tv)

    hr    = np.exp(float(ctvf.params_[var_tv]))
    hr_lo = np.exp(float(ctvf.confidence_intervals_['95% lower-bound'][var_tv]))
    hr_hi = np.exp(float(ctvf.confidence_intervals_['95% upper-bound'][var_tv]))
    p_val = float(ctvf.summary['p'][var_tv])

    p_text        = f"p = {p_val:.4f}" if p_val >= 0.0001 else "p < 0.0001"
    hr_text       = f"HR = {hr:.3f} (95% CI {hr_lo:.3f}-{hr_hi:.3f})"
    combined_text = f"{p_text}\n{hr_text}"    
    
    return combined_text

def find_tcut(times, at_risks, k_min):
    if k_min == 0:
        return times[-1]   # no truncation
    # ─────── skip t=0 (index 0) — that's the artificial starting point ───────
    ok = at_risks[1:] >= k_min   # check from index 1 onwards
    times_check = times[1:]      # corresponding times
    return float(times_check[ok].max()) if ok.any() else 0.0

def simon_makuch_curve(long_df,scale,extra_space,group,var_tv):
    # ─────── get all actual event times (deaths only, not censored) ───────
    event_times = sorted(long_df.loc[long_df["event"] == 1, "stop"].unique())
    max_time = scale[-1] + extra_space
    times    = [0.0]
    survival = [1.0]
    at_risks = [len(long_df["id"].unique())]  # all patients at risk at t=0

    for t in event_times:
        if t > max_time: # stop at max_time
            break

        n_at_risk = 0
        n_events  = 0

        for pid in long_df["id"].unique():
            pat = long_df[long_df["id"] == pid]

            # ─────── find patient's active interval at time t ───────
            active = pat[(pat["start"] < t) & (pat["stop"] >= t)]

            if active.empty:
                continue  # not at risk at time t

            row       = active.iloc[0]
            pat_group = int(row[var_tv])
            pat_stop  = float(row["stop"])
            pat_event = int(row["event"])

            if pat_group != group:
                continue  # wrong group -- skip

            # ─────── at risk in this group ───────
            n_at_risk += 1

            # ─────── event in this group at exactly t ───────
            if pat_stop == t and pat_event == 1:
                n_events += 1

        if n_at_risk == 0:
            continue  # no one at risk in this group at this time

        # ─────── KM formula ───────
        s = survival[-1] * (1 - n_events / n_at_risk)
        times.append(t)
        survival.append(s)
        at_risks.append(n_at_risk)  # track at-risk count at each time point

    # ─────── extend curve to last observed time in this group ───────
    # matches R survfit behavior — flat horizontal line after last event
    # find last stop time for patients in this group
    group_stops = []
    for pid in long_df["id"].unique():
        pat = long_df[long_df["id"] == pid]
        group_intervals = pat[pat[var_tv] == group]
        if not group_intervals.empty:
            group_stops.append(float(group_intervals["stop"].max()))

    if group_stops:
        last_observed = min(max(group_stops), max_time)
        if last_observed > times[-1]:
            times    = np.append(times,    last_observed)
            survival = np.append(survival, survival[-1])  # flat — no drop
            at_risks = np.append(at_risks, 0)             # 0 at risk after last event

    return np.array(times), np.array(survival), np.array(at_risks)


def plot_simon_makuch(timing,long_df, title, scale, extra_space,k_min0=0, k_min1=0):

    max_time = scale[-1] + extra_space
    if timing == []:
        # ─────── compute Simon-Makuch ───────
        t0, s0, r0 = simon_makuch_curve(long_df,scale,extra_space,group=0,var_tv='vax_tv')
        t1, s1, r1 = simon_makuch_curve(long_df,scale,extra_space,group=1,var_tv='vax_tv')
        sm_group_0=[t0, s0, r0]
        sm_group_1=[t1, s1, r1]
        # ─────── visual truncation───────
        tcut0 = find_tcut(t0, r0, k_min0)
        tcut1 = find_tcut(t1, r1, k_min1)
        print(f"DEBUG: tcut0={tcut0:.2f}, tcut1={tcut1:.2f}")
        
        # ─────── last observed time per group (capped at max_time) ───────
        last_obs0 = last_observed_time(long_df, group=0, max_time=max_time,var_tv='vax_tv')
        last_obs1 = last_observed_time(long_df, group=1, max_time=max_time,var_tv='vax_tv')
        t0_orig, s0_orig = t0.copy(), s0.copy()
        t1_orig, s1_orig = t1.copy(), s1.copy()
        # ─────── Mimic how R handles sudden drop ───────
        t0, s0_plot = apply_tcut_with_flat_tail(t0, s0, tcut0, last_obs0,max_time=max_time)
        t1, s1_plot = apply_tcut_with_flat_tail(t1, s1, tcut1, last_obs1,max_time=max_time)
        
        # ─────── plot starts───────
        combined_text = ctv_metrics(long_df,var_tv='vax_tv')
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        ax.set_xlim(0, max_time)
        ax.step(t0, s0_plot, where="post", color="Red",linewidth=2.5, label="Not Vaccinated")
        ax.step(t1, s1_plot, where="post", color="royalblue", linewidth=2.5, label="Vaccinated")
        
        # ─────── add censor marks───────
        cx0, cy0 = get_censor_marks(long_df, group=0, times=t0, surv_plot=s0_plot, tcut=tcut0, last_obs=last_obs0, var_tv='vax_tv') #
        cx1, cy1 = get_censor_marks(long_df, group=1, times=t1, surv_plot=s1_plot, tcut=tcut1, last_obs=last_obs1, var_tv='vax_tv')
        print(f"DEBUG: censor marks group0={len(cx0)}, group1={len(cx1)}")

        if cx0:
            ax.plot(cx0, cy0, linestyle="None", marker="+",markersize=5, markeredgewidth=1.2, color="Red")
        if cx1:
            ax.plot(cx1, cy1, linestyle="None", marker="+",markersize=5, markeredgewidth=1.2, color="royalblue")
        # ─────── print metrics in a box───────
        ax.text(0.02, 0.15, combined_text, transform=ax.transAxes, fontsize=10,verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',edgecolor='grey', alpha=0.8))
        # ─────── Other formatting ───────
        ax.set_title(title)
        ax.set_xlabel("Time (months)", fontsize=13)
        ax.set_ylabel("Survival probability", fontsize=13)
        ax.tick_params(axis='both', labelsize=13.5)
        ax.set_xticks(scale)
        ax.set_ylim(0.0, 1.02)
        ax.legend(loc="lower right", frameon=False)
        plt.tight_layout()
        out_name = f"SM_{title.replace(' ', '_')}.png"
        plt.savefig(out_name, dpi=600, bbox_inches='tight')
        print(f"Saved: {out_name}")
        plt.show()
        return fig,sm_group_0,sm_group_1
        
    elif timing == 'time':
        # ─────── pre-process ───────
        long_df['vax_time'] = long_df['vax_time'].replace('No',0)
        long_df['vax_time'] = long_df['vax_time'].replace('Pre',1)
        long_df['vax_time'] = long_df['vax_time'].replace('Post',2)
        # ─────── compute Simon-Makuch ───────
        t_no,s_no,r_no = simon_makuch_curve(long_df,scale,extra_space,group=0,var_tv='vax_time')
        t_pre,s_pre,r_pre = simon_makuch_curve(long_df,scale,extra_space,group=1,var_tv='vax_time')
        t_post,s_post,r_post = simon_makuch_curve(long_df,scale,extra_space,group=2,var_tv='vax_time')
        sm_group_no = [t_no,s_no,r_no]
        sm_group_pre = [t_pre,s_pre,r_pre]
        sm_group_post = [t_post,s_post,r_post] 
        
        # ─────── visual truncation───────
        tcut_no = find_tcut(t_no,r_no,k_min0)
        tcut_pre = find_tcut(t_pre,r_pre,k_min1)
        tcut_post = find_tcut(t_post,r_post,k_min1)
        print(f"DEBUG: tcut_no={tcut_no:.2f}, tcut_pre={tcut_pre:.2f}, tcut_post={tcut_post:.2f}")
        
        # ─────── last observed time per group (capped at max_time) ───────
        last_obs_no = last_observed_time(long_df, group=0, max_time=max_time,var_tv='vax_time')
        last_obs_pre = last_observed_time(long_df, group=1, max_time=max_time,var_tv='vax_time')
        last_obs_post = last_observed_time(long_df, group=2, max_time=max_time,var_tv='vax_time')
        t_no_orig, s_no_orig = t_no.copy(), s_no.copy()
        t_pre_orig, s_pre_orig = t_pre.copy(), s_pre.copy()
        t_post_orig, s_post_orig = t_post.copy(), s_post.copy()
        
        # ─────── Mimic how R handles sudden drop ───────
        t_no, s_no_plot = apply_tcut_with_flat_tail(t_no, s_no, tcut_no, last_obs_no, max_time=max_time)
        t_pre, s_pre_plot = apply_tcut_with_flat_tail(t_pre, s_pre, tcut_pre, last_obs_pre,max_time=max_time)
        t_post, s_post_plot = apply_tcut_with_flat_tail(t_post, s_post, tcut_post, last_obs_post,max_time=max_time)
        
        # ─────── plot starts───────
        combined_text = ctv_metrics(long_df,var_tv='vax_time')
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        ax.set_xlim(0, max_time)
        ax.step(t_no, s_no_plot, where="post", color="Red",linewidth=2.5, label="Never")
        ax.step(t_pre, s_pre_plot, where="post", color="royalblue", linewidth=2.5, label="Prior")
        ax.step(t_post, s_post_plot, where="post", color="green", linewidth=2.5, label="Post")
        
        # ─────── pair-wise metrics───────
        df_pre_never = long_df[long_df['vax_time'].isin([0, 1])].reset_index(drop=True)
        pre_never_text = ctv_metrics(df_pre_never,var_tv='vax_time')
        df_post_never = long_df[long_df['vax_time'].isin([0, 2])].reset_index(drop=True)
        post_never_text = ctv_metrics(df_post_never,var_tv='vax_time')
        df_pre_post = long_df[long_df['vax_time'].isin([1, 2])].reset_index(drop=True)
        pre_post_text = ctv_metrics(df_pre_post,var_tv='vax_time')
        
        pairwise_text = [pre_never_text,post_never_text,pre_post_text]
        
        
        # ─────── add censor marks───────
        cx_no, cy_no = get_censor_marks(long_df, group=0, times=t_no, surv_plot=s_no_plot, tcut=tcut_no, last_obs=last_obs_no, var_tv='vax_time') #
        cx_pre, cy_pre = get_censor_marks(long_df, group=1, times=t_pre, surv_plot=s_pre_plot, tcut=tcut_pre, last_obs=last_obs_pre, var_tv='vax_time')
        cx_post, cy_post = get_censor_marks(long_df, group=2, times=t_post, surv_plot=s_post_plot, tcut=tcut_post, last_obs=last_obs_post, var_tv='vax_time')
        print(f"DEBUG: censor marks group0={len(cx_no)}, group1={len(cx_pre)}, group2={len(cx_post)}")
        if cx_no:
            ax.plot(cx_no, cy_no, linestyle="None", marker="+",markersize=5, markeredgewidth=1.2, color="Red")
        if cx_pre:
            ax.plot(cx_pre, cy_pre, linestyle="None", marker="+",markersize=5, markeredgewidth=1.2, color="royalblue")
        if cx_post:
            ax.plot(cx_post, cy_post, linestyle="None", marker="+",markersize=5, markeredgewidth=1.2, color="green")
        # ─────── print metrics in a box───────
        ax.text(0.02, 0.15, combined_text, transform=ax.transAxes, fontsize=10,verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',edgecolor='grey', alpha=0.8))

        # ─────── Other formatting ───────
        ax.set_title(title)
        ax.set_xlabel("Time (months)", fontsize=13)
        ax.set_ylabel("Survival probability", fontsize=13)
        ax.tick_params(axis='both', labelsize=13.5)
        ax.set_xticks(scale)
        ax.set_ylim(0.0, 1.02)
        ax.legend(loc="lower right", frameon=False)
        plt.tight_layout()
        out_name = f"SM_{title.replace(' ', '_')}.png"
        plt.savefig(out_name, dpi=600, bbox_inches='tight')
        print(f"Saved: {out_name}")
        plt.show()
        return fig,sm_group_no,sm_group_pre,sm_group_post,pairwise_text
    
    else:
        
        # ─────── pre-process ───────
        long_df['new_group'] = long_df['new_group'].replace('sabr_no_vax',0)
        long_df['new_group'] = long_df['new_group'].replace('sabr_with_vax',1)
        long_df['new_group'] = long_df['new_group'].replace('isabr_no_vax',2)
        long_df['new_group'] = long_df['new_group'].replace('isabr_with_vax',3)

        # ─────── compute Simon-Makuch ───────

        t0, s0, r0 = simon_makuch_curve(long_df,scale,extra_space,group=0,var_tv='new_group')
        t1, s1, r1 = simon_makuch_curve(long_df,scale,extra_space,group=1,var_tv='new_group')
        t2, s2, r2 = simon_makuch_curve(long_df,scale,extra_space,group=2,var_tv='new_group')
        t3, s3, r3 = simon_makuch_curve(long_df,scale,extra_space,group=3,var_tv='new_group')
        
        sm_group_0 = [t0, s0, r0]
        sm_group_1 = [t1, s1, r1]
        sm_group_2 = [t2, s2, r2]
        sm_group_3 = [t3, s3, r3]
        
        # ─────── visual truncation───────
        tcut0 = find_tcut(t0, r0, k_min0)
        tcut1 = find_tcut(t1, r1, k_min1) # yes vax
        tcut2 = find_tcut(t2, r2, k_min0)
        tcut3 = find_tcut(t3, r3, k_min1) # yes vax
        
        #print(f"DEBUG: tcut0={tcut0:.2f}, tcut1={tcut1:.2f}")
        
        # ─────── last observed time per group (capped at max_time) ───────
        last_obs0 = last_observed_time(long_df, group=0, max_time=max_time,var_tv='new_group')
        last_obs1 = last_observed_time(long_df, group=1, max_time=max_time,var_tv='new_group')
        last_obs2 = last_observed_time(long_df, group=2, max_time=max_time,var_tv='new_group')
        last_obs3 = last_observed_time(long_df, group=3, max_time=max_time,var_tv='new_group')
        
        t0_orig, s0_orig = t0.copy(), s0.copy()
        t1_orig, s1_orig = t1.copy(), s1.copy()
        t2_orig, s2_orig = t2.copy(), s2.copy()
        t3_orig, s3_orig = t3.copy(), s3.copy()
        
        # ─────── Mimic how R handles sudden drop ───────
        t0, s0_plot = apply_tcut_with_flat_tail(t0, s0, tcut0, last_obs0,max_time=max_time)
        t1, s1_plot = apply_tcut_with_flat_tail(t1, s1, tcut1, last_obs1,max_time=max_time)
        t2, s2_plot = apply_tcut_with_flat_tail(t2, s2, tcut2, last_obs2,max_time=max_time)
        t3, s3_plot = apply_tcut_with_flat_tail(t3, s3, tcut3, last_obs3,max_time=max_time)
        
        # ─────── plot starts───────
        combined_text = ctv_metrics(long_df,var_tv='new_group')
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        ax.set_xlim(0, max_time)
        ax.step(t0, s0_plot, where="post", color="royalblue",linewidth=2.5, linestyle="dotted", label="sabr_no_vax",)
        ax.step(t1, s1_plot, where="post", color="royalblue", linewidth=2.5, linestyle="solid",label="sabr_with_vax")
        ax.step(t2, s2_plot, where="post", color="green", linewidth=2.5, linestyle="dotted",label="isabr_no_vax")
        ax.step(t3, s3_plot, where="post", color="green", linewidth=2.5, linestyle="solid",label="isabr_with_vax")
        
        # ─────── print metrics in a box───────
        ax.text(0.02, 0.15, combined_text, transform=ax.transAxes, fontsize=10,verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',edgecolor='grey', alpha=0.8))
        # ─────── Other formatting ───────
        ax.set_title(title)
        ax.set_xlabel("Time (months)", fontsize=13)
        ax.set_ylabel("Survival probability", fontsize=13)
        ax.tick_params(axis='both', labelsize=13.5)
        ax.set_xticks(scale)
        ax.set_ylim(0.0, 1.02)
        ax.legend(loc="lower right", frameon=False)
        plt.tight_layout()
        out_name = f"SM_{title.replace(' ', '_')}.png"
        plt.savefig(out_name, dpi=600, bbox_inches='tight')
        print(f"Saved: {out_name}")
        plt.show()
        return fig,sm_group_0,sm_group_1,sm_group_2,sm_group_3







'''==============================================='''
#           Confounder Correction
'''=============================================='''

def tv_cox_adjust(long_df, baseline_df,pen,covariates=None):
    covariates = covariates or []
    base = baseline_df[["MRN"] + covariates].copy()
    df_tv = long_df.merge(base, left_on="id", right_on="MRN", how="left")

    if covariates:
        cat_cols = [c for c in covariates if df_tv[c].dtype.name in ("category", "object")]
        if cat_cols:
            df_tv = pd.get_dummies(df_tv, columns=cat_cols, drop_first=True)

    drop_cols = {"MRN", "id", "start", "stop", "event","interval",'vax_time'}
    feature_cols = [c for c in df_tv.columns if c not in drop_cols]
    if "vax_tv" not in feature_cols:
        feature_cols = ["vax_tv"] + feature_cols

    ctv = CoxTimeVaryingFitter(penalizer=pen)
    ctv.fit(
        df_tv[["id","start","stop","event"] + feature_cols],
        id_col="id", start_col="start", stop_col="stop", event_col="event")

    summ = ctv.summary
    r = summ.loc["vax_tv"]

    HR  = float(np.exp(r["coef"]))
    lo  = float(np.exp(r["coef lower 95%"]))
    hi  = float(np.exp(r["coef upper 95%"]))
    p   = float(r["p"])

    out = {
        "HR": HR,
        "lo": lo,
        "hi": hi,
        "p": p,
        "HR_CI": f"{HR:.2f} ({lo:.2f}–{hi:.2f})",
        "p_fmt": f"{p:.3f}",
        "n_intervals": int(len(df_tv)),
        "n_patients": int(df_tv["id"].nunique()),
        "events": int(df_tv.groupby("id")["event"].max().sum()),  # events among patients
    }
    return out,summ


def run_correct_sel_bias(df_wide, covariates,pen):
 
    base_data = df_wide.copy()
    long_data = build_timevarying_table(base_data, id_col="MRN", t0_col="TX_EndDate",time_col="Time", event_col="Event", vax_date_col="Vaccine_Date")

    uni_metrics,summ_uni = tv_cox_adjust(long_data, base_data, pen, covariates=None)
    multi_metrics,summ_multi = tv_cox_adjust(long_data, base_data, pen,covariates=covariates)
    
    table_uni = pd.DataFrame([{"Method":"Unadjusted",  **uni_metrics}])
    table_adj = pd.DataFrame([{"Method":"Adjusted",  **multi_metrics}])
    
    return table_uni,table_adj,summ_multi



