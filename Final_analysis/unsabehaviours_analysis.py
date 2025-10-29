# O2_RQ2_UnsafeBehaviours_Analysis.py
# =============================================================================
# Academic-grade analysis for RQ2/O2:
#   "Does the AI-enabled app reduce unsafe driving behaviours (post vs pre)?"
#
# Inputs expected in ./recomputed_results:
#   - unsafe_events_pre.csv / unsafe_events_post.csv  (optional; EDA)
#   - trip_event_summary_pre.csv / ..._post.csv       (required)
#   - trip_event_summary_all.csv                       (optional; built if missing)
#
# Outputs written to ./analysis_outputs:
#   CSV: counts, per-type rates, driver aggregates, GLMs, Bayes summary
#   PNG: trip-level UBPK hist/box, spaghetti & forest plots
#   TXT: paired-tests summary, APA/JARS-style report, high-level summary
# =============================================================================

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional libs (script runs even if these are missing)
try:
    from scipy import stats
    SCIPY_OK = True
except Exception as e:
    print("[WARN] SciPy not available:", e)
    SCIPY_OK = False

try:
    import statsmodels.api as sm
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Poisson
    SM_OK = True
except Exception as e:
    print("[WARN] statsmodels not available:", e)
    SM_OK = False

try:
    import pymc as pm
    import arviz as az
    PYMC_OK = True
except Exception as e:
    print("[WARN] PyMC/ArviZ not available:", e)
    PYMC_OK = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------- Config (constants only) ----------------------------
# IN_DIR  = Path("recomputed_results")
IN_DIR = Path("../complete_dataset/computed_unsafe_behaviours_dataset")
OUT_DIR = Path("analysis_outputs")

FILES = {
    "events_pre":  IN_DIR/"unsafe_events_pre.csv",
    "events_post": IN_DIR/"unsafe_events_post.csv",
    "trip_pre":    IN_DIR/"trip_event_summary_pre.csv",
    "trip_post":   IN_DIR/"trip_event_summary_post.csv",
    "trip_all":    IN_DIR/"trip_event_summary_all.csv",
}

MIN_KM = 1e-6          # minimal distance to keep a trip
SHORT_TRIP_KM = 0.5    # sensitivity cutoff
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Bayesian settings (used only if PyMC is installed)
MCMC_CHAINS   = 2      # sequential chains on Windows
MCMC_DRAWS    = 1500
MCMC_TUNE     = 1000
MCMC_TARG_ACC = 0.9
MCMC_SEED     = RANDOM_SEED
BAYES_CORES   = 1      # <- critical for Windows (no multiprocessing)

# For GLM exposure column
MIN_EXPOSURE_KM = 0.01

# ---------------------------- Utilities ----------------------------
def read_csv_auto(p: Path) -> pd.DataFrame:
    for enc in ("utf-8","utf-8-sig","latin1"):
        try:
            return pd.read_csv(p, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
        except Exception:
            pass
    return pd.read_csv(p, low_memory=False)

def save_table(df: pd.DataFrame, name: str, out_dir: Path):
    path = out_dir/f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"[saved] {path}")

def save_txt(text: str, name: str, out_dir: Path):
    path = out_dir/f"{name}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[saved] {path}")

def save_fig(name: str, out_dir: Path, tight=True):
    path = out_dir/f"{name}.png"
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=220)
    print(f"[saved] {path}")

def norm_trip_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    # driver_id alias
    if "driver_id" not in out.columns and "driverProfileId" in out.columns:
        out["driver_id"] = out["driverProfileId"].astype(str)

    # >>> NEW: trip_id alias (fixes your KeyError) <<<
    if "trip_id" not in out.columns:
        if "id" in out.columns:           # Phase 2 trip summaries use 'id'
            out["trip_id"] = out["id"].astype(str)
        elif "TripId" in out.columns:     # just in case of other casings
            out["trip_id"] = out["TripId"].astype(str)
        else:
            # last-resort stable surrogate
            out["trip_id"] = np.arange(len(out)).astype(str)

    # numeric coercions
    for c in ["distance_km","speeding","harsh_brake","rapid_accel","swerve","total_events","ubpk"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["distance_km"] = out["distance_km"].fillna(0.0)
    out["total_events"] = out["total_events"].fillna(0.0)
    out["ubpk"] = out["ubpk"].replace([np.inf, -np.inf], np.nan)
    return out


def norm_events_df(df: pd.DataFrame, phase_label: str) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()
    if "driver_id" not in out.columns and "driverProfileId" in out.columns:
        out["driver_id"] = out["driverProfileId"]
    if "phase" not in out.columns:
        out["phase"] = phase_label
    return out

def basic_counts(tr: pd.DataFrame, label: str) -> dict:
    if tr.empty:
        return {"label": label, "drivers": 0, "trips": 0, "total_km": 0.0,
                "trips_with_any_event": 0, "pct_trips_with_any_event": 0.0,
                "total_events": 0}
    drivers = tr["driver_id"].nunique() if "driver_id" in tr.columns else np.nan
    trips   = tr["trip_id"].nunique()   if "trip_id" in tr.columns   else len(tr)
    total_km = float(tr["distance_km"].sum())
    trips_with = int((tr["total_events"] > 0).sum()) if "total_events" in tr.columns else np.nan
    pct_with   = float(trips_with / max(1, trips))
    total_events = int(tr["total_events"].sum()) if "total_events" in tr.columns else np.nan
    return {"label": label, "drivers": drivers, "trips": trips,
            "total_km": total_km, "trips_with_any_event": trips_with,
            "pct_trips_with_any_event": pct_with, "total_events": total_events}

def driver_aggregate(tr: pd.DataFrame) -> pd.DataFrame:
    if tr.empty: return pd.DataFrame()
    g = tr.groupby(["driver_id","phase"], as_index=False).agg(
        trips=("trip_id","nunique"),
        total_km=("distance_km","sum"),
        total_events=("total_events","sum"),
        mean_ubpk=("ubpk","mean"),
        median_ubpk=("ubpk","median"),
    )
    g["rate_events_per_km"] = g["total_events"] / g["total_km"].clip(lower=MIN_KM)
    return g

def paired_driver_rates(drv: pd.DataFrame) -> pd.DataFrame:
    pre  = drv.loc[drv["phase"]=="pre",  ["driver_id","rate_events_per_km","total_km","trips"]].rename(
        columns={"rate_events_per_km":"rate_pre","total_km":"km_pre","trips":"trips_pre"})
    post = drv.loc[drv["phase"]=="post", ["driver_id","rate_events_per_km","total_km","trips"]].rename(
        columns={"rate_events_per_km":"rate_post","total_km":"km_post","trips":"trips_post"})
    m = pre.merge(post, on="driver_id", how="inner")
    m["delta"] = m["rate_post"] - m["rate_pre"]
    m["ratio"] = (m["rate_post"].replace(0, np.nan) / m["rate_pre"].replace(0, np.nan))
    return m

def prepare_trip_glm_df(tr: pd.DataFrame, y_col: str = "total_events") -> pd.DataFrame:
    df = tr.copy()
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
    df = df.loc[df["distance_km"] >= MIN_KM].copy()
    df["phase"] = df["phase"].astype(str)
    df["phase_bin"] = (df["phase"].str.lower() == "post").astype(int)
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce").fillna(0).clip(lower=0).astype(int)
    df["exposure_km"] = df["distance_km"].clip(lower=MIN_EXPOSURE_KM)
    if "driver_id" not in df.columns:
        if "driverProfileId" in df.columns:
            df["driver_id"] = df["driverProfileId"]
        elif "trip_id" in df.columns:
            df["driver_id"] = df["trip_id"]
        else:
            df["driver_id"] = "driver_0"
    df["driver_id"] = df["driver_id"].astype(str).replace({"nan": "missing", "None": "missing"})
    df["driver_group"] = pd.Categorical(df["driver_id"]).codes.astype(int)
    keep = np.isfinite(df["exposure_km"]) & np.isfinite(df[y_col]) & np.isfinite(df["phase_bin"])
    return df.loc[keep].copy()

def fit_glm_models(tr: pd.DataFrame, y_col="total_events"):
    if not SM_OK:
        return None, None, {}

    df = prepare_trip_glm_df(tr, y_col=y_col)
    if df.empty:
        return None, None, {}

    if "exposure_km" not in df.columns:
        df["exposure_km"] = np.exp(df["log_km"]) if "log_km" in df.columns else pd.to_numeric(df["distance_km"], errors="coerce")
    df["exposure_km"] = pd.to_numeric(df["exposure_km"], errors="coerce").clip(lower=MIN_EXPOSURE_KM)

    keep = (
        np.isfinite(df[y_col]) &
        np.isfinite(df["exposure_km"]) &
        (df["exposure_km"] > 0) &
        np.isfinite(df["phase_bin"])
    )
    df = df.loc[keep].copy()
    if df.empty:
        out = pd.DataFrame([{"model": "Poisson+exposure (skipped: no finite rows)", "y": y_col, "coef": np.nan, "se": np.nan, "RR_post_vs_pre": np.nan, "CI95_lo": np.nan, "CI95_hi": np.nan}])
        return out, df, {}

    if df[y_col].sum() == 0:
        out = pd.DataFrame([{"model": "Poisson+exposure (skipped: all zeros)", "y": y_col, "coef": np.nan, "se": np.nan, "RR_post_vs_pre": 1.0, "CI95_lo": np.nan, "CI95_hi": np.nan}])
        return out, df, {}

    groups_vec = df["driver_group"].to_numpy()
    n_groups = int(np.unique(groups_vec).size)
    use_cluster = n_groups >= 2

    X = sm.add_constant(df[["phase_bin"]], has_constant="add")
    exposure = df["exposure_km"].to_numpy()

    poisson = sm.GLM(df[y_col], X, family=sm.families.Poisson(), exposure=exposure)
    try:
        if use_cluster:
            pois_res = poisson.fit(cov_type="cluster", cov_kwds={"groups": groups_vec})
        else:
            pois_res = poisson.fit(cov_type="HC1")
    except Exception:
        exposure2 = exposure + 1e-9
        pois_res = sm.GLM(df[y_col], X, family=sm.families.Poisson(), exposure=exposure2).fit(cov_type=("cluster" if use_cluster else "HC1"), cov_kwds=({"groups": groups_vec} if use_cluster else None))

    try:
        nb = sm.GLM(df[y_col], X, family=sm.families.NegativeBinomial(alpha=1.0), exposure=exposure)
        nb_res = nb.fit(cov_type=("cluster" if use_cluster else "HC1"), cov_kwds=({"groups": groups_vec} if use_cluster else None))
    except Exception:
        nb_res = None

    gee_res = None
    try:
        gcounts = df.groupby("driver_group")[y_col].size()
        valid_geegroups = gcounts[gcounts >= 2].index
        gee_df = df[df["driver_group"].isin(valid_geegroups)].copy()
        enough_groups = gee_df["driver_group"].nunique() >= 2
        if enough_groups:
            gee_offset = np.log(gee_df["exposure_km"].to_numpy().clip(min=MIN_EXPOSURE_KM))
            Xg = sm.add_constant(gee_df[["phase_bin"]], has_constant="add")
            yg = gee_df[y_col].to_numpy()
            gg = gee_df["driver_group"].to_numpy()
            gee = GEE(endog=yg, exog=Xg, groups=gg, family=Poisson(), offset=gee_offset)
            gee_res = gee.fit()
    except Exception:
        gee_res = None

    def summarize(res, name):
        if res is None:
            return {"model": name, "y": y_col, "coef": np.nan, "se": np.nan, "RR_post_vs_pre": np.nan, "CI95_lo": np.nan, "CI95_hi": np.nan}
        coef = float(res.params.get("phase_bin", np.nan))
        se = float(res.bse["phase_bin"]) if hasattr(res, "bse") and "phase_bin" in res.bse else np.nan
        rr = float(np.exp(coef)) if np.isfinite(coef) else np.nan
        lo, hi = (np.exp(coef - 1.96*se), np.exp(coef + 1.96*se)) if np.isfinite(se) else (np.nan, np.nan)
        return {"model": name, "y": y_col, "coef": coef, "se": se, "RR_post_vs_pre": rr, "CI95_lo": lo, "CI95_hi": hi}

    rows = [summarize(pois_res, "Poisson+exposure+" + ("cluster" if use_cluster else "HC1"))]
    rows.append(summarize(nb_res, "NegBin+exposure+" + ("cluster" if use_cluster else "HC1")))
    if gee_res is not None:
        rows.append(summarize(gee_res, "GEE-Poisson+offset(log exposure)"))

    out = pd.DataFrame(rows)
    return out, df, {"poisson": pois_res, "nb": nb_res, "gee": gee_res}

def plot_forest_rr(df: pd.DataFrame, title: str, fname: str, out_dir: Path):
    if df is None or df.empty: return
    plt.figure(figsize=(7, max(3, 0.5*len(df)+1)))
    y = np.arange(len(df))[::-1]
    rr = df["RR_post_vs_pre"].values
    lo = df["CI95_lo"].values
    hi = df["CI95_hi"].values
    labels = [f"{row['model']} [{row['y']}]" for _,row in df.iterrows()]
    for yi, l, h in zip(y, lo, hi):
        plt.hlines(yi, l, h)
    plt.scatter(rr, y)
    plt.axvline(1.0, linestyle="--", alpha=0.6)
    plt.yticks(y, labels)
    plt.xlabel("Rate Ratio (Post vs Pre)")
    plt.title(title)
    save_fig(fname, out_dir)
    plt.close()

def per_type_rates(tr: pd.DataFrame, label: str) -> pd.DataFrame:
    if tr.empty: return pd.DataFrame()
    types = ["speeding","harsh_brake","rapid_accel","swerve","total_events"]
    rows = []
    for t in types:
        if t not in tr.columns: continue
        count = float(tr[t].sum())
        km = float(tr["distance_km"].sum())
        rate_per100 = 100.0 * count / max(MIN_KM, km)
        rows.append({"phase": label, "type": t, "count": int(count), "distance_km": km, "rate_per100km": rate_per100})
    return pd.DataFrame(rows)

def apa_decimal(x, k=3):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))): return "NA"
    try: return f"{float(x):.{k}f}"
    except Exception: return "NA"

def _pick_poisson_row(df):
    if df is None or df.empty: return None
    pref = df["model"].astype(str)
    m1 = df[pref.str.startswith("Poisson+exposure+robust")]
    if len(m1): return m1.iloc[0]
    m2 = df[pref.str.contains("Poisson", case=False, regex=False)]
    return m2.iloc[0] if len(m2) else None

def fit_bayesian_hierarchical_poisson(tr: pd.DataFrame, y_col="total_events"):
    if not PYMC_OK:
        print("[WARN] PyMC/ArviZ not available; skipping Bayesian model.")
        return None, None
    df = prepare_trip_glm_df(tr, y_col=y_col)
    if df.empty or df[y_col].sum() == 0:
        print("[WARN] Bayesian model skipped (empty data or all-zero outcome).")
        return None, None

    y = df[y_col].astype("int64").to_numpy()
    exposure = df["exposure_km"].to_numpy()
    phase = df["phase_bin"].astype("int64").to_numpy()
    driver_idx = df["driver_group"].astype("int64").to_numpy()
    n_drivers = int(df["driver_group"].nunique())

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=5.0)
        beta  = pm.Normal("beta",  mu=0.0, sigma=5.0)
        sigma_u = pm.HalfNormal("sigma_u", sigma=1.0)
        u_driver = pm.Normal("u_driver", mu=0.0, sigma=sigma_u, shape=n_drivers)
        mu_log = np.log(exposure) + alpha + beta * phase + u_driver[driver_idx]
        lam = pm.Deterministic("lambda", pm.math.exp(mu_log))
        _ = pm.Poisson("y_obs", mu=lam, observed=y)
        rr = pm.Deterministic("RR_post_vs_pre", pm.math.exp(beta))
        idata = pm.sample(draws=MCMC_DRAWS, tune=MCMC_TUNE, chains=MCMC_CHAINS, cores=BAYES_CORES, target_accept=MCMC_TARG_ACC, random_seed=MCMC_SEED, progressbar=True, return_inferencedata=True)

    post_rr = np.asarray(idata.posterior["RR_post_vs_pre"]).reshape(-1)
    rr_med = float(np.median(post_rr))
    rr_lo, rr_hi = np.quantile(post_rr, [0.025, 0.975])
    sigma_samples = np.asarray(idata.posterior["sigma_u"]).reshape(-1)
    sig_med = float(np.median(sigma_samples))
    sig_lo, sig_hi = np.quantile(sigma_samples, [0.025, 0.975])
    summary = {"rr_med": rr_med, "rr_lo": float(rr_lo), "rr_hi": float(rr_hi), "sigma_u_med": sig_med, "sigma_u_lo": float(sig_lo), "sigma_u_hi": float(sig_hi), "n_trips": int(len(df)), "n_drivers": int(n_drivers)}
    return {"idata": idata}, summary

# ======================= temporal & dynamics helpers =======================
def _parse_ts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ts" in out.columns: return out
    t_ms = pd.to_numeric(out.get("start_time"), errors="coerce")
    ts1 = pd.to_datetime(t_ms, unit="ms", errors="coerce")
    ts2 = pd.to_datetime(out.get("start_date"), errors="coerce")
    out["ts"] = ts1
    out.loc[out["ts"].isna(), "ts"] = ts2
    return out.dropna(subset=["ts"]).copy()

def _add_period_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    w = out["ts"].dt.to_period("W-MON")
    out["week_start"] = w.dt.to_timestamp(how="start")
    out["year_week"] = out["week_start"].dt.strftime("%Y-W%U")
    m = out["ts"].dt.to_period("M")
    out["month_start"] = m.dt.to_timestamp(how="start")
    out["year_month"] = out["month_start"].dt.strftime("%Y-%m")
    out["dow_name"] = out["ts"].dt.day_name()
    return out

def _agg_period(df: pd.DataFrame, period_col: str) -> pd.DataFrame:
    keep_cols = [c for c in ["speeding","harsh_brake","rapid_accel","swerve","total_events"] if c in df.columns]
    grp = (df.groupby(["driver_id","phase",period_col], as_index=False).agg(trips=("trip_id","nunique"), distance_km=("distance_km","sum"), **{f"{c}_sum": (c,"sum") for c in keep_cols}, mean_ubpk=("ubpk","mean")))
    grp["rate_events_per_km"] = grp["total_events_sum"] / grp["distance_km"].clip(lower=MIN_KM)
    for c in keep_cols:
        grp[f"rate_{c}_per_km"] = grp[f"{c}_sum"] / grp["distance_km"].clip(lower=MIN_KM)
    return grp

def _overall_period_rates(df: pd.DataFrame, period_col: str) -> pd.DataFrame:
    keep_cols = [c for c in ["speeding","harsh_brake","rapid_accel","swerve","total_events"] if c in df.columns]
    grp = (df.groupby(["phase",period_col], as_index=False).agg(distance_km=("distance_km","sum"), **{f"{c}_sum": (c,"sum") for c in keep_cols}))
    grp["rate_events_per_km"] = grp["total_events_sum"] / grp["distance_km"].clip(lower=MIN_KM)
    for c in keep_cols:
        grp[f"rate_{c}_per_km"] = grp[f"{c}_sum"] / grp["distance_km"].clip(lower=MIN_KM)
    return grp

def _driver_weekly_slopes(weekly: pd.DataFrame) -> pd.DataFrame:
    if weekly.empty: return pd.DataFrame()
    df = weekly.sort_values(["driver_id","phase","week_start"]).copy()
    df["week_idx"] = df.groupby(["driver_id","phase"]).cumcount() + 1
    rows = []
    for (d,p), g in df.groupby(["driver_id","phase"]):
        g = g.dropna(subset=["rate_events_per_km"])
        if len(g) < 3: continue
        x = g["week_idx"].to_numpy().astype(float)
        y = g["rate_events_per_km"].to_numpy().astype(float)
        if np.isfinite(x).all() and np.isfinite(y).all():
            slope = np.polyfit(x, y, 1)[0]
            rows.append({"driver_id": d, "phase": p, "weeks": len(g), "slope_per_week": slope})
    return pd.DataFrame(rows)


def _driver_monthly_slopes(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Compute driver-level linear trend slopes (events/km per *month index*) within each phase.
    Requires >=3 monthly points per (driver, phase).
    """
    if monthly.empty:
        return pd.DataFrame()
    df = monthly.sort_values(["driver_id", "phase", "month_start"]).copy()
    # month_idx is 1..n within each (driver, phase)
    df["month_idx"] = df.groupby(["driver_id", "phase"]).cumcount() + 1
    rows = []
    for (d, p), g in df.groupby(["driver_id", "phase"]):
        g = g.dropna(subset=["rate_events_per_km"])
        if len(g) < 3:
            continue
        x = g["month_idx"].to_numpy(dtype=float)
        y = g["rate_events_per_km"].to_numpy(dtype=float)
        if np.isfinite(x).all() and np.isfinite(y).all():
            slope = np.polyfit(x, y, 1)[0]  # events/km per month
            rows.append({"driver_id": d, "phase": p, "months": len(g), "slope_per_month": slope})
    return pd.DataFrame(rows)


def _paired_slope_tests(slope_df: pd.DataFrame,
                        slope_col: str,
                        label: str,
                        out_txt: str,
                        out_dir: Path):
    """
    Pivot (driver x phase) slopes, paired t-test on post-pre. Saves APA-style summary.
    """
    if slope_df.empty:
        save_txt(f"No data for {label} slope tests.", out_txt, out_dir)
        return
    piv = slope_df.pivot(index="driver_id", columns="phase", values=slope_col).dropna()
    if piv.empty or "pre" not in piv.columns or "post" not in piv.columns:
        save_txt(f"Insufficient paired data for {label} slope tests.", out_txt, out_dir)
        return
    lines = [f"n={len(piv)} drivers with {label} slopes in both phases"]
    if SCIPY_OK and len(piv) >= 2:
        tstat, tp = stats.ttest_rel(piv["post"], piv["pre"], nan_policy="omit")
        lines.append(f"Paired t-test on slopes (post - pre): t({len(piv)-1})={tstat:.3f}, p={tp:.4g}")
        try:
            wstat, wp = stats.wilcoxon(piv["post"], piv["pre"], zero_method="wilcox", alternative="two-sided")
            lines.append(f"Wilcoxon signed-rank: W={wstat}, p={wp:.4g}")
        except ValueError:
            pass
        # Effect size (Cohen's d for paired samples)
        diff = (piv["post"] - piv["pre"]).dropna()
        if diff.std(ddof=1) > 0:
            d = diff.mean() / diff.std(ddof=1)
            lines.append(f"Cohen's d (paired)={d:.3f}")
    save_txt("\n".join(lines), out_txt, out_dir)


def _overall_monthly_wls_test(monthly_overall: pd.DataFrame) -> str:
    """
    Weighted least squares of rate_events_per_km ~ month_idx within each phase,
    weights = distance_km (km-weighted trend). Returns a multi-line APA-style report.
    """
    if monthly_overall.empty or not SM_OK:
        return "Overall monthly WLS slope test skipped (missing data or statsmodels)."
    df = monthly_overall.sort_values(["phase", "month_start"]).copy()
    df["month_idx"] = df.groupby("phase").cumcount() + 1
    out_lines = []
    for ph, g in df.groupby("phase"):
        g = g.dropna(subset=["rate_events_per_km", "distance_km"])
        if len(g) < 3:
            out_lines.append(f"{ph}: insufficient months (<3) for WLS.")
            continue
        # WLS: rate ~ const + month_idx, weights = distance_km
        X = sm.add_constant(g["month_idx"].to_numpy(dtype=float), has_constant="add")
        y = g["rate_events_per_km"].to_numpy(dtype=float)
        w = g["distance_km"].to_numpy(dtype=float).clip(min=1e-9)
        try:
            res = sm.WLS(y, X, weights=w).fit()
            b1 = float(res.params[1])
            se1 = float(res.bse[1])
            tval = float(res.tvalues[1])
            pval = float(res.pvalues[1])
            lo, hi = b1 - 1.96*se1, b1 + 1.96*se1
            out_lines.append(
                f"{ph}: WLS slope={b1:.4f} (events/km per month), 95% CI [{lo:.4f}, {hi:.4f}], "
                f"t={tval:.3f}, p={pval:.4g}, months={len(g)}"
            )
        except Exception as e:
            out_lines.append(f"{ph}: WLS failed: {e}")
    return "\n".join(out_lines)



def _classify_responders(drv_pairs: pd.DataFrame, abs_eps: float = 0.01, rel_eps: float = 0.10) -> pd.DataFrame:
    m = drv_pairs.copy()
    m["rel_change"] = (m["rate_post"] - m["rate_pre"]) / m["rate_pre"].replace(0, np.nan)
    cond_improve = (m["delta"] <= -abs_eps) & (m["rel_change"].abs() >= rel_eps)
    cond_worsen  = (m["delta"] >=  abs_eps) & (m["rel_change"].abs() >= rel_eps)
    m["response_class"] = np.where(cond_improve, "improved", np.where(cond_worsen, "worsened", "no_change"))
    return m

def _post_bad_day_share(tr_all: pd.DataFrame) -> pd.DataFrame:
    df = tr_all.copy()
    pre_thr = (df[df["phase"]=="pre"].groupby("driver_id")["ubpk"].quantile(0.75).rename("pre_q75_ubpk"))
    post = df[df["phase"]=="post"][["driver_id","trip_id","ubpk"]].copy()
    post = post.merge(pre_thr, on="driver_id", how="left")
    post["bad_day"] = post["ubpk"] > post["pre_q75_ubpk"]
    share = (post.groupby("driver_id")["bad_day"].mean().rename("post_bad_day_share").reset_index())
    return share

def _dow_rates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep_cols = [c for c in ["speeding","harsh_brake","rapid_accel","swerve","total_events"] if c in df.columns]
    g1 = (df.groupby(["phase","dow_name"], as_index=False).agg(distance_km=("distance_km","sum"), **{f"{c}_sum": (c,"sum") for c in keep_cols}))
    g1["rate_events_per_km"] = g1["total_events_sum"] / g1["distance_km"].clip(lower=MIN_KM)
    g2 = (df.groupby(["driver_id","phase","dow_name"], as_index=False).agg(distance_km=("distance_km","sum"), **{f"{c}_sum": (c,"sum") for c in keep_cols}))
    g2["rate_events_per_km"] = g2["total_events_sum"] / g2["distance_km"].clip(lower=MIN_KM)
    return g1, g2

def _driver_type_delta_matrix(tr_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    types = [c for c in ["speeding","harsh_brake","rapid_accel","swerve"] if c in tr_all.columns]
    if not types: return pd.DataFrame(), pd.DataFrame()
    ag = (tr_all.groupby(["driver_id","phase"], as_index=False).agg(distance_km=("distance_km","sum"), **{f"{t}_sum": (t,"sum") for t in types}))
    for t in types:
        ag[f"rate_{t}_per_km"] = ag[f"{t}_sum"] / ag["distance_km"].clip(lower=MIN_KM)
    pre = ag[ag["phase"]=="pre"][["driver_id"] + [f"rate_{t}_per_km" for t in types]].rename(columns={f"rate_{t}_per_km": f"pre_{t}" for t in types})
    post= ag[ag["phase"]=="post"][["driver_id"] + [f"rate_{t}_per_km" for t in types]].rename(columns={f"rate_{t}_per_km": f"post_{t}" for t in types})
    m = pre.merge(post, on="driver_id", how="inner")
    for t in types:
        m[f"delta_{t}"] = m[f"post_{t}"] - m[f"pre_{t}"]
        m[f"ratio_{t}"] = m[f"post_{t}"].replace(0,np.nan) / m[f"pre_{t}"].replace(0,np.nan)
    delta_cols = [f"delta_{t}" for t in types]
    corr = m[delta_cols].corr() if len(m) else pd.DataFrame()
    return m, corr

def ensure_driver_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "driver_id" not in out.columns:
        for c in ("driver_id_x","driver_id_y","driverProfileId","driver","driverID"):
            if c in out.columns:
                out["driver_id"] = out[c].astype(str)
                break
        else:
            out["driver_id"] = out.get("driverProfileId", "unknown").astype(str)
    else:
        out["driver_id"] = out["driver_id"].astype(str)
    return out

# ---------------------------- Main pipeline ----------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    # Load
    tr_pre  = read_csv_auto(FILES["trip_pre"])     if FILES["trip_pre"].exists()    else pd.DataFrame()
    tr_post = read_csv_auto(FILES["trip_post"])    if FILES["trip_post"].exists()   else pd.DataFrame()
    tr_all  = read_csv_auto(FILES["trip_all"])     if FILES["trip_all"].exists()    else pd.DataFrame()

    tr_pre  = norm_trip_df(tr_pre)
    tr_post = norm_trip_df(tr_post)
    tr_all  = norm_trip_df(tr_all)
    if tr_all.empty and (not tr_pre.empty or not tr_post.empty):
        pre = tr_pre.assign(phase="pre")   if "phase" not in tr_pre.columns  else tr_pre
        post= tr_post.assign(phase="post") if "phase" not in tr_post.columns else tr_post
        tr_all = pd.concat([pre, post], ignore_index=True)

    # Descriptives
    bc_pre, bc_post, bc_all = basic_counts(tr_pre, "pre"), basic_counts(tr_post, "post"), basic_counts(tr_all, "all")
    save_table(pd.DataFrame([bc_pre, bc_post, bc_all]), "basic_counts", OUT_DIR)
    ptr = pd.concat([per_type_rates(tr_pre, "pre"), per_type_rates(tr_post, "post")], ignore_index=True)
    if not ptr.empty:
        save_table(ptr, "per_type_rates_per100km", OUT_DIR)

    # UBPK plots
    if not tr_all.empty:
        for ph in ["pre","post"]:
            d = tr_all.loc[tr_all["phase"]==ph, "ubpk"].dropna()
            if len(d):
                plt.figure(figsize=(7,5)); plt.hist(d, bins=40, alpha=0.6); plt.xlabel("UBPK (events per km)"); plt.ylabel("Trip count"); plt.title(f"Trip-level UBPK distribution — {ph}"); save_fig(f"dist_ubpk_trip_phase_{ph}", OUT_DIR); plt.close()
        data_pre  = tr_all.loc[tr_all["phase"]=="pre", "ubpk"].dropna()
        data_post = tr_all.loc[tr_all["phase"]=="post","ubpk"].dropna()
        if len(data_pre) and len(data_post):
            plt.figure(figsize=(6,5)); plt.boxplot([data_pre, data_post], tick_labels=["pre","post"], showfliers=False); plt.ylabel("UBPK (events per km)"); plt.title("Trip-level UBPK by phase"); save_fig("box_ubpk_trip_phase", OUT_DIR); plt.close()

    # Driver aggregation
    drv_agg = driver_aggregate(tr_all)
    save_table(drv_agg, "driver_aggregate_rates", OUT_DIR)
    drv_pairs = paired_driver_rates(drv_agg)
    save_table(drv_pairs, "driver_pairs_rates", OUT_DIR)

    # ======================= REFACTORED: temporal & dynamics outputs =======================
    tr_all_ts = tr_all.copy()
    if tr_all_ts.empty:
        save_txt("No data for temporal/dynamics add-ons.", "temporal_extra_notice", OUT_DIR)
    else:
        tr_all_ts = _parse_ts(tr_all_ts)
        tr_all_ts = _add_period_keys(tr_all_ts)
        tr_all_ts = ensure_driver_id(tr_all_ts)
        
        weekly_drv = _agg_period(tr_all_ts, "week_start")
        monthly_drv = _agg_period(tr_all_ts, "month_start")
        save_table(weekly_drv, "driver_weekly_summary", OUT_DIR)
        save_table(monthly_drv, "driver_monthly_summary", OUT_DIR)

        weekly_overall = _overall_period_rates(tr_all_ts, "week_start")
        monthly_overall = _overall_period_rates(tr_all_ts, "month_start")
        save_table(weekly_overall, "overall_weekly_rates", OUT_DIR)
        save_table(monthly_overall, "overall_monthly_rates", OUT_DIR)

        if not weekly_overall.empty:
            plt.figure(figsize=(9,5));
            for ph, g in weekly_overall.groupby("phase"):
                g = g.sort_values("week_start")
                plt.plot(g["week_start"], g["rate_events_per_km"], label=ph, alpha=0.9)
            plt.axhline(0, color="k", lw=0.5, alpha=0.4); plt.ylabel("Events per km"); plt.title("Overall weekly rate (km-weighted)"); plt.legend(); save_fig("weekly_overall_rates", OUT_DIR); plt.close()

        if not monthly_overall.empty:
            plt.figure(figsize=(9,5))
            for ph, g in monthly_overall.groupby("phase"):
                g = g.sort_values("month_start")
                plt.plot(g["month_start"], g["rate_events_per_km"], marker="o", label=ph, alpha=0.9)
            plt.axhline(0, color="k", lw=0.5, alpha=0.4); plt.ylabel("Events per km"); plt.title("Overall monthly rate (km-weighted)"); plt.legend(); save_fig("monthly_overall_rates", OUT_DIR); plt.close()

        weekly_slopes = _driver_weekly_slopes(weekly_drv)
        save_table(weekly_slopes, "driver_weekly_slopes", OUT_DIR)
        
        # ➕ NEW: driver-level MONTHLY slopes + tests + plot
        monthly_slopes = _driver_monthly_slopes(monthly_drv)
        if not monthly_slopes.empty:
            save_table(monthly_slopes, "driver_monthly_slopes", OUT_DIR)
            _paired_slope_tests(
                monthly_slopes,
                slope_col="slope_per_month",
                label="monthly",
                out_txt="trend_monthly_slope_tests",
                out_dir=OUT_DIR
            )
            # Plot histogram of POST monthly slopes
            if "post" in monthly_slopes["phase"].unique():
                plt.figure(figsize=(7,4))
                post_ms = monthly_slopes.loc[monthly_slopes["phase"]=="post", "slope_per_month"].dropna()
                if not post_ms.empty:
                    plt.hist(post_ms, bins=30, alpha=0.85)
                    plt.axvline(0, color="k", lw=0.8)
                    plt.xlabel("Slope (events/km per month)")
                    plt.title("Distribution of POST monthly slopes (driver-level)")
                    save_fig("driver_monthly_slope_hist", OUT_DIR)
                plt.close()

        # ➕ NEW: overall monthly WLS slope test by phase (km-weighted)
        if not monthly_overall.empty:
            wls_report = _overall_monthly_wls_test(monthly_overall)
            save_txt(wls_report, "overall_monthly_slope_tests", OUT_DIR)

        if not weekly_slopes.empty:
            slope_pivot = weekly_slopes.pivot(index="driver_id", columns="phase", values="slope_per_week").dropna()
            if not slope_pivot.empty and SCIPY_OK:
                tstat, tp = stats.ttest_rel(slope_pivot["post"], slope_pivot["pre"], nan_policy="omit")
                slope_report = [f"n={len(slope_pivot)} drivers with weekly slopes in both phases", f"Paired t-test on slopes (post - pre): t({len(slope_pivot)-1})={tstat:.3f}, p={tp:.4g}"]
                save_txt("\n".join(slope_report), "trend_slope_tests", OUT_DIR)
            
            if "post" in weekly_slopes["phase"].unique():
                plt.figure(figsize=(7,4))
                post_slopes = weekly_slopes.loc[weekly_slopes["phase"]=="post","slope_per_week"].dropna()
                if not post_slopes.empty:
                    plt.hist(post_slopes, bins=30, alpha=0.8); plt.axvline(0, color="k", lw=0.8); plt.xlabel("Slope (events/km per week)"); plt.title("Distribution of POST weekly slopes (driver-level)"); save_fig("driver_weekly_slope_hist", OUT_DIR)
                plt.close()

        responders = _classify_responders(drv_pairs, abs_eps=0.01, rel_eps=0.10) if not drv_pairs.empty else pd.DataFrame()
        if not responders.empty:
            save_table(responders, "responders_by_driver", OUT_DIR)

        bad_share = _post_bad_day_share(tr_all_ts)
        if not bad_share.empty:
            save_table(bad_share, "post_bad_day_share", OUT_DIR)
            plt.figure(figsize=(7,4)); plt.hist(bad_share["post_bad_day_share"].dropna(), bins=20, alpha=0.85); plt.xlabel("Share of POST trips above PRE Q75 UBPK"); plt.title("Non-adherence / bad-day share (POST)"); save_fig("nonadherence_share_hist", OUT_DIR); plt.close()

        dow_overall, dow_by_driver = _dow_rates(tr_all_ts)
        if not dow_overall.empty:
            save_table(dow_overall, "dow_rates_overall", OUT_DIR)
            order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            plt.figure(figsize=(9,4))
            for ph, g in dow_overall.groupby("phase"):
                g = g.copy(); g["dow_name"] = pd.Categorical(g["dow_name"], categories=order, ordered=True); g = g.sort_values("dow_name")
                plt.plot(g["dow_name"].astype(str), g["rate_events_per_km"], marker="o", label=ph)
            plt.ylabel("Events per km"); plt.title("Day-of-week rates (overall, km-weighted)"); plt.legend(); save_fig("dow_rates_overall", OUT_DIR); plt.close()
        if not dow_by_driver.empty:
            save_table(dow_by_driver, "dow_rates_by_driver", OUT_DIR)

        delta_by_type, delta_corr = _driver_type_delta_matrix(tr_all_ts)
        if not delta_by_type.empty:
            save_table(delta_by_type, "delta_by_type_per_driver", OUT_DIR)
        if not delta_corr.empty:
            delta_corr.to_csv(OUT_DIR/"delta_correlation_by_type.csv", index=True)
            print(f"[saved] {OUT_DIR/'delta_correlation_by_type.csv'}")

    # Paired tests
    paired_report = []
    if not drv_pairs.empty and SCIPY_OK:
        d = drv_pairs.dropna(subset=["rate_pre","rate_post"])
        if len(d) >= 2:
            tstat, tp = stats.ttest_rel(d["rate_post"], d["rate_pre"], nan_policy="omit")
            try: wstat, wp = stats.wilcoxon(d["rate_post"], d["rate_pre"], zero_method="wilcox", alternative="two-sided")
            except ValueError: wstat, wp = np.nan, np.nan
            diff = (d["rate_post"] - d["rate_pre"]).dropna()
            cohen_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
            def cliffs_delta(x, y):
                x, y = np.asarray(x), np.asarray(y); m, n = len(x), len(y)
                more = sum((xi > yi) for xi in x for yi in y)
                less = sum((xi < yi) for xi in x for yi in y)
                return (more - less) / (m*n) if m*n > 0 else np.nan
            cd = cliffs_delta(d["rate_post"], d["rate_pre"])
            paired_report.extend([f"n={len(d)} drivers with both phases", f"Paired t-test: t({len(d)-1})={tstat:.3f}, p={tp:.4g}", f"Wilcoxon signed-rank: W={wstat}, p={wp:.4g}", f"Cohen's d (paired)={cohen_d:.3f}", f"Cliff's delta={cd:.3f} (negative favours POST<PRE)"])
    save_txt("\n".join(paired_report), "driver_paired_tests", OUT_DIR)

    # Spaghetti plot
    if not drv_pairs.empty:
        plt.figure(figsize=(8,6))
        for _, r in drv_pairs.iterrows():
            plt.plot([0,1], [r["rate_pre"], r["rate_post"]], alpha=0.3)
        plt.scatter(np.zeros(len(drv_pairs)), drv_pairs["rate_pre"], alpha=0.7, label="pre"); plt.scatter(np.ones(len(drv_pairs)), drv_pairs["rate_post"], alpha=0.7, label="post"); plt.xticks([0,1], ["pre","post"]); plt.ylabel("Driver-level rate (events/km)"); plt.title("Driver-level change in rate (pre→post)"); plt.legend(); save_fig("spaghetti_driver_rates", OUT_DIR); plt.close()

    # GLMs
    glm_total, glm_df, _ = fit_glm_models(tr_all, y_col="total_events")
    if glm_total is not None:
        print(f"[GLM] total rows={len(glm_df)}, drivers={glm_df['driver_id'].nunique()}, groups={glm_df['driver_group'].nunique()}"); save_table(glm_total, "glm_total_events_models", OUT_DIR)
    etype_cols = [c for c in ["speeding","harsh_brake","rapid_accel","swerve"] if c in tr_all.columns]
    etype_results = []
    for y in etype_cols:
        if pd.to_numeric(tr_all[y], errors="coerce").fillna(0).sum() == 0:
            etype_results.append(pd.DataFrame([{"model": "Poisson+exposure (skipped: all zeros)", "y": y, "coef": np.nan, "se": np.nan, "RR_post_vs_pre": 1.0, "CI95_lo": np.nan, "CI95_hi": np.nan}])); continue
        tab, _, _ = fit_glm_models(tr_all, y_col=y)
        if tab is not None: etype_results.append(tab)
    if etype_results:
        pertype = pd.concat(etype_results, ignore_index=True); save_table(pertype, "glm_per_event_type_models", OUT_DIR)

    if glm_total is not None:
        plot_forest_rr(glm_total, "Rate Ratio (Post vs Pre) — Total Events", "forest_rr_total", OUT_DIR)
    if etype_results:
        plot_forest_rr(pd.concat(etype_results, ignore_index=True), "Rate Ratio (Post vs Pre) — By Event Type", "forest_rr_by_type", OUT_DIR)

    # Bayesian model
    bayes_artifacts, bayes_summary = fit_bayesian_hierarchical_poisson(tr_all, y_col="total_events")
    if bayes_summary is not None:
        save_table(pd.DataFrame([bayes_summary]), "bayes_total_events_summary", OUT_DIR)

    # Sensitivity
    if not tr_all.empty:
        sensitivity_short_trip_exclusion(tr_all, SHORT_TRIP_KM)
        sensitivity_trim_ubpk(tr_all, 0.99)

    # APA/JARS report and final summaries
   
    generate_apa_report_and_summary(bc_pre, bc_post, bc_all, drv_pairs, glm_total, bayes_summary, OUT_DIR)
    
    # Console checks
    check_overall_rr(OUT_DIR)
    check_per_type_rr(OUT_DIR)
    print("\nAll analysis artifacts written to:", OUT_DIR.resolve())

def sensitivity_short_trip_exclusion(tr: pd.DataFrame, cutoff_km=SHORT_TRIP_KM):
    df = tr.loc[tr["distance_km"] >= cutoff_km].copy()
    tab, _, _ = fit_glm_models(df, y_col="total_events")
    if tab is not None:
        save_table(tab, f"glm_total_events_sensitivity_ge_{cutoff_km}km", OUT_DIR)

def sensitivity_trim_ubpk(tr: pd.DataFrame, trim_pct=0.99):
    thr = tr["ubpk"].quantile(trim_pct)
    df = tr.loc[tr["ubpk"] <= thr].copy()
    tab, _, _ = fit_glm_models(df, y_col="total_events")
    if tab is not None:
        tail = int((1-trim_pct)*100)
        save_table(tab, f"glm_total_events_sensitivity_trim_{tail}pct", OUT_DIR)

 # APA/JARS report and final summaries
def generate_apa_report_and_summary(bc_pre, bc_post, bc_all, drv_pairs, glm_total, bayes_summary, out_dir):
    lines = ["**Sample & Exposure.** We analyzed pre- and post-intervention trips with distance as exposure (events per km).", f"Pre: drivers={bc_pre.get('drivers','NA')}, trips={bc_pre.get('trips','NA')}, total distance={bc_pre.get('total_km',0.0):.1f} km.", f"Post: drivers={bc_post.get('drivers','NA')}, trips={bc_post.get('trips','NA')}, total distance={bc_post.get('total_km',0.0):.1f} km.\n"]
    if not drv_pairs.empty and SCIPY_OK:
        d = drv_pairs.dropna(subset=['rate_pre','rate_post'])
        if len(d) >= 2:
            tstat, tp = stats.ttest_rel(d["rate_post"], d["rate_pre"], nan_policy="omit")
            try: wstat, wp = stats.wilcoxon(d["rate_post"], d["rate_pre"], zero_method="wilcox", alternative="two-sided")
            except ValueError: wstat, wp = np.nan, np.nan
            diff = (d["rate_post"] - d["rate_pre"]).dropna()
            cohen_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
            lines.extend(["**Driver-level paired analysis.** Per-driver event rates (events/km) were compared across phases.", f"Paired t-test, n={len(d)}: t({len(d)-1})={apa_decimal(tstat)}, p={apa_decimal(tp)}."])
            if not np.isnan(wp): lines.append(f"Wilcoxon signed-rank: W={wstat}, p={apa_decimal(wp)}.")
            lines.append(f"Cohen's d (paired)={apa_decimal(cohen_d)}.\n")
    if isinstance(glm_total, pd.DataFrame) and not glm_total.empty:
        row = _pick_poisson_row(glm_total)
        if row is not None:
            rr, lo, hi = float(row["RR_post_vs_pre"]), float(row["CI95_lo"]), float(row["CI95_hi"])
            lines.extend(["**Trip-level regression (Frequentist).** Poisson GLM with exposure and driver clustering.", f"Rate ratio (Post vs Pre) = {apa_decimal(rr)} [95% CI {apa_decimal(lo)}, {apa_decimal(hi)}].\n"])
    if bayes_summary is not None:
        lines.extend(["**Bayesian hierarchical model.** Poisson with driver random intercepts and log-distance exposure.", f"Posterior rate ratio (Post vs Pre): median={apa_decimal(bayes_summary['rr_med'])}, 95% CrI [{apa_decimal(bayes_summary['rr_lo'])}, {apa_decimal(bayes_summary['rr_hi'])}].\n"])
    else:
        lines.append("**Bayesian hierarchical model.** Skipped (PyMC/ArviZ unavailable, data empty, or convergence issue). Results rely on frequentist models.\n")
    lines.append("**Reporting standard.** Analyses follow JARS: data screening, exposure adjustment, model specification, effect sizes with uncertainty (CIs/CrIs), and sensitivity checks (short-trip exclusion; UBPK trimming).")
    save_txt("".join(lines), "APA_JARS_report", out_dir)
    print("APA/JARS-style report written to analysis_outputs/APA_JARS_report.txt")

    if not drv_pairs.empty:
        top_improve = drv_pairs.sort_values("delta").head(10); worst = drv_pairs.sort_values("delta", ascending=False).head(10)
        save_table(top_improve, "top10_driver_improvement_delta", out_dir); save_table(worst, "top10_driver_deterioration_delta", out_dir)

    def fmt_bc(d): return f"{d['label']}: drivers={d['drivers']}, trips={d['trips']}, total_km={d['total_km']:.1f}, trips_with_any_event={d['trips_with_any_event']} ({100*d['pct_trips_with_any_event']:.1f}%), total_events={d['total_events']}"
    summary_txt = ["=== BASIC COUNTS ===", fmt_bc(bc_pre), fmt_bc(bc_post), fmt_bc(bc_all)]
    if isinstance(glm_total, pd.DataFrame) and not glm_total.empty:
        row = _pick_poisson_row(glm_total)
        if row is not None:
            rr, lo, hi = float(row["RR_post_vs_pre"]), float(row["CI95_lo"]), float(row["CI95_hi"])
            summary_txt.append(f"Trip-level Poisson (cluster-robust): RR_post_vs_pre={rr:.3f} [{lo:.3f}, {hi:.3f}]")
    if not drv_pairs.empty:
        diff = (drv_pairs.dropna(subset=["rate_pre","rate_post"])["rate_post"] - drv_pairs.dropna(subset=["rate_pre","rate_post"])["rate_pre"]).mean()
        summary_txt.append(f"Driver-level mean Δ rate (post - pre): {diff:.4f} events/km")
    save_txt("\n".join(summary_txt), "high_level_summary", out_dir)

def check_overall_rr(out_dir):
    p = out_dir/"glm_total_events_models.csv"
    if not p.exists(): return
    row = _pick_poisson_row(pd.read_csv(p))
    if row is None: return
    rr = row["RR_post_vs_pre"]; verdict = "GOOD (post<pre)" if rr < 1 else "Needs work (post≥pre)"
    print(f"[CHECK] Total events RR (Post/Pre) = {rr:.3f}   → {verdict}")

def check_per_type_rr(out_dir):
    p = out_dir/"glm_per_event_type_models.csv"
    if not p.exists(): return
    df = pd.read_csv(p)
    df = df[df["model"].str.contains("Poisson", na=False)][["y","RR_post_vs_pre","CI95_lo","CI95_hi"]]
    print("\n[CHECK] Per-event type RRs (Post/Pre):")
    for _,r in df.iterrows():
        verdict = "GOOD" if r.RR_post_vs_pre < 1 else "Needs work"
        print(f"  {r.y:12s}   RR={r.RR_post_vs_pre:.3f}   CI[{r.CI95_lo:.3f},{r.CI95_hi:.3f}]   → {verdict}")

if __name__ == "__main__":
    main()