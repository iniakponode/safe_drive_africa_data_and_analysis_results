# phase2_recompute_from_phase1.py  (you can paste into a notebook cell)
from __future__ import annotations
from pathlib import Path
import logging, numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("phase2")

# ---------- input/output ----------
IN_PRE  = Path("pre_dataset")
IN_POST = Path("post_dataset")
OUT_DIR = Path("computed_unsafe_behaviours_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- thresholds (aligned with your request) ----------
SPEED_MARGIN_KPH   = 6.0
ABS_SPEED_THR_KPH  = 115.0
MIN_SPEEDING_SEC   = 5.0       # ↑ longer contiguous overspeed to count 1 event

ACC_Y_ACCEL_THR    = 2.3
ACC_Y_BRAKE_THR    = -2.6
GYRO_Z_SWERVE_THR  = 0.32      # swerve = simple threshold on |gz_smooth|
MIN_PULSE_SEC      = 0.50      # ↑ longer pulses => fewer accel/brake/swerve events
ROLL_MA_WIN        = 3

# MIN_PULSE_SEC      = 0.50   # was 0.70 -
# ACC_Y_ACCEL_THR    = 2.3    # was 2.5 -
# ACC_Y_BRAKE_THR    = -2.6   # was -2.8 -
# GYRO_Z_SWERVE_THR  = 0.32  # - was 0.35 -

MIN_KM             = 1e-6

def read_csv_auto(p: Path) -> pd.DataFrame:
    """Read CSV with robust encoding handling, including chunked reading for large files."""
    if not p.exists():
        log.warning(f"File not found: {p}")
        return pd.DataFrame()
    
    file_size_mb = p.stat().st_size / (1024 * 1024)
    log.info(f"Reading {p.name} ({file_size_mb:.1f} MB)...")
    
    # For very large files (>500MB), use chunked reading with latin1
    if file_size_mb > 500 and p.name.startswith("raw_sensor"):
        log.info(f"Large file detected, using chunked reading with latin1 encoding...")
        try:
            # Read in chunks to avoid memory issues
            chunks = []
            chunk_size = 100000
            for chunk in pd.read_csv(p, encoding='latin1', low_memory=False, chunksize=chunk_size, on_bad_lines='skip'):
                chunks.append(chunk)
                if len(chunks) % 10 == 0:
                    log.info(f"  Read {len(chunks) * chunk_size:,} rows so far...")
            df = pd.concat(chunks, ignore_index=True)
            log.info(f"  Successfully read {len(df):,} rows")
            return df
        except Exception as e:
            log.error(f"Chunked reading failed for {p.name}: {e}")
            return pd.DataFrame()
    
    # For normal-sized files, try multiple encodings
    for enc in ("latin1", "utf-8", "utf-8-sig", "cp1252", "iso-8859-1"):
        try:
            return pd.read_csv(p, encoding=enc, low_memory=False, on_bad_lines='skip')
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            log.debug(f"Failed with encoding {enc}: {e}")
            pass
    
    # If all encodings fail, return empty DataFrame
    log.error(f"Failed to read {p.name} with any encoding. File may be corrupted.")
    return pd.DataFrame()

def _safe_num(s: pd.Series, dtype=None) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    return out.astype(dtype) if dtype is not None else out

def _moving_avg(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or x.size == 0: return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(k, float) / k
    return np.convolve(xp, ker, mode="valid")

def _clusters_from_boolean(time_ms: np.ndarray, mask: np.ndarray, min_sec: float) -> int:
    if time_ms.size == 0: return 0
    t = np.asarray(time_ms, dtype=np.int64)
    m = np.asarray(mask, dtype=bool)
    order = np.argsort(t, kind="mergesort")
    t = t[order]; m = m[order]
    starts = np.where(m & np.concatenate([[True], ~m[:-1]]))[0]
    ends   = np.where(m & np.concatenate([~m[1:], [True]]))[0]
    if starts.size == 0: return 0
    cnt = 0
    for s, e in zip(starts, ends):
        dur = (t[e] - t[s]) / 1000.0 if e >= s else 0.0
        if dur >= min_sec: cnt += 1
    return cnt

def _parse_values_3cols(s: pd.Series):
    ss = s.astype(str).str.strip("[]")
    parts = ss.str.split(",", expand=True)
    if parts.shape[1] < 3:
        for _ in range(3 - parts.shape[1]): parts[str(parts.shape[1])] = np.nan
        parts = parts.iloc[:, :3]
    x = pd.to_numeric(parts.iloc[:,0].str.strip(), errors="coerce").to_numpy()
    y = pd.to_numeric(parts.iloc[:,1].str.strip(), errors="coerce").to_numpy()
    z = pd.to_numeric(parts.iloc[:,2].str.strip(), errors="coerce").to_numpy()
    return x, y, z

def _estimate_distance_km_from_loc(loc_df: pd.DataFrame) -> float:
    if loc_df.empty or "timestamp" not in loc_df or "speed" not in loc_df: return 0.0
    t = _safe_num(loc_df["timestamp"]).to_numpy(dtype="float64")
    v = _safe_num(loc_df["speed"]).to_numpy(dtype="float64")
    ok = np.isfinite(t) & np.isfinite(v); t = t[ok]; v = v[ok]
    if t.size < 2: return 0.0
    order = np.argsort(t, kind="mergesort"); t = t[order]; v = v[order]
    dt = np.diff(t) / 1000.0
    if dt.size == 0: return 0.0
    v_mid = 0.5 * (v[1:] + v[:-1])
    km = float(np.sum(v_mid * (dt / 3600.0)))
    return max(0.0, km)

def load_phase_minimal(in_dir: Path, phase_label: str):
    trips = read_csv_auto(in_dir/"trip_cleaned.csv")
    loc   = read_csv_auto(in_dir/"location_cleaned.csv")
    raw   = read_csv_auto(in_dir/"raw_sensor_data_cleaned.csv")
    # drivers file optional for integrity; not used in counts
    drivers_path = in_dir/"driver_profile_cleaned.csv"
    drivers = read_csv_auto(drivers_path) if drivers_path.exists() else pd.DataFrame()
    # normalize IDs/cols
    for c in ("id","driverProfileId","phase","start_time","end_time"):
        if c not in trips.columns:
            if c == "phase": trips[c] = phase_label
            else: trips[c] = pd.NA
    trips["id"] = trips["id"].astype(str)
    trips["driverProfileId"] = trips["driverProfileId"].astype(str)
    trips["start_time"] = _safe_num(trips["start_time"]).astype("Int64")
    trips["end_time"]   = _safe_num(trips["end_time"]).astype("Int64")
    trips["phase"] = trips["phase"].astype(str).fillna(phase_label)

    if not loc.empty:
        keep = [c for c in ("trip_id","timestamp","speed","speedLimit") if c in loc.columns]
        loc = loc[keep].copy()
        if "trip_id" in loc: loc["trip_id"] = loc["trip_id"].astype(str)
        if "timestamp" in loc: loc["timestamp"] = _safe_num(loc["timestamp"]).astype("Int64")
        if "speed" in loc: loc["speed"] = _safe_num(loc["speed"])
        if "speedLimit" in loc: loc["speedLimit"] = _safe_num(loc["speedLimit"])
        loc.sort_values(["trip_id","timestamp"], inplace=True, kind="mergesort")

    if not raw.empty:
        keep = [c for c in ("trip_id","timestamp","sensor_type_name","values") if c in raw.columns]
        raw = raw[keep].copy()
        raw["trip_id"] = raw["trip_id"].astype(str)
        raw["timestamp"] = _safe_num(raw["timestamp"]).astype("Int64")
        raw["sensor_type_name"] = raw["sensor_type_name"].astype(str).str.lower()
    return trips, loc, raw, drivers

def recompute_phase(in_dir: Path, out_dir: Path, phase_label: str):
    trips, loc, raw, _ = load_phase_minimal(in_dir, phase_label)
    if trips.empty: return pd.DataFrame(), pd.DataFrame()

    acc = raw.loc[raw["sensor_type_name"].str.contains("accel", na=False)].copy() if not raw.empty else pd.DataFrame()
    gyr = raw.loc[raw["sensor_type_name"].str.contains("gyro",  na=False)].copy() if not raw.empty else pd.DataFrame()
    acc.sort_values(["trip_id","timestamp"], inplace=True, kind="mergesort")
    gyr.sort_values(["trip_id","timestamp"], inplace=True, kind="mergesort")

    loc_groups = dict(tuple(loc.groupby("trip_id", sort=False))) if not loc.empty else {}
    acc_groups = dict(tuple(acc.groupby("trip_id", sort=False))) if not acc.empty else {}
    gyr_groups = dict(tuple(gyr.groupby("trip_id", sort=False))) if not gyr.empty else {}

    rows_sum, rows_evt = [], []
    for _, r in trips.iterrows():
        tid = str(r["id"]); drv = str(r["driverProfileId"])
        # distance from loc (integrate speed/time)
        L = loc_groups.get(tid, pd.DataFrame())
        dkm = _estimate_distance_km_from_loc(L)

        # speeding
        sp_cnt = 0
        if not L.empty:
            t = L["timestamp"].to_numpy(dtype="int64")
            v = L["speed"].to_numpy(dtype="float64") if "speed" in L else np.full_like(t, np.nan, float)
            lim = L["speedLimit"].to_numpy(dtype="float64") if "speedLimit" in L else np.full_like(t, np.nan, float)
            have_lim = np.isfinite(lim)
            over_lim = (v - lim) > SPEED_MARGIN_KPH
            over_abs = v > ABS_SPEED_THR_KPH
            mask = np.where(have_lim, over_lim, over_abs)
            sp_cnt = _clusters_from_boolean(t, mask, MIN_SPEEDING_SEC)

        # accel / brake
        ac_cnt = br_cnt = 0
        A = acc_groups.get(tid, pd.DataFrame())
        if not A.empty:
            t = A["timestamp"].to_numpy(dtype="int64")
            _, ay, _ = _parse_values_3cols(A["values"])
            ay_s = _moving_avg(ay, ROLL_MA_WIN)
            ac_cnt = _clusters_from_boolean(t, (ay_s >= ACC_Y_ACCEL_THR), MIN_PULSE_SEC)
            br_cnt = _clusters_from_boolean(t, (ay_s <= ACC_Y_BRAKE_THR), MIN_PULSE_SEC)

        # swerve (simple threshold on |gz| after smoothing)
        sw_cnt = 0
        G = gyr_groups.get(tid, pd.DataFrame())
        if not G.empty:
            t = G["timestamp"].to_numpy(dtype="int64")
            _, _, gz = _parse_values_3cols(G["values"])
            gz_s = _moving_avg(gz, ROLL_MA_WIN)
            sw_cnt = _clusters_from_boolean(t, (np.abs(gz_s) >= GYRO_Z_SWERVE_THR), MIN_PULSE_SEC)

        total = int(sp_cnt + ac_cnt + br_cnt + sw_cnt)
        ubpk  = total / max(MIN_KM, dkm)

        rows_sum.append({
            "trip_id": tid, "driverProfileId": drv, "phase": phase_label,
            "distance_km": dkm, "speeding": sp_cnt, "rapid_accel": ac_cnt,
            "harsh_brake": br_cnt, "swerve": sw_cnt, "total_events": total, "ubpk": ubpk,
            "start_time": r.get("start_time", pd.NA)
        })
        rows_evt.append({
            "trip_id": tid, "driverProfileId": drv, "phase": phase_label,
            "speeding": sp_cnt, "rapid_accel": ac_cnt, "harsh_brake": br_cnt,
            "swerve": sw_cnt, "total_events": total
        })

    trip_summary = pd.DataFrame(rows_sum)
    per_trip     = pd.DataFrame(rows_evt)

    any_evt = int((trip_summary["total_events"] > 0).sum())
    log.info(f"[{phase_label}] trips with ≥1 event: {any_evt} ({100.0*any_evt/max(1,len(trip_summary)):.1f}%)")
    trip_summary.to_csv(out_dir/f"trip_event_summary_{phase_label}.csv", index=False)
    per_trip.to_csv(out_dir/f"unsafe_events_{phase_label}.csv", index=False)
    return trip_summary, per_trip

def run_phase2():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pre_sum, pre_evt   = recompute_phase(IN_PRE,  OUT_DIR, "pre")
    post_sum, post_evt = recompute_phase(IN_POST, OUT_DIR, "post")

    if not pre_sum.empty or not post_sum.empty:
        all_sum = pd.concat([pre_sum, post_sum], ignore_index=True)
        all_sum.to_csv(OUT_DIR/"trip_event_summary_all.csv", index=False)
        log.info("[all] saved trip_event_summary_all.csv")

        for ph in ("pre","post"):
            df = all_sum.loc[all_sum["phase"]==ph]
            if df.empty: continue
            n = len(df)
            any_evt = int((df["total_events"]>0).sum())
            mean_ubpk = float(df["ubpk"].mean())
            log.info(f"[{ph}] trips={n} | ≥1 event={any_evt} | mean UBPK={mean_ubpk:.3f}")
    log.info("Phase 2 recompute done.")

if __name__ == "__main__":
    run_phase2()
