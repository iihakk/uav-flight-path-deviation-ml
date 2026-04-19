"""
UAV Flight Path Deviation Pipeline v2 (final report)
====================================================
Extends pipeline.py to use onboard ArduPilot .bin telemetry parsed with
pymavlink, in addition to the Open-Meteo hourly weather backfill. Produces
the figures and tables referenced by final_report/report.tex.

Feature sets compared under 5-fold GroupKFold CV:
  WEATHER  — Open-Meteo hourly only (same family as midterm).
  ONBOARD  — attitude, motor output, rates, vibration, EKF innovations.
  ALL      — both combined.

Models: Decision Tree, KNN (standardized), Random Forest.
Targets:
  regression     = cross-track deviation in metres.
  classification = deviation >= 75th percentile.
"""
import os, re, glob, json, urllib.request, urllib.parse
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymavlink import mavutil

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------
PROJECT = "/Users/abdulazizal-haidary/development/Machine Learning/Project"
DATA = os.path.join(PROJECT, "Data", "extracted")
FLIGHTS_DIR = os.path.join(DATA, "Flight_path_for_UAV_correction")
BIN_DIR = os.path.join(FLIGHTS_DIR, "Flight_logs_during_mission")
DESIRED_MASTER = os.path.join(FLIGHTS_DIR,
                              "Desired_flight_path_soil_moisture_Research_plot.kml")
OUT_DIR = os.path.join(PROJECT, "final_report")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
CACHE_DIR = os.path.join(PROJECT, "cache_v2")
for d in (FIG_DIR, TAB_DIR, CACHE_DIR):
    os.makedirs(d, exist_ok=True)

# --------------------------------------------------------------------------
# 1. Parse desired KML
def parse_kml_waypoints(path):
    tree = ET.parse(path); root = tree.getroot()
    pts = []
    for coord in root.iter("{http://www.opengis.net/kml/2.2}coordinates"):
        for token in coord.text.strip().split():
            parts = token.split(",")
            if len(parts) >= 2:
                pts.append((float(parts[1]), float(parts[0])))  # lat, lon
    return pd.DataFrame(pts, columns=["lat", "lon"])

desired = parse_kml_waypoints(DESIRED_MASTER)
R_EARTH = 6371000.0
lat0 = np.deg2rad(desired.lat.mean())
lon0 = np.deg2rad(desired.lon.mean())

def to_xy(lat, lon):
    lat_r = np.deg2rad(np.asarray(lat))
    lon_r = np.deg2rad(np.asarray(lon))
    x = R_EARTH * (lon_r - lon0) * np.cos(lat0)
    y = R_EARTH * (lat_r - lat0)
    return x, y

dx, dy = to_xy(desired.lat.values, desired.lon.values)

def cross_track(px, py, sx, sy):
    px = np.asarray(px); py = np.asarray(py)
    min_d = np.full(px.shape, np.inf)
    for i in range(len(sx) - 1):
        x1, y1, x2, y2 = sx[i], sy[i], sx[i + 1], sy[i + 1]
        vx, vy = x2 - x1, y2 - y1
        L2 = vx*vx + vy*vy
        if L2 == 0:
            d = np.hypot(px - x1, py - y1)
        else:
            t = ((px - x1)*vx + (py - y1)*vy) / L2
            t = np.clip(t, 0, 1)
            cx = x1 + t*vx; cy = y1 + t*vy
            d = np.hypot(px - cx, py - cy)
        min_d = np.minimum(min_d, d)
    return min_d

print(f"[desired] {len(desired)} waypoints; plot center=({desired.lat.mean():.5f},{desired.lon.mean():.5f})")

# --------------------------------------------------------------------------
# 2. Parse one BIN file → dict of DataFrames per message type, then merge on TimeUS
NEEDED = ["GPS", "ATT", "RATE", "RCOU", "CTUN", "VIBE", "XKF4"]

def parse_bin(path):
    cache = os.path.join(CACHE_DIR, os.path.basename(path) + ".parquet.feather")
    cache_csv = os.path.join(CACHE_DIR, os.path.basename(path) + ".csv.gz")
    if os.path.exists(cache_csv):
        return pd.read_csv(cache_csv, compression="gzip")
    m = mavutil.mavlink_connection(path)
    buckets = {k: [] for k in NEEDED}
    while True:
        msg = m.recv_match(type=NEEDED, blocking=False)
        if msg is None: break
        buckets[msg.get_type()].append(msg.to_dict())
    frames = {k: pd.DataFrame(v) for k, v in buckets.items() if v}
    if "GPS" not in frames: return None
    gps = frames["GPS"].copy()
    gps = gps[gps["Status"] >= 3]                       # valid 3D fix
    gps = gps.rename(columns={"Lat":"lat", "Lng":"lon", "Alt":"alt",
                              "Spd":"ground_speed", "GCrs":"ground_course",
                              "NSats":"n_sats"})
    keep = ["TimeUS","lat","lon","alt","ground_speed","ground_course","VZ","n_sats"]
    base = gps[keep].sort_values("TimeUS").reset_index(drop=True)
    # Derived subsets
    for k in ["ATT", "RATE", "CTUN", "VIBE", "XKF4", "RCOU"]:
        if k not in frames: continue
        sub = frames[k].sort_values("TimeUS")
        if k == "ATT":
            sub = sub[["TimeUS","Roll","Pitch","Yaw","DesRoll","DesPitch","DesYaw"]]
        elif k == "RATE":
            sub = sub[["TimeUS","R","RDes","P","PDes","Y","YDes"]]
            sub.columns = ["TimeUS","rate_R","rate_RDes","rate_P","rate_PDes","rate_Y","rate_YDes"]
        elif k == "CTUN":
            cols = [c for c in ["TimeUS","ThI","ABst","ThO","ThH","DAlt","Alt","DCRt","CRt"] if c in sub.columns]
            sub = sub[cols]
        elif k == "VIBE":
            cols = [c for c in ["TimeUS","VibeX","VibeY","VibeZ","Clip0","Clip1","Clip2"] if c in sub.columns]
            sub = sub[cols]
        elif k == "XKF4":
            cols = [c for c in ["TimeUS","SV","SP","SH","SM","SVT","errRP","OFN","OFE"] if c in sub.columns]
            sub = sub[cols]
        elif k == "RCOU":
            cols = [c for c in ["TimeUS","C1","C2","C3","C4","C5","C6","C7","C8"] if c in sub.columns]
            sub = sub[cols]
        sub = sub.sort_values("TimeUS")
        base = pd.merge_asof(base, sub, on="TimeUS", direction="nearest",
                             tolerance=int(0.5e6))      # 0.5 s max
    base.to_csv(cache_csv, compression="gzip", index=False)
    return base

# --------------------------------------------------------------------------
# 3. Load every BIN
bin_files = sorted(glob.glob(os.path.join(BIN_DIR, "*.bin")))
print(f"[bin] {len(bin_files)} BIN files found")

flights = []
for f in bin_files:
    fname = os.path.basename(f)
    m = re.match(r"(\d{4})-(\d{2})-(\d{2}) (\d{2})-(\d{2})-(\d{2})\.bin$", fname)
    if not m: continue
    y, mo, d, hh, mm, ss = map(int, m.groups())
    print(f"[bin] parsing {fname} …", end=" ", flush=True)
    df = parse_bin(f)
    if df is None or len(df) < 100:
        print("skipped (insufficient GPS)")
        continue
    # bbox filter around plot
    pad = 0.01
    lat_min, lat_max = desired.lat.min()-pad, desired.lat.max()+pad
    lon_min, lon_max = desired.lon.min()-pad, desired.lon.max()+pad
    df = df[(df.lat.between(lat_min,lat_max)) & (df.lon.between(lon_min,lon_max))]
    if len(df) < 100:
        print("skipped (bbox)")
        continue
    anchor = pd.Timestamp(year=y, month=mo, day=d, hour=hh, minute=mm, second=ss,
                          tz="US/Eastern")
    df = df.reset_index(drop=True)
    df["ts"] = anchor + pd.to_timedelta(df["TimeUS"] - df["TimeUS"].iloc[0], unit="us")
    df["flight_id"] = fname
    flights.append(df)
    print(f"rows={len(df)}, span={df.ts.min().strftime('%H:%M')}–{df.ts.max().strftime('%H:%M')}")

flights = pd.concat(flights, ignore_index=True)
print(f"[bin] total rows: {len(flights):,}  |  flights: {flights.flight_id.nunique()}")

# --------------------------------------------------------------------------
# 4. Cross-track deviation
fx, fy = to_xy(flights.lat.values, flights.lon.values)
flights["x"] = fx; flights["y"] = fy
flights["deviation_m"] = cross_track(fx, fy, dx, dy)
print(f"[deviation] mean={flights.deviation_m.mean():.2f} m "
      f"median={flights.deviation_m.median():.2f} m "
      f"p95={flights.deviation_m.quantile(0.95):.2f} m "
      f"max={flights.deviation_m.max():.2f} m")

# Drone heading from GPS ground_course directly (ArduPilot already computes it)
flights["heading_deg"] = flights["ground_course"]

# --------------------------------------------------------------------------
# 5. Onboard-derived features
f = flights.copy()
# Tilt magnitude (radians) — ArduPilot logs ATT.Roll/Pitch in degrees
f["tilt_mag_deg"] = np.hypot(f["Roll"], f["Pitch"])
# Motor output spread (std of 4 motors)
motor_cols = [c for c in ["C1","C2","C3","C4"] if c in f.columns]
if len(motor_cols) == 4:
    f["motor_std"] = f[motor_cols].std(axis=1)
    f["motor_mean"] = f[motor_cols].mean(axis=1)
else:
    f["motor_std"] = np.nan
    f["motor_mean"] = np.nan
# Vibration magnitude
f["vibe_mag"] = np.sqrt(f["VibeX"].pow(2) + f["VibeY"].pow(2) + f["VibeZ"].pow(2))
# Body rate magnitude
f["rate_mag"] = np.sqrt(f["rate_R"].pow(2) + f["rate_P"].pow(2) + f["rate_Y"].pow(2))
# EKF horizontal position innovation magnitude (from XKF4 OFN/OFE)
if "OFN" in f.columns and "OFE" in f.columns:
    f["ekf_pos_err"] = np.hypot(f["OFN"], f["OFE"])
else:
    f["ekf_pos_err"] = np.nan
# Attitude tracking error
f["roll_err"] = f["DesRoll"] - f["Roll"]
f["pitch_err"] = f["DesPitch"] - f["Pitch"]
f["att_err_mag"] = np.hypot(f["roll_err"], f["pitch_err"])
# Throttle delta
if "ThI" in f.columns and "ThO" in f.columns:
    f["throttle_err"] = f["ThO"] - f["ThI"]
else:
    f["throttle_err"] = np.nan
# Climb rate error
if "DCRt" in f.columns and "CRt" in f.columns:
    f["climb_err"] = f["DCRt"] - f["CRt"]
else:
    f["climb_err"] = np.nan
# Altitude
f["altitude_m"] = f["alt"]

# --------------------------------------------------------------------------
# 6. Fetch Open-Meteo hourly weather for full date range
def fetch_open_meteo(lat, lon, start, end):
    url = ("https://archive-api.open-meteo.com/v1/archive?"
           + urllib.parse.urlencode(dict(
               latitude=lat, longitude=lon,
               start_date=start, end_date=end,
               hourly="wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
                      "temperature_2m,relative_humidity_2m,surface_pressure",
               timezone="America/New_York", wind_speed_unit="ms")))
    with urllib.request.urlopen(url, timeout=30) as r:
        js = json.loads(r.read().decode())
    h = js["hourly"]
    wx = pd.DataFrame(h)
    wx["ts"] = pd.to_datetime(wx["time"]).dt.tz_localize("US/Eastern",
                                                         nonexistent="shift_forward",
                                                         ambiguous="NaT")
    return wx.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

start_date = f.ts.min().date().isoformat()
end_date = f.ts.max().date().isoformat()
wx = fetch_open_meteo(desired.lat.mean(), desired.lon.mean(), start_date, end_date)
print(f"[weather] Open-Meteo hourly rows: {len(wx)}")

f = f.sort_values("ts").reset_index(drop=True)
wx_cols = ["wind_speed_10m","wind_direction_10m","wind_gusts_10m",
           "temperature_2m","relative_humidity_2m","surface_pressure"]
f = pd.merge_asof(f, wx[["ts"] + wx_cols], on="ts", direction="nearest",
                  tolerance=pd.Timedelta("90min"))

# wind-angle features
f["wind_dir_rad"] = np.deg2rad(f["wind_direction_10m"])
f["wind_sin"] = np.sin(f["wind_dir_rad"])
f["wind_cos"] = np.cos(f["wind_dir_rad"])
hdg_rad = np.deg2rad(f["heading_deg"])
rel = (f["wind_dir_rad"] - hdg_rad + np.pi) % (2*np.pi) - np.pi
f["rel_wind_abs_deg"] = np.degrees(np.abs(rel))
f["crosswind_comp"] = f["wind_speed_10m"] * np.sin(rel).abs()
f["headwind_comp"] = f["wind_speed_10m"] * np.cos(rel)

# Down-sample to ~1Hz for computational sanity (still 10x midterm scale in total)
f["sec_bucket"] = (f["ts"].astype("int64") // 10**9)
f = f.groupby(["flight_id","sec_bucket"], as_index=False).first()
print(f"[features] after 1-Hz downsample: {len(f):,} rows × {f.shape[1]} cols "
      f"| flights: {f.flight_id.nunique()}")

# --------------------------------------------------------------------------
# 7. Feature sets
FS = {
    "WEATHER": ["wind_speed_10m","wind_gusts_10m","wind_sin","wind_cos",
                "rel_wind_abs_deg","crosswind_comp","headwind_comp",
                "temperature_2m","relative_humidity_2m","surface_pressure"],
    "ONBOARD": ["tilt_mag_deg","motor_std","motor_mean","vibe_mag","rate_mag",
                "ekf_pos_err","att_err_mag","throttle_err","climb_err",
                "ground_speed","altitude_m"],
}
FS["ALL"] = FS["WEATHER"] + FS["ONBOARD"]
all_feats = sorted({c for cols in FS.values() for c in cols})

# drop rows missing essentials
f = f.dropna(subset=all_feats + ["deviation_m"]).reset_index(drop=True)
print(f"[features] after NA drop: {len(f):,} rows")

# save processed dataset
f.to_csv(os.path.join(CACHE_DIR, "merged_v2.csv.gz"), compression="gzip", index=False)

# --------------------------------------------------------------------------
# 8. Evaluate models
groups = f["flight_id"].values
y_reg = f["deviation_m"].values
threshold = np.quantile(y_reg, 0.75)
y_cls = (y_reg >= threshold).astype(int)
n_groups = len(np.unique(groups))
n_splits = min(5, n_groups)
cv = GroupKFold(n_splits=n_splits)
print(f"[cv] GroupKFold n_splits={n_splits}, threshold={threshold:.2f} m, "
      f"n_flights={n_groups}")

reg_models = {
    "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=0),
    "KNN":          Pipeline([("s", StandardScaler()),
                              ("knn", KNeighborsRegressor(n_neighbors=15))]),
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=14,
                                          n_jobs=-1, random_state=0),
}
cls_models = {
    "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=0),
    "KNN":          Pipeline([("s", StandardScaler()),
                              ("knn", KNeighborsClassifier(n_neighbors=15))]),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=14,
                                           n_jobs=-1, random_state=0),
}

reg_rows, cls_rows, fi_by_fs = [], [], {}
for fs_name, cols in FS.items():
    X = f[cols].values
    for name, mdl in reg_models.items():
        pred = cross_val_predict(mdl, X, y_reg, groups=groups, cv=cv, n_jobs=1)
        reg_rows.append(dict(features=fs_name, model=name,
                             MAE=mean_absolute_error(y_reg,pred),
                             RMSE=np.sqrt(mean_squared_error(y_reg,pred)),
                             R2=r2_score(y_reg,pred)))
        print(f"[reg|{fs_name:7s}] {name:12s} MAE={reg_rows[-1]['MAE']:.3f}  "
              f"RMSE={reg_rows[-1]['RMSE']:.3f}  R2={reg_rows[-1]['R2']:+.3f}")
    for name, mdl in cls_models.items():
        pred = cross_val_predict(mdl, X, y_cls, groups=groups, cv=cv, n_jobs=1)
        cls_rows.append(dict(features=fs_name, model=name,
                             Accuracy=accuracy_score(y_cls,pred),
                             Precision=precision_score(y_cls,pred,zero_division=0),
                             Recall=recall_score(y_cls,pred,zero_division=0),
                             F1=f1_score(y_cls,pred,zero_division=0)))
        print(f"[cls|{fs_name:7s}] {name:12s} acc={cls_rows[-1]['Accuracy']:.3f}  "
              f"P={cls_rows[-1]['Precision']:.3f}  R={cls_rows[-1]['Recall']:.3f}  "
              f"F1={cls_rows[-1]['F1']:.3f}")
    # RF feature importance on this feature set
    rf = RandomForestRegressor(n_estimators=300, max_depth=14, n_jobs=-1, random_state=0)
    rf.fit(X, y_reg)
    fi = pd.DataFrame({"feature": cols, "importance": rf.feature_importances_})
    fi_by_fs[fs_name] = fi.sort_values("importance", ascending=False)

reg_df = pd.DataFrame(reg_rows); cls_df = pd.DataFrame(cls_rows)
reg_df.to_csv(os.path.join(TAB_DIR, "metrics_regression.csv"), index=False)
cls_df.to_csv(os.path.join(TAB_DIR, "metrics_classification.csv"), index=False)
for fs, d in fi_by_fs.items():
    d.to_csv(os.path.join(TAB_DIR, f"importance_{fs}.csv"), index=False)

# --------------------------------------------------------------------------
# 9. Figures
# (a) example flight overlay
sample_id = f.flight_id.unique()[0]
sub = f[f.flight_id == sample_id]
fig, ax = plt.subplots(figsize=(6.2, 5.4))
ax.plot(dx, dy, "-", color="black", lw=1.5, label="Desired", zorder=3)
sc = ax.scatter(sub.x, sub.y, c=sub.deviation_m, s=3, cmap="viridis")
plt.colorbar(sc, label="Deviation (m)")
ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
ax.set_aspect("equal"); ax.set_title(f"Sample flight  — {sample_id}")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_sample_flight.pdf")); plt.close()

# (b) deviation distribution
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.hist(f.deviation_m, bins=80, color="steelblue", edgecolor="white")
ax.axvline(threshold, color="red", ls="--", label=f"75th pct = {threshold:.2f} m")
ax.set_xlabel("Cross-track deviation (m)"); ax.set_ylabel("Count")
ax.set_title("Deviation distribution (all 15 BIN-logged flights)")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_deviation_hist.pdf")); plt.close()

# (c) feature importance per feature set
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (fs, imp) in zip(axes, fi_by_fs.items()):
    top = imp.head(10).iloc[::-1]
    ax.barh(top.feature, top.importance, color="darkorange")
    ax.set_title(f"RF importance — {fs}")
    ax.tick_params(axis="y", labelsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_importance.pdf")); plt.close()

# (d) regression comparison
fig, ax = plt.subplots(figsize=(6.5, 3.8))
xs = np.arange(len(reg_models))
w = 0.25
for i, fs in enumerate(FS.keys()):
    vals = [reg_df[(reg_df.features==fs)&(reg_df.model==mn)].MAE.iloc[0]
            for mn in reg_models]
    ax.bar(xs + (i-1)*w, vals, w, label=fs)
ax.set_xticks(xs); ax.set_xticklabels(list(reg_models.keys()))
ax.set_ylabel("MAE (m)  (lower is better)")
ax.set_title("Regression MAE by model and feature set")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_reg_compare.pdf")); plt.close()

# (e) classification F1 comparison
fig, ax = plt.subplots(figsize=(6.5, 3.8))
xs = np.arange(len(cls_models))
for i, fs in enumerate(FS.keys()):
    vals = [cls_df[(cls_df.features==fs)&(cls_df.model==mn)].F1.iloc[0]
            for mn in cls_models]
    ax.bar(xs + (i-1)*w, vals, w, label=fs)
ax.set_xticks(xs); ax.set_xticklabels(list(cls_models.keys()))
ax.set_ylabel("F1 (higher is better)")
ax.set_title("High-deviation classification F1 by model and feature set")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_cls_compare.pdf")); plt.close()

# (f) tilt vs deviation scatter (wind-proxy from onboard data)
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.scatter(f.tilt_mag_deg, f.deviation_m, s=2, alpha=0.2, color="steelblue")
ax.set_xlabel("Hover tilt magnitude (deg)  [onboard wind proxy]")
ax.set_ylabel("Cross-track deviation (m)")
ax.set_title("Onboard tilt magnitude vs deviation")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_tilt_vs_dev.pdf")); plt.close()

# (g) dataset summary
per_flight = (f.groupby("flight_id")
                .agg(N=("deviation_m","size"),
                     mean_dev=("deviation_m","mean"),
                     max_dev=("deviation_m","max"),
                     duration_min=("ts", lambda s: (s.max()-s.min()).total_seconds()/60))
                .reset_index())
per_flight.to_csv(os.path.join(TAB_DIR, "dataset_per_flight.csv"), index=False)
print(per_flight.to_string(index=False))

print("\n[done] artifacts written to", OUT_DIR)
