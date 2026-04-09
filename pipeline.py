"""
UAV Flight Path Deviation Pipeline
==================================
Loads all 13 UAV flight logs, computes per-second cross-track deviation
against the desired KML survey path, joins Open-Meteo historical hourly
weather, engineers physics-aware features, and trains DT/KNN/RF models
for both regression (predict deviation magnitude) and classification
(predict whether deviation exceeds the 75th percentile).

Saves: results/metrics.csv, results/feature_importance.csv,
       results/*.png, results/merged.parquet (if pyarrow available else csv)
"""
import os, re, glob, json, urllib.request, urllib.parse
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ----------------------------------------------------------------------
PROJECT = "/Users/abdulazizal-haidary/development/Machine Learning/Project"
DATA = os.path.join(PROJECT, "Data", "extracted")
FLIGHTS_DIR = os.path.join(DATA, "Flight_path_for_UAV_correction")
DESIRED_MASTER = os.path.join(FLIGHTS_DIR,
                              "Desired_flight_path_soil_moisture_Research_plot.kml")
RESULTS = os.path.join(PROJECT, "results")
os.makedirs(RESULTS, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Parse the desired flight path
def parse_kml_waypoints(path):
    tree = ET.parse(path)
    root = tree.getroot()
    pts = []
    for coord in root.iter("{http://www.opengis.net/kml/2.2}coordinates"):
        for token in coord.text.strip().split():
            parts = token.split(",")
            if len(parts) >= 2:
                lon, lat = float(parts[0]), float(parts[1])
                pts.append((lat, lon))
    return pd.DataFrame(pts, columns=["lat", "lon"])

desired = parse_kml_waypoints(DESIRED_MASTER)
print(f"[desired] {len(desired)} waypoints")

# Local equirectangular projection centered on plot
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
    """Min perpendicular distance from each (px,py) to nearest segment of polyline."""
    px = np.asarray(px); py = np.asarray(py)
    min_d = np.full(px.shape, np.inf)
    for i in range(len(sx) - 1):
        x1, y1, x2, y2 = sx[i], sy[i], sx[i + 1], sy[i + 1]
        vx, vy = x2 - x1, y2 - y1
        L2 = vx * vx + vy * vy
        if L2 == 0:
            d = np.hypot(px - x1, py - y1)
        else:
            t = ((px - x1) * vx + (py - y1) * vy) / L2
            t = np.clip(t, 0, 1)
            cx = x1 + t * vx
            cy = y1 + t * vy
            d = np.hypot(px - cx, py - cy)
        min_d = np.minimum(min_d, d)
    return min_d

# ----------------------------------------------------------------------
# 2. Parse all flight logs
def parse_flight_filename(fn):
    """Extract takeoff (HH, MM) and date (mo, d, yy) from a flight log filename."""
    base = os.path.basename(fn)
    m = re.search(r"(\d{1,2})_(\d{2})_.*?(\d{1,2})_(\d{1,2})_(\d{2})\.txt$", base)
    if not m:
        return None
    hh, mm, mo, d, yy = map(int, m.groups())
    cam = "IR" if re.search(r"IR", base) else "L"
    return dict(hh=hh, mm=mm, mo=mo, d=d, yy=yy, camera=cam, basename=base)

def load_flight(path, desired_lat, desired_lon, pad=0.01):
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    lat_min, lat_max = desired_lat.min() - pad, desired_lat.max() + pad
    lon_min, lon_max = desired_lon.min() - pad, desired_lon.max() + pad
    df = df[(df.Latitude.between(lat_min, lat_max)) &
            (df.Longitude.between(lon_min, lon_max))].copy()
    if df.empty:
        return None
    meta = parse_flight_filename(path)
    if meta is None:
        return None
    anchor = pd.Timestamp(year=2000 + meta["yy"], month=meta["mo"], day=meta["d"],
                          hour=meta["hh"], minute=meta["mm"], tz="US/Eastern")
    df["ts"] = anchor + pd.to_timedelta(df["Time"] - df["Time"].iloc[0], unit="s")
    df["camera"] = meta["camera"]
    df["flight_id"] = meta["basename"]
    return df

flight_files = sorted(glob.glob(os.path.join(FLIGHTS_DIR, "Flight*", "*.txt")))
print(f"[flights] {len(flight_files)} log files found")

dfs = []
for f in flight_files:
    d = load_flight(f, desired.lat.values, desired.lon.values)
    if d is not None and len(d) > 30:
        dfs.append(d)
print(f"[flights] kept {len(dfs)} valid flights")

flights = pd.concat(dfs, ignore_index=True)
ax_, ay_ = to_xy(flights.Latitude.values, flights.Longitude.values)
flights["x"] = ax_; flights["y"] = ay_
flights["deviation_m"] = cross_track(ax_, ay_, dx, dy)
print(f"[deviation] {len(flights):,} points; "
      f"mean={flights.deviation_m.mean():.2f} m, "
      f"max={flights.deviation_m.max():.2f} m")

# Drone heading from consecutive points (degrees, 0=North)
flights = flights.sort_values(["flight_id", "ts"]).reset_index(drop=True)
flights["dx_step"] = flights.groupby("flight_id")["x"].diff()
flights["dy_step"] = flights.groupby("flight_id")["y"].diff()
flights["heading_deg"] = (np.degrees(np.arctan2(flights.dx_step, flights.dy_step))
                          + 360) % 360
flights["ground_speed_ms"] = np.hypot(flights.dx_step, flights.dy_step)  # 1 Hz logs

# ----------------------------------------------------------------------
# 3. Fetch Open-Meteo historical hourly weather
def fetch_open_meteo(lat, lon, start, end):
    base = "https://archive-api.open-meteo.com/v1/archive"
    params = dict(latitude=lat, longitude=lon, start_date=start, end_date=end,
                  hourly="wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
                         "temperature_2m,relative_humidity_2m,surface_pressure",
                  timezone="America/New_York", wind_speed_unit="ms")
    url = base + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as r:
        js = json.loads(r.read().decode())
    h = js["hourly"]
    wx = pd.DataFrame(h)
    wx["ts"] = pd.to_datetime(wx["time"]).dt.tz_localize("US/Eastern",
                                                         nonexistent="shift_forward",
                                                         ambiguous="NaT")
    wx = wx.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return wx

start_date = flights.ts.min().date().isoformat()
end_date = flights.ts.max().date().isoformat()
print(f"[weather] fetching Open-Meteo {start_date} -> {end_date}")
wx = fetch_open_meteo(desired.lat.mean(), desired.lon.mean(), start_date, end_date)
print(f"[weather] {len(wx)} hourly records")

# ----------------------------------------------------------------------
# 4. asof-merge weather onto flight points (nearest hour)
flights = flights.sort_values("ts").reset_index(drop=True)
merged = pd.merge_asof(flights, wx[["ts", "wind_speed_10m", "wind_direction_10m",
                                    "wind_gusts_10m", "temperature_2m",
                                    "relative_humidity_2m", "surface_pressure"]],
                       on="ts", direction="nearest",
                       tolerance=pd.Timedelta("90min"))
print(f"[merge] joined rows: {len(merged):,}; with weather: {merged.wind_speed_10m.notna().sum():,}")

# ----------------------------------------------------------------------
# 5. Feature engineering
m = merged.copy()
# Wind direction sin/cos (cyclical)
m["wind_dir_rad"] = np.deg2rad(m["wind_direction_10m"])
m["wind_sin"] = np.sin(m["wind_dir_rad"])
m["wind_cos"] = np.cos(m["wind_dir_rad"])
# Heading-relative wind angle (0 = headwind, 180 = tailwind, 90 = crosswind)
m["heading_rad"] = np.deg2rad(m["heading_deg"])
rel = (m["wind_dir_rad"] - m["heading_rad"] + np.pi) % (2 * np.pi) - np.pi
m["rel_wind_abs_deg"] = np.degrees(np.abs(rel))
m["crosswind_component"] = m["wind_speed_10m"] * np.sin(rel).abs()
m["headwind_component"] = m["wind_speed_10m"] * np.cos(rel)
# Rolling features per flight (10-second window)
m = m.sort_values(["flight_id", "ts"])
g = m.groupby("flight_id")
m["ws_roll_mean_10s"] = g["wind_speed_10m"].transform(lambda s: s.rolling(10, min_periods=1).mean())
m["gust_roll_max_10s"] = g["wind_gusts_10m"].transform(lambda s: s.rolling(10, min_periods=1).max())
m["dev_cum"] = g["deviation_m"].transform(lambda s: s.expanding().mean())

# Drop rows with missing essentials
feature_cols = ["wind_speed_10m", "wind_gusts_10m", "wind_sin", "wind_cos",
                "rel_wind_abs_deg", "crosswind_component", "headwind_component",
                "ws_roll_mean_10s", "gust_roll_max_10s",
                "temperature_2m", "relative_humidity_2m", "surface_pressure",
                "ground_speed_ms", "Altitude"]
m = m.dropna(subset=feature_cols + ["deviation_m"]).reset_index(drop=True)
m["camera_ir"] = (m["camera"] == "IR").astype(int)
feature_cols.append("camera_ir")

print(f"[features] final dataset: {len(m):,} rows × {len(feature_cols)} features, "
      f"{m.flight_id.nunique()} flights")

# Save full merged dataset
out_csv = os.path.join(RESULTS, "merged_dataset.csv")
m.to_csv(out_csv, index=False)
print(f"[save] {out_csv}")

# ----------------------------------------------------------------------
# 6. Modeling — GroupKFold by flight_id (no leakage between flights)
X = m[feature_cols].values
y_reg = m["deviation_m"].values
threshold = np.quantile(y_reg, 0.75)
y_cls = (y_reg >= threshold).astype(int)
groups = m["flight_id"].values
n_groups = len(np.unique(groups))
n_splits = min(5, n_groups)
cv = GroupKFold(n_splits=n_splits)
print(f"[cv] GroupKFold n_splits={n_splits}, threshold={threshold:.2f} m")

reg_models = {
    "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=0),
    "KNN":          Pipeline([("scaler", StandardScaler()),
                              ("knn", KNeighborsRegressor(n_neighbors=15))]),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=12,
                                          n_jobs=-1, random_state=0),
}
cls_models = {
    "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=0),
    "KNN":          Pipeline([("scaler", StandardScaler()),
                              ("knn", KNeighborsClassifier(n_neighbors=15))]),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=12,
                                           n_jobs=-1, random_state=0),
}

# Two feature sets:
#   ALL  = drone state + wind   (upper bound on what we can learn)
#   WIND = wind features only   (isolates wind-only signal — the actual research question)
wind_only_cols = ["wind_speed_10m", "wind_gusts_10m", "wind_sin", "wind_cos",
                  "rel_wind_abs_deg", "crosswind_component", "headwind_component",
                  "ws_roll_mean_10s", "gust_roll_max_10s",
                  "temperature_2m", "relative_humidity_2m", "surface_pressure",
                  "camera_ir"]
feature_sets = {"ALL": feature_cols, "WIND_ONLY": wind_only_cols}

reg_rows, cls_rows = [], []

for fs_name, cols in feature_sets.items():
    Xfs = m[cols].values
    for name, mdl in reg_models.items():
        pred = cross_val_predict(mdl, Xfs, y_reg, groups=groups, cv=cv, n_jobs=1)
        reg_rows.append({
            "features": fs_name, "model": name,
            "MAE":  mean_absolute_error(y_reg, pred),
            "RMSE": np.sqrt(mean_squared_error(y_reg, pred)),
            "R2":   r2_score(y_reg, pred),
        })
        print(f"[reg|{fs_name}] {name}: MAE={reg_rows[-1]['MAE']:.3f}  "
              f"RMSE={reg_rows[-1]['RMSE']:.3f}  R2={reg_rows[-1]['R2']:.3f}")

    for name, mdl in cls_models.items():
        pred = cross_val_predict(mdl, Xfs, y_cls, groups=groups, cv=cv, n_jobs=1)
        cls_rows.append({
            "features": fs_name, "model": name,
            "Accuracy":  accuracy_score(y_cls, pred),
            "Precision": precision_score(y_cls, pred, zero_division=0),
            "Recall":    recall_score(y_cls, pred, zero_division=0),
            "F1":        f1_score(y_cls, pred, zero_division=0),
        })
        print(f"[cls|{fs_name}] {name}: acc={cls_rows[-1]['Accuracy']:.3f}  "
              f"P={cls_rows[-1]['Precision']:.3f}  "
              f"R={cls_rows[-1]['Recall']:.3f}  F1={cls_rows[-1]['F1']:.3f}")

reg_df = pd.DataFrame(reg_rows)
cls_df = pd.DataFrame(cls_rows)
reg_df.to_csv(os.path.join(RESULTS, "metrics_regression.csv"), index=False)
cls_df.to_csv(os.path.join(RESULTS, "metrics_classification.csv"), index=False)

# Feature importance from Random Forest (fit on full data)
rf = RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=0)
rf.fit(X, y_reg)
imp = pd.DataFrame({"feature": feature_cols, "importance": rf.feature_importances_})
imp = imp.sort_values("importance", ascending=False)
imp.to_csv(os.path.join(RESULTS, "feature_importance.csv"), index=False)
print("[importance]")
print(imp.to_string(index=False))

# ----------------------------------------------------------------------
# 7. Figures
# (a) example flight: actual vs desired colored by deviation
sample_id = m.flight_id.unique()[0]
sub = m[m.flight_id == sample_id]
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(dx, dy, "-", color="black", lw=1.5, label="Desired path", zorder=3)
sc = ax.scatter(sub.x, sub.y, c=sub.deviation_m, s=4, cmap="viridis")
plt.colorbar(sc, label="Deviation (m)")
ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
ax.set_aspect("equal")
ax.set_title(f"Sample flight: {sample_id}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "fig_sample_flight.png"), dpi=140)
plt.close()

# (b) deviation distribution
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(m.deviation_m, bins=60, color="steelblue", edgecolor="white")
ax.axvline(threshold, color="red", linestyle="--",
           label=f"75th pct = {threshold:.2f} m")
ax.set_xlabel("Cross-track deviation (m)"); ax.set_ylabel("count")
ax.set_title("Per-second deviation distribution (all flights)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "fig_deviation_hist.png"), dpi=140)
plt.close()

# (c) feature importance bar
fig, ax = plt.subplots(figsize=(7, 5))
imp_top = imp.head(12).iloc[::-1]
ax.barh(imp_top.feature, imp_top.importance, color="darkorange")
ax.set_xlabel("Random Forest feature importance")
ax.set_title("Top features (regression target = deviation_m)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "fig_importance.png"), dpi=140)
plt.close()

# (d) wind speed vs deviation scatter
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(m.wind_speed_10m, m.deviation_m, s=2, alpha=0.25, color="steelblue")
ax.set_xlabel("Wind speed (m/s, Open-Meteo)")
ax.set_ylabel("Deviation (m)")
ax.set_title("Wind speed vs cross-track deviation")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "fig_wind_vs_dev.png"), dpi=140)
plt.close()

# (e) regression model comparison bar — grouped by feature set
fig, ax = plt.subplots(figsize=(7, 4))
models_order = list(reg_models.keys())
xs = np.arange(len(models_order)); w = 0.35
mae_all  = [reg_df[(reg_df.features=="ALL")  & (reg_df.model==mn)].MAE.iloc[0] for mn in models_order]
mae_wind = [reg_df[(reg_df.features=="WIND_ONLY") & (reg_df.model==mn)].MAE.iloc[0] for mn in models_order]
ax.bar(xs - w/2, mae_all,  w, label="ALL features",     color="steelblue")
ax.bar(xs + w/2, mae_wind, w, label="WIND_ONLY", color="darkorange")
ax.set_xticks(xs); ax.set_xticklabels(models_order)
ax.set_ylabel("MAE (m)")
ax.set_title("Regression MAE — drone+wind vs wind only")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "fig_reg_compare.png"), dpi=140)
plt.close()

print("\n[done] all results in", RESULTS)
