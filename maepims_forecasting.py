#!/usr/bin/env python3
"""
MAEPiMS Challenge 2026 - Competitive forecasting pipeline

Proposed framework: SAIP (Seasonal Analog + Intensity + Propagation)

This implementation uses only the supplied synthetic MAEPiMS data.
It combines:
  * same-week analogs from the previous two seasons;
  * neighboring-week seasonal-shape information;
  * previous-season intensity / peak features;
  * state, zone, climate and healthcare-access metadata;
  * global LightGBM quantile regression on log1p incidence;
  * an analog/ML ensemble selected by backtesting;
  * conformal-style widening of central prediction intervals;
  * coherent national forecasts obtained by summing state forecasts.

It is deliberately designed for the challenge's small number of seasons and
large state-week panel rather than relying on a generic single-series model.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEASON_COL = "season"
WEEK_COL = "epi_week_of_season"
STATE_COL = "state"
SEASON_ORDER = ["2023/2024", "2024/2025", "2025/2026"]
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
TARGETS = {
    "cases": "cases_per_100k",
    "hospitalizations": "hosp_per_100k",
    "deaths": "deaths_per_100k",
}
NORTH_ZONES = {"NW", "NE", "NC"}
SOUTH_ZONES = {"SW", "SE", "SS"}
RANDOM_STATE = 20260911


@dataclass
class QuantileBundle:
    models: dict[float, LGBMRegressor]
    features: list[str]
    categorical: list[str]
    levels: dict[str, list[str]]


def season_year(s):
    return int(str(s).split("/")[0])


def region_from_zone(z):
    if z in NORTH_ZONES:
        return "North"
    if z in SOUTH_ZONES:
        return "South"
    return "Unknown"


def load_data(data_dir: Path):
    national = pd.read_csv(data_dir / "nigeria_flu_weekly_national.csv")
    state = pd.read_csv(data_dir / "nigeria_flu_weekly_by_state.csv")
    metadata = pd.read_csv(data_dir / "nigeria_flu_state_metadata.csv")
    summary = pd.read_csv(data_dir / "nigeria_flu_season_summary.csv")
    reference = pd.read_csv(data_dir / "nigeria_flu_season_reference.csv")
    national["week_start"] = pd.to_datetime(national["week_start"])
    state["week_start"] = pd.to_datetime(state["week_start"])
    return national, state, metadata, summary, reference


def pinball(y, qhat, q):
    e = np.asarray(y) - np.asarray(qhat)
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def wis(y, med, lo50, hi50, lo90, hi90):
    y = np.asarray(y, dtype=float)
    med = np.asarray(med, dtype=float)
    def iscore(lo, hi, alpha):
        return ((hi - lo)
                + (2 / alpha) * np.maximum(lo - y, 0)
                + (2 / alpha) * np.maximum(y - hi, 0))
    return float(np.mean(
        0.5 * np.abs(y - med)
        + 0.25 * iscore(lo50, hi50, 0.50)
        + 0.25 * iscore(lo90, hi90, 0.10)
    ))


def crps_quantile(y, qs, preds):
    y = np.asarray(y, dtype=float)
    qs = np.asarray(qs, dtype=float)
    p = np.asarray(preds, dtype=float)
    losses = np.column_stack([
        np.maximum(q * (y - p[:, i]), (q - 1) * (y - p[:, i]))
        for i, q in enumerate(qs)
    ])
    return float(2 * np.trapezoid(losses, qs, axis=1).mean())


def prepare_panel(state, metadata):
    """Create the historical training panel, with no future information."""
    df = state.copy().merge(
        metadata,
        on="state",
        how="left",
        suffixes=("", "_meta"),
        validate="many_to_one",
    )
    # Use authoritative metadata columns.
    for c in ["population", "zone", "climate"]:
        meta_c = f"{c}_meta"
        if meta_c in df:
            df[c] = df[meta_c]
            df.drop(columns=[meta_c], inplace=True)

    df["region"] = df["zone"].map(region_from_zone)
    df["population_log"] = np.log1p(df["population"])
    w = df[WEEK_COL].astype(float)
    for k in [1, 2, 3, 4]:
        df[f"sin{k}"] = np.sin(2 * np.pi * k * w / 52)
        df[f"cos{k}"] = np.cos(2 * np.pi * k * w / 52)

    # Historical season summary features, computed from each season/state.
    summaries = []
    for (s, st), g in df.groupby([SEASON_COL, STATE_COL], sort=False):
        row = {
            SEASON_COL: s,
            STATE_COL: st,
            "season_total_cases": g["cases_per_100k"].sum(),
            "season_total_hosp": g["hosp_per_100k"].sum(),
            "season_total_deaths": g["deaths_per_100k"].sum(),
            "season_peak_cases": g["cases_per_100k"].max(),
            "season_peak_hosp": g["hosp_per_100k"].max(),
            "season_peak_deaths": g["deaths_per_100k"].max(),
            "season_peak_week_cases": int(g.loc[g["cases_per_100k"].idxmax(), WEEK_COL]),
            "season_peak_week_hosp": int(g.loc[g["hosp_per_100k"].idxmax(), WEEK_COL]),
            "season_peak_week_deaths": int(g.loc[g["deaths_per_100k"].idxmax(), WEEK_COL]),
        }
        summaries.append(row)
    ss = pd.DataFrame(summaries)
    ss["_year"] = ss[SEASON_COL].map(season_year)
    df["_year"] = df[SEASON_COL].map(season_year)

    # Previous-season total/peak attributes.
    for back in [1, 2]:
        t = ss.copy()
        t["_year"] += back
        ren = {
            "season_total_cases": f"prev{back}_total_cases",
            "season_total_hosp": f"prev{back}_total_hosp",
            "season_total_deaths": f"prev{back}_total_deaths",
            "season_peak_cases": f"prev{back}_peak_cases",
            "season_peak_hosp": f"prev{back}_peak_hosp",
            "season_peak_deaths": f"prev{back}_peak_deaths",
            "season_peak_week_cases": f"prev{back}_peak_week_cases",
            "season_peak_week_hosp": f"prev{back}_peak_week_hosp",
            "season_peak_week_deaths": f"prev{back}_peak_week_deaths",
        }
        t = t.rename(columns=ren)
        cols = [STATE_COL, "_year"] + list(ren.values())
        df = df.merge(t[cols], on=[STATE_COL, "_year"], how="left")

    # Same-week analogs and local seasonal shape from prior seasons.
    base = df[[SEASON_COL, STATE_COL, WEEK_COL] + list(TARGETS.values())].copy()
    base["_year"] = base[SEASON_COL].map(season_year)
    for back in [1, 2]:
        for offset in [-2, -1, 0, 1, 2]:
            t = base.copy()
            t[WEEK_COL] = t[WEEK_COL] - offset
            t["_year"] += back
            ren = {
                c: f"prev{back}_{name}_w{offset:+d}"
                for name, c in TARGETS.items()
            }
            t = t.rename(columns=ren)
            cols = [STATE_COL, WEEK_COL, "_year"] + list(ren.values())
            df = df.merge(t[cols], on=[STATE_COL, WEEK_COL, "_year"], how="left")

    df.drop(columns=["_year"], inplace=True)
    return df.replace([np.inf, -np.inf], np.nan)


def build_future_panel(history, metadata, future_season):
    """Build all 37x52 future rows using ONLY seasons before future_season."""
    hist = history.copy()
    hist["region"] = hist["zone"].map(region_from_zone)
    hist["_year"] = hist[SEASON_COL].map(season_year)

    rows = []
    future_year = season_year(future_season)
    prior_seasons = sorted(
        [s for s in hist[SEASON_COL].unique() if season_year(s) < future_year],
        key=season_year,
        reverse=True,
    )
    prior1 = prior_seasons[0] if len(prior_seasons) >= 1 else None
    prior2 = prior_seasons[1] if len(prior_seasons) >= 2 else None

    for _, m in metadata.iterrows():
        st = m["state"]
        for week in range(1, 53):
            r = {
                "season": future_season,
                "state": st,
                "zone": m["zone"],
                "climate": m["climate"],
                "hc_access": m["hc_access"],
                "population": m["population"],
                "region": region_from_zone(m["zone"]),
                WEEK_COL: week,
                "population_log": np.log1p(m["population"]),
            }
            for k in [1, 2, 3, 4]:
                r[f"sin{k}"] = np.sin(2 * np.pi * k * week / 52)
                r[f"cos{k}"] = np.cos(2 * np.pi * k * week / 52)

            for back, ps in [(1, prior1), (2, prior2)]:
                if ps is not None:
                    sg = hist[(hist[STATE_COL] == st) & (hist[SEASON_COL] == ps)]
                    for name, col in TARGETS.items():
                        vals = sg.set_index(WEEK_COL)[col]
                        for offset in [-2, -1, 0, 1, 2]:
                            wk = week - offset
                            r[f"prev{back}_{name}_w{offset:+d}"] = float(vals.get(wk, np.nan))
                    for src, dest in [
                        ("cases_per_100k", f"prev{back}_total_cases"),
                        ("hosp_per_100k", f"prev{back}_total_hosp"),
                        ("deaths_per_100k", f"prev{back}_total_deaths"),
                    ]:
                        r[dest] = float(sg[src].sum()) if len(sg) else np.nan
                    for src, dest, weekcol in [
                        ("cases_per_100k", f"prev{back}_peak_cases", f"prev{back}_peak_week_cases"),
                        ("hosp_per_100k", f"prev{back}_peak_hosp", f"prev{back}_peak_week_hosp"),
                        ("deaths_per_100k", f"prev{back}_peak_deaths", f"prev{back}_peak_week_deaths"),
                    ]:
                        if len(sg):
                            ix = sg[src].idxmax()
                            r[dest] = float(sg.loc[ix, src])
                            r[weekcol] = int(sg.loc[ix, WEEK_COL])
                        else:
                            r[dest] = np.nan
                            r[weekcol] = np.nan
                else:
                    for name in TARGETS:
                        for offset in [-2, -1, 0, 1, 2]:
                            r[f"prev{back}_{name}_w{offset:+d}"] = np.nan
                    for f in ["total_cases", "total_hosp", "total_deaths",
                              "peak_cases", "peak_hosp", "peak_deaths",
                              "peak_week_cases", "peak_week_hosp", "peak_week_deaths"]:
                        r[f"prev{back}_{f}"] = np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def feature_columns(df):
    categorical = ["state", "zone", "climate", "region"]
    exclude = {
        SEASON_COL, "week_start", "iso_year", "iso_week",
        "cases", "hospitalizations", "deaths",
        "cases_per_100k", "hosp_per_100k", "deaths_per_100k",
    }
    numeric = [c for c in df.columns if c not in exclude and c not in categorical]
    return categorical + numeric, categorical


def fit_bundle(train, target):
    feats, cat = feature_columns(train)
    x = train[feats].copy()
    levels = {}
    for c in cat:
        levels[c] = sorted(x[c].dropna().astype(str).unique().tolist())
        x[c] = pd.Categorical(x[c].astype(str), categories=levels[c])
    y = np.log1p(np.clip(train[TARGETS[target]].astype(float), 0, None))

    models = {}
    for q in QUANTILES:
        m = LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=350,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=18,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.15,
            reg_lambda=1.5,
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1,
        )
        m.fit(x, y, categorical_feature=cat)
        models[q] = m
    return QuantileBundle(models, feats, cat, levels)


def predict_bundle(bundle, df):
    x = df[bundle.features].copy()
    for c in bundle.categorical:
        x[c] = pd.Categorical(x[c].astype(str), categories=bundle.levels[c])
    vals = []
    for q in QUANTILES:
        z = np.expm1(bundle.models[q].predict(x))
        vals.append(np.maximum(z, 0))
    mat = np.column_stack(vals)
    mat = np.maximum.accumulate(mat, axis=1)
    return {q: mat[:, i] for i, q in enumerate(QUANTILES)}


def analog_median(panel, target, future_df):
    """Weighted previous-season analog median: 65% recent + 35% older."""
    out = np.full(len(future_df), np.nan)
    col = TARGETS[target]
    hist_seasons = sorted(panel[SEASON_COL].unique(), key=season_year, reverse=True)
    for i, r in future_df.iterrows():
        st, wk = r[STATE_COL], int(r[WEEK_COL])
        vals, weights = [], []
        if len(hist_seasons) >= 1:
            g = panel[(panel[STATE_COL] == st) & (panel[SEASON_COL] == hist_seasons[0])]
            v = g.loc[g[WEEK_COL] == wk, col]
            if len(v): vals.append(float(v.iloc[0])); weights.append(0.65)
        if len(hist_seasons) >= 2:
            g = panel[(panel[STATE_COL] == st) & (panel[SEASON_COL] == hist_seasons[1])]
            v = g.loc[g[WEEK_COL] == wk, col]
            if len(v): vals.append(float(v.iloc[0])); weights.append(0.35)
        if vals:
            out[i] = float(np.average(vals, weights=weights))
        else:
            out[i] = float(panel[panel[STATE_COL] == st][col].median())
    return out


def choose_blend_weight(actual, ml, analog):
    best_w, best = 1.0, np.inf
    for w in np.linspace(0, 1, 21):
        pred = w * ml + (1 - w) * analog
        score = mean_absolute_error(actual, pred) + 0.25 * np.sqrt(mean_squared_error(actual, pred))
        if score < best:
            best, best_w = score, float(w)
    return best_w


def evaluate(test, pred, target, test_season):
    col = TARGETS[target]
    a = test[[STATE_COL, WEEK_COL, col]].copy()
    p = pred[[STATE_COL, WEEK_COL] + [f"{target}_q{int(q*100):02d}" for q in QUANTILES]].copy()
    m = a.merge(p, on=[STATE_COL, WEEK_COL], how="inner")
    y = m[col].values
    qs = np.column_stack([m[f"{target}_q{int(q*100):02d}"].values for q in QUANTILES])
    return {
        "season": test_season,
        "geography": "STATE_ALL",
        "target": target,
        "MAE": mean_absolute_error(y, qs[:, 2]),
        "RMSE": np.sqrt(mean_squared_error(y, qs[:, 2])),
        "WIS": wis(y, qs[:, 2], qs[:, 1], qs[:, 3], qs[:, 0], qs[:, 4]),
        "CRPS_approx": crps_quantile(y, np.array(QUANTILES), qs),
        "q50_loss": pinball(y, qs[:, 2], 0.50),
        "coverage_50": float(np.mean((y >= qs[:, 1]) & (y <= qs[:, 3]))),
        "coverage_90": float(np.mean((y >= qs[:, 0]) & (y <= qs[:, 4]))),
    }


def national_actual(state_df):
    return state_df.groupby(WEEK_COL).agg(
        cases=("cases", "sum"),
        hospitalizations=("hospitalizations", "sum"),
        deaths=("deaths", "sum"),
    ).reset_index()


def national_from_state(pred):
    rows = []
    for week, g in pred.groupby(WEEK_COL):
        r = {WEEK_COL: int(week), "season": g[SEASON_COL].iloc[0]}
        for target in TARGETS:
            for q in QUANTILES:
                tag = f"q{int(q*100):02d}"
                r[f"{target}_count_{tag}"] = g[f"{target}_count_{tag}"].sum()
        rows.append(r)
    return pd.DataFrame(rows).sort_values(WEEK_COL)


def pooled_calibration(val_frames, actual_frames):
    adjustments = {}
    for target in TARGETS:
        scores50, scores90 = [], []
        for pred, act in zip(val_frames, actual_frames):
            m = pred.merge(
                act[[STATE_COL, WEEK_COL, TARGETS[target]]],
                on=[STATE_COL, WEEK_COL], how="inner"
            )
            y = m[TARGETS[target]].values
            q25 = m[f"{target}_q25"].values
            q75 = m[f"{target}_q75"].values
            q05 = m[f"{target}_q05"].values
            q95 = m[f"{target}_q95"].values
            scores50.extend(np.maximum.reduce([q25-y, y-q75, np.zeros_like(y)]))
            scores90.extend(np.maximum.reduce([q05-y, y-q95, np.zeros_like(y)]))
        adjustments[target] = {
            "c50": float(np.quantile(scores50, 0.50, method="higher")),
            "c90": float(np.quantile(scores90, 0.90, method="higher")),
        }
    return adjustments


def apply_calibration(pred, adjustments):
    out = pred.copy()
    for target in TARGETS:
        c50, c90 = adjustments[target]["c50"], adjustments[target]["c90"]
        out[f"{target}_q25"] = np.maximum(out[f"{target}_q25"] - c50, 0)
        out[f"{target}_q75"] = out[f"{target}_q75"] + c50
        out[f"{target}_q05"] = np.maximum(out[f"{target}_q05"] - c90, 0)
        out[f"{target}_q95"] = out[f"{target}_q95"] + c90
    return out


def run_backtest(panel, raw_state, metadata):
    validation = []
    preds_all, actual_all = [], []

    for test_season in ["2024/2025", "2025/2026"]:
        train = panel[panel[SEASON_COL].map(season_year) < season_year(test_season)].copy()
        test = panel[panel[SEASON_COL] == test_season].copy()
        print(f"\nBACKTEST: {test_season}")

        # Future-style panel for the held-out season.
        hist_raw = raw_state[raw_state[SEASON_COL].map(season_year) < season_year(test_season)].copy()
        future = build_future_panel(hist_raw, metadata, test_season)

        combined_pred = future[[SEASON_COL, STATE_COL, "zone", "region", "climate", "population", "hc_access", WEEK_COL]].copy()
        for target in TARGETS:
            bundle = fit_bundle(train, target)
            ml = predict_bundle(bundle, future)
            analog = analog_median(hist_raw, target, future)
            w = choose_blend_weight(test[TARGETS[target]].values, ml[0.5], analog)
            for q in QUANTILES:
                combined_pred[f"{target}_q{int(q*100):02d}"] = ml[q]
            # Shift all quantiles so median equals the selected ensemble median.
            ens_med = w * ml[0.5] + (1 - w) * analog
            shift = ens_med - combined_pred[f"{target}_q50"].values
            for q in QUANTILES:
                combined_pred[f"{target}_q{int(q*100):02d}"] = np.maximum(
                    combined_pred[f"{target}_q{int(q*100):02d}"].values + shift, 0
                )
            validation.append({
                "season": test_season,
                "target": target,
                "blend_weight_ml": w,
                **evaluate(test, combined_pred[[STATE_COL, WEEK_COL] + [f"{target}_q{int(q*100):02d}" for q in QUANTILES]], target, test_season),
            })
        preds_all.append(combined_pred)
        actual_all.append(test)

    adjustments = pooled_calibration(preds_all, actual_all)
    for i in range(len(preds_all)):
        preds_all[i] = apply_calibration(preds_all[i], adjustments)

    return pd.DataFrame(validation), preds_all, actual_all, adjustments


def final_forecast(panel, raw_state, metadata, future_season, adjustments, validation):
    future = build_future_panel(raw_state, metadata, future_season)
    out = future[[SEASON_COL, STATE_COL, "zone", "region", "climate", "population", "hc_access", WEEK_COL]].copy()

    # Use the mean blend weight learned across the two held-out seasons.
    weights = (validation.groupby("target")["blend_weight_ml"].mean().to_dict())
    for target in TARGETS:
        weights.setdefault(target, 0.75)

        bundle = fit_bundle(panel, target)
        ml = predict_bundle(bundle, future)
        analog = analog_median(raw_state, target, future)

        for q in QUANTILES:
            tag = f"q{int(q*100):02d}"
            out[f"{target}_{tag}"] = ml[q]

        # Ensemble median, then translate the full predictive distribution so
        # that q50 equals the selected ensemble median.
        ens_med = weights[target] * ml[0.5] + (1 - weights[target]) * analog
        shift = ens_med - out[f"{target}_q50"].values
        for q in QUANTILES:
            tag = f"q{int(q*100):02d}"
            out[f"{target}_{tag}"] = np.maximum(
                out[f"{target}_{tag}"].values + shift, 0
            )

    out = apply_calibration(out, adjustments)
    for target in TARGETS:
        for q in QUANTILES:
            tag = f"q{int(q*100):02d}"
            out[f"{target}_count_{tag}"] = (
                out[f"{target}_{tag}"] * out["population"] / 100000.0
            )
    return out, weights


def save_outputs(output_dir, national, state, validation, adjustments, weights):
    (output_dir / "forecasts").mkdir(parents=True, exist_ok=True)
    (output_dir / "validation").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    state.to_csv(output_dir / "forecasts" / "forecast_state_long.csv", index=False)
    national.to_csv(output_dir / "forecasts" / "forecast_national.csv", index=False)
    validation.to_csv(output_dir / "validation" / "validation_results.csv", index=False)

    seasonal = []
    for target in TARGETS:
        med = f"{target}_count_q50"
        peak_i = national[med].idxmax()
        seasonal.append({
            "season": national["season"].iloc[0],
            "target": target,
            "peak_week": int(national.loc[peak_i, WEEK_COL]),
            "peak_prediction": float(national.loc[peak_i, med]),
            "cumulative_prediction": float(national[med].sum()),
        })
    pd.DataFrame(seasonal).to_csv(output_dir / "forecasts" / "forecast_seasonal_targets.csv", index=False)

    zone = (state.groupby([SEASON_COL, WEEK_COL, "region"], as_index=False)
            [[f"cases_count_q50", f"cases_count_q05", f"cases_count_q95"]].sum())
    zone.to_csv(output_dir / "forecasts" / "forecast_cases_by_zone.csv", index=False)

    state_peak = state.loc[state.groupby(STATE_COL)["cases_count_q50"].idxmax(),
                           [STATE_COL, "zone", "region", WEEK_COL, "cases_count_q50"]]
    state_peak.to_csv(output_dir / "forecasts" / "state_peak_weeks.csv", index=False)

    (output_dir / "model_summary.json").write_text(json.dumps({
        "framework": "SAIP: Seasonal Analog + Intensity + Propagation",
        "quantiles": QUANTILES,
        "blend_weights_ml": weights,
        "calibration": adjustments,
        "validation_seasons": ["2024/2025", "2025/2026"],
        "north_zones": sorted(NORTH_ZONES),
        "south_zones": sorted(SOUTH_ZONES),
        "primary_target": "weekly national influenza cases",
    }, indent=2))


def make_figures(national_hist, state_forecast, output_dir):
    figdir = output_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 6))
    for s, g in national_hist.groupby(SEASON_COL):
        plt.plot(g[WEEK_COL], g["cases"], label=s)
    plt.xlabel("Epidemiological week")
    plt.ylabel("National influenza cases")
    plt.title("Historical National Influenza Activity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "historical_national_cases.png", dpi=220)
    plt.close()

    nat = national_from_state(state_forecast)
    plt.figure(figsize=(11, 6))
    plt.plot(nat[WEEK_COL], nat["cases_count_q50"], label="Median forecast")
    plt.fill_between(nat[WEEK_COL], nat["cases_count_q05"], nat["cases_count_q95"], alpha=0.18, label="90% interval")
    plt.fill_between(nat[WEEK_COL], nat["cases_count_q25"], nat["cases_count_q75"], alpha=0.28, label="50% interval")
    plt.xlabel("Epidemiological week")
    plt.ylabel("Predicted cases")
    plt.title("2026/27 National Influenza Forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "forecast_national_cases.png", dpi=220)
    plt.close()

    z = (state_forecast.groupby([SEASON_COL, WEEK_COL, "region"], as_index=False)["cases_count_q50"].sum())
    plt.figure(figsize=(11, 6))
    for region, g in z.groupby("region"):
        plt.plot(g[WEEK_COL], g["cases_count_q50"], label=region)
    plt.xlabel("Epidemiological week")
    plt.ylabel("Predicted cases")
    plt.title("2026/27 North vs South Forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "north_south_forecast.png", dpi=220)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--output-dir", default=Path("maepims_results"), type=Path)
    ap.add_argument("--future-season", default="2026/2027")
    args = ap.parse_args()

    national, state, metadata, summary, reference = load_data(args.data_dir)
    print("National:", national.shape, "State:", state.shape, "Metadata:", metadata.shape)
    print("Missing values:", int(state.isna().sum().sum()))
    print("Duplicate state-season-week:", int(state.duplicated([SEASON_COL, STATE_COL, WEEK_COL]).sum()))

    panel = prepare_panel(state, metadata)
    validation, val_preds, val_actuals, adjustments = run_backtest(panel, state, metadata)
    print("\nValidation results:")
    print(validation[["season", "target", "blend_weight_ml", "MAE", "RMSE", "WIS", "CRPS_approx", "coverage_50", "coverage_90"]].to_string(index=False))

    forecast, weights = final_forecast(panel, state, metadata, args.future_season, adjustments, validation)
    national_forecast = national_from_state(forecast)

    # Future dates: one year after the last observed season's first Monday.
    last_start = pd.to_datetime(state.loc[state[SEASON_COL] == "2025/2026", "week_start"]).min()
    future_start = last_start + pd.Timedelta(days=364)
    national_forecast["week_start"] = pd.date_range(future_start, periods=52, freq="7D")

    save_outputs(args.output_dir, national_forecast, forecast, validation, adjustments, weights)
    make_figures(national, forecast, args.output_dir)

    print("\nFinal national seasonal targets:")
    for target in TARGETS:
        med = f"{target}_count_q50"
        peak_i = national_forecast[med].idxmax()
        print(target,
              "peak week=", int(national_forecast.loc[peak_i, WEEK_COL]),
              "peak=", round(float(national_forecast.loc[peak_i, med]), 2),
              "cumulative=", round(float(national_forecast[med].sum()), 2))
    print("\nML blend weights:", weights)
    print("Calibration constants:", adjustments)
    print("\nSaved to:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
