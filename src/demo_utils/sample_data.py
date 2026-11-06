import os

import numpy as np
import pandas as pd

REGIONS = ["North America", "Europe", "East Asia", "South Asia", "LATAM"]

HEADLINES = {
    "strike": ["Dockworkers in {r} vote to strike at major container ports",
               "Port labor talks collapse in {r}; walkout expected next week"],
    "weather": ["Typhoon warning issued for shipping routes near {r}",
                "Record flooding disrupts rail and trucking across {r}"],
    "policy": ["{r} announces surprise export restrictions on key components",
               "New tariffs on inbound goods take effect in {r}"],
    "decoy": ["Analysts speculate about possible port strike in {r}; unions deny any plans",
              "Rumors of new {r} tariffs dismissed by trade ministry"],
    "calm": ["Freight volumes steady across {r} as ports run on schedule",
             "{r} logistics index flat; carriers report normal transit times",
             "New warehouse opens in {r}, adding regional capacity"],
}


def make_news(months, rng):
    """One headline per region per month; ~25% are real disruptions, ~10% are decoys that never happen."""
    rows = []
    for month in months:
        for region in REGIONS:
            kind = rng.choice(["strike", "weather", "policy", "decoy", "calm"], p=[.09, .08, .08, .10, .65])
            rows.append({"month": month, "region": region, "kind": kind,
                         "headline": rng.choice(HEADLINES[kind]).format(r=region)})
    news = pd.DataFrame(rows)
    news.insert(0, "news_id", [f"N{i:04d}" for i in range(len(news))])
    return news


def make_shipments(n, first_id, start, news, rng):
    """Shipments whose `is_delayed` label depends on operations data and on that month's regional news."""
    region = rng.choice(REGIONS, n, p=[.25, .2, .25, .15, .15])
    carrier = rng.choice(["FastShip", "OceanLine", "AirCo", "RailX"], n)
    mode = np.where(carrier == "AirCo", "Air", np.where(carrier == "RailX", "Rail", rng.choice(["Sea", "Road"], n)))
    order_date = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    month = order_date.strftime("%Y-%m")
    real_event = news[news["kind"].isin(["strike", "weather", "policy"])].set_index(["month", "region"]).index
    hit = pd.MultiIndex.from_arrays([month, region]).isin(real_event)

    distance = rng.gamma(2.0, 1500, n).round(0)
    weather = rng.beta(2, 5, n)
    warehouse_load = rng.uniform(30, 100, n)
    supplier_rating = rng.normal(4.0, 0.6, n).clip(1, 5)
    qty = rng.integers(1, 800, n)
    logit = (-3.0 + 2.5 * weather + 0.03 * (warehouse_load - 60) - 0.6 * (supplier_rating - 4)
             + 0.0002 * distance + 0.6 * (mode == "Sea") + 1.8 * hit)
    df = pd.DataFrame({
        "shipment_id": [f"SHP{i:06d}" for i in range(first_id, first_id + n)],
        "order_date": order_date.strftime("%Y-%m-%d"),
        "origin_region": region,
        "carrier": carrier,
        "transport_mode": mode,
        "product_category": rng.choice(["Electronics", "Apparel", "Food", "Pharma", "Industrial"], n),
        "distance_km": distance,
        "planned_days": rng.integers(2, 35, n),
        "weather_risk": weather.round(3),
        "warehouse_load_pct": warehouse_load.round(1),
        "supplier_rating": supplier_rating.round(2),
        "order_quantity": qty,
        "order_value_usd": (qty * rng.uniform(5, 250, n)).round(2),
        "is_delayed": (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int),
    })
    df.loc[rng.uniform(size=n) < 0.05, "weather_risk"] = np.nan
    df.loc[rng.uniform(size=n) < 0.03, "carrier"] = ""
    return df


def write_sample_data(volume_path, overwrite=False, seed=7):
    """Write Kaggle-style train/test/sample_submission CSVs plus news.csv; existing files are kept unless overwrite=True."""
    rng = np.random.default_rng(seed)
    news = make_news(pd.period_range("2024-01", "2025-12", freq="M").strftime("%Y-%m"), rng)
    train = make_shipments(20_000, 0, "2024-01-01", news, rng)
    test = make_shipments(5_000, 20_000, "2025-01-01", news, rng)
    outputs = {
        "train.csv": train,
        "test.csv": test.drop(columns="is_delayed"),
        "sample_submission.csv": pd.DataFrame({"shipment_id": test["shipment_id"], "is_delayed": 0.5}),
        "news.csv": news.drop(columns="kind"),
    }
    written = []
    for name, df in outputs.items():
        path = f"{volume_path}/{name}"
        if os.path.exists(path) and not overwrite:
            continue
        df.to_csv(path, index=False)
        written.append(name)
    return written
