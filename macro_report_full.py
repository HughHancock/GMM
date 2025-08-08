#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, math, warnings
from datetime import datetime, timedelta
from io import BytesIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from pandas_datareader import data as web

START = datetime(2018, 1, 1)
END   = datetime.now()

OUT_PDF  = "macro_monitor.pdf"
OUT_XLSX = "macro_tracker.xlsx"

# Series that should be shown as point changes (not % returns)
YIELD_OR_SPREAD = {
    "FEDFUNDS", "TB3MS", "GS2", "GS5", "GS10", "GS30",
    "AAA", "BAA", "BAMLH0A0HYM2", "BAMLC0A0CMEY", "BAMLC0A4CBBBEY"
}

# --------- Data Map (section -> {code: display_name}) ---------
SECTIONS = [
    ("Major Indices", {
        # FRED has SP500 (close). Stooq for ETFs as fallback if available.
        "FRED:SP500": "S&P 500",
        "STQ:SPY":    "SPY (ETF)",
        "STQ:IWM":    "Russell 2000 (IWM)",
    }),
    ("Sectors (SPDR)", {
        "STQ:XLB": "Materials",
        "STQ:XLE": "Energy",
        "STQ:XLF": "Financials",
        "STQ:XLI": "Industrials",
        "STQ:XLK": "Technology",
        "STQ:XLP": "Staples",
        "STQ:XLU": "Utilities",
        "STQ:XLV": "Health Care",
        "STQ:XLY": "Discretionary",
        "STQ:XLC": "Comm Services",
    }),
    ("Volatility & Credit", {
        "FRED:VIXCLS":        "VIX",
        "FRED:BAMLH0A0HYM2":  "HY OAS",
        "FRED:AAA":           "Moody's AAA",
        "FRED:BAA":           "Moody's BAA",
    }),
    ("Commodities & Energy Stocks", {
        "FRED:DCOILWTICO": "WTI Crude",
        "FRED:DCOILBRENTEU": "Brent",
        "FRED:DHHNGSP":   "Natural Gas",
        "STQ:XOP":        "Oil & Gas E&P (XOP)",
        "STQ:OIH":        "Oil Services (OIH)",
    }),
    ("Rates & Curve", {
        "FRED:FEDFUNDS": "Fed Funds",
        "FRED:TB3MS":    "3M T-Bill",
        "FRED:GS2":      "2-Yr Treasury",
        "FRED:GS5":      "5-Yr Treasury",
        "FRED:GS10":     "10-Yr Treasury",
        "FRED:GS30":     "30-Yr Treasury",
    }),
    ("Real Estate", {
        "STQ:VNQ":          "US REITs (VNQ)",
        "STQ:IYR":          "US Real Estate (IYR)",
        "FRED:MORTGAGE30US":"30-Yr Mortgage Rate",
    }),
]

# ---------- helpers ----------

def fetch_series(ident: str, start=START, end=END) -> pd.Series:
    """Fetch a single series from FRED or Stooq and return as Series (UTC-naive)."""
    try:
        source, code = ident.split(":", 1)
    except ValueError:
        raise ValueError(f"Bad ident '{ident}' (expected 'FRED:CODE' or 'STQ:TICKER')")

    try:
        if source == "FRED":
            df = web.DataReader(code, "fred", start, end)
            s  = df.iloc[:, 0].dropna()
        elif source == "STQ":
            # Stooq returns OHLCV; we'll use 'Close'
            df = web.DataReader(code, "stooq", start, end)
            s  = df["Close"].sort_index().dropna()
        else:
            return pd.Series(dtype=float)
        # daily to daily, no resample; forward-fill gaps for charts
        s = s.loc[(s.index >= start) & (s.index <= end)]
        return s.ffill()
    except Exception as e:
        warnings.warn(f"Fetch failed for {ident}: {e}")
        return pd.Series(dtype=float)

def pct_return(a, b):
    try:
        return (a / b - 1.0) * 100.0
    except Exception:
        return np.nan

def diff_return(a, b):
    try:
        return a - b
    except Exception:
        return np.nan

def compute_returns(s: pd.Series, end_date: datetime, treat_as_diff: bool) -> dict:
    """YTD, 1M, 3M, 1Y, 3Y, 5Y, 10Y (percent unless diff)."""
    horizons = {
        "YTD": datetime(end_date.year, 1, 1),
        "1M": end_date - timedelta(days=30),
        "3M": end_date - timedelta(days=90),
        "1Y": end_date - timedelta(days=365),
        "3Y": end_date - timedelta(days=3*365),
        "5Y": end_date - timedelta(days=5*365),
        "10Y": end_date - timedelta(days=10*365)
    }
    out = {}
    for k, t0 in horizons.items():
        s0 = s[:t0].iloc[-1] if not s[:t0].empty else np.nan
        s1 = s.iloc[-1] if not s.empty else np.nan
        out[k] = diff_return(s1, s0) if treat_as_diff else pct_return(s1, s0)
    return out

def normalized_100(s: pd.Series) -> pd.Series:
    if s.empty: return s
    return (s / s.iloc[0]) * 100.0

def draw_section(pdf, title: str, items: dict, data_map: dict):
    """Table + two stacked charts per series (Normalized, and level or diff)."""
    # Prepare summary table
    rows = []
    for ident, name in items.items():
        s = data_map.get(ident, pd.Series(dtype=float))
        if s.empty: 
            rows.append((name, np.nan, *[np.nan]*7))
            continue
        treat_as_diff = ident.split(":")[1] in YIELD_OR_SPREAD
        end_date = s.index[-1]
        ret = compute_returns(s, end_date, treat_as_diff)
        rows.append((
            name,
            s.iloc[-1],
            ret["YTD"], ret["1M"], ret["3M"], ret["1Y"], ret["3Y"], ret["5Y"], ret["10Y"]
        ))
    cols = ["Series","Current","YTD","1M","3M","1Y","3Y","5Y","10Y"]
    tab = pd.DataFrame(rows, columns=cols)

    # Figure
    n = len(items)
    fig = plt.figure(figsize=(11, 8.5), dpi=150)
    fig.suptitle(title, x=0.02, y=0.98, ha="left", va="top", fontsize=16, fontweight="bold")

    # Header table
    ax_table = fig.add_axes([0.05, 0.86, 0.9, 0.09])  # left, bottom, width, height
    ax_table.axis("off")
    ax_table.table(
        cellText=np.round(tab.fillna("").values, 2),
        colLabels=cols,
        cellLoc="center",
        loc="center",
        colColours=["#E9EEF6"]*len(cols),
    )

    # Charts area
    top = 0.83
    bottom = 0.06
    gs = gridspec.GridSpec(nrows=2*n, ncols=1, left=0.07, right=0.95, bottom=bottom, top=top, hspace=0.18)

    for i, (ident, name) in enumerate(items.items()):
        s = data_map.get(ident, pd.Series(dtype=float))
        ax1 = fig.add_subplot(gs[2*i, 0])
        ax2 = fig.add_subplot(gs[2*i+1, 0])

        if s.empty:
            ax1.text(0.5, 0.5, f"{name}: no data", ha="center", va="center")
            ax1.axis("off"); ax2.axis("off")
            continue

        # Upper: normalized
        s_norm = normalized_100(s)
        ax1.plot(s_norm.index, s_norm.values)
        ax1.set_title(f"{name} – Normalized (2018=100)", fontsize=10)
        ax1.grid(True, alpha=0.25)
        ax1.set_xlim(s_norm.index.min(), s_norm.index.max())

        # Lower: level (or diff baseline)
        treat_as_diff = ident.split(":")[1] in YIELD_OR_SPREAD
        ax2.plot(s.index, s.values)
        ax2.grid(True, alpha=0.25)
        ax2.set_xlim(s.index.min(), s.index.max())
        ax2.set_title(f"{name} – {'Level / Spread' if treat_as_diff else 'Level'}", fontsize=10)
        # Only show x labels on the very last axis
        if i < n-1:
            ax2.tick_params(labelbottom=False)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return tab

def main():
    # Fetch all data first (so a single failure won’t corrupt a page mid-draw)
    data = {}
    for section, items in SECTIONS:
        for ident in items.keys():
            if ident not in data:
                data[ident] = fetch_series(ident)

    # Write PDF
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(OUT_PDF) as pdf:
        all_tables = []
        for section, items in SECTIONS:
            tab = draw_section(pdf, section, items, data)
            all_tables.append((section, tab))

    # Write Excel
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        for section, tab in all_tables:
            safe_name = section[:31].replace("/", "-")
            tab.to_excel(writer, index=False, sheet_name=safe_name)
        # raw sheets
        for ident, s in data.items():
            if s.empty: 
                continue
            safe = ident.replace(":","_")[:31]
            s.to_frame(name=ident).to_excel(writer, sheet_name=safe)

    print(f"✅ Wrote {OUT_PDF} and {OUT_XLSX}")

if __name__ == "__main__":
    # Make non-fatal if a source is down
    warnings.simplefilter("always", UserWarning)
    main()
