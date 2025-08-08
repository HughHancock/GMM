#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pandas_datareader import data as web

# ---------- Config ----------
START = datetime(2018, 1, 1)
END   = datetime.now()

OUT_PDF  = "macro_monitor.pdf"
OUT_XLSX = "macro_tracker.xlsx"

# Treat these as point changes (not percent)
YIELD_OR_SPREAD = {
    "FEDFUNDS","TB3MS","GS2","GS5","GS10","GS30",
    "AAA","BAA","BAMLH0A0HYM2","BAMLC0A0CMEY","BAMLC0A4CBBBEY",
    "MORTGAGE30US"
}

SECTIONS: Tuple[Tuple[str, Dict[str,str]], ...] = (
    ("Major Indices", {
        "FRED:SP500": "S&P 500",
        "STQ:SPY":    "SPY (ETF)",
        "STQ:IWM":    "Russell 2000 (IWM)",
    }),
    ("Sectors (SPDR)", {
        "STQ:XLB":"Materials","STQ:XLE":"Energy","STQ:XLF":"Financials",
        "STQ:XLI":"Industrials","STQ:XLK":"Technology","STQ:XLP":"Staples",
        "STQ:XLU":"Utilities","STQ:XLV":"Health Care","STQ:XLY":"Discretionary",
        "STQ:XLC":"Comm Services",
    }),
    ("Volatility & Credit", {
        "FRED:VIXCLS":"VIX","FRED:BAMLH0A0HYM2":"HY OAS",
        "FRED:AAA":"Moody's AAA","FRED:BAA":"Moody's BAA",
    }),
    ("Commodities & Energy Stocks", {
        "FRED:DCOILWTICO":"WTI Crude","FRED:DCOILBRENTEU":"Brent",
        "FRED:DHHNGSP":"Nat Gas","STQ:XOP":"Oil & Gas E&P (XOP)","STQ:OIH":"Oil Services (OIH)",
    }),
    ("Rates & Curve", {
        "FRED:FEDFUNDS":"Fed Funds","FRED:TB3MS":"3M T-Bill","FRED:GS2":"2-Yr UST",
        "FRED:GS5":"5-Yr UST","FRED:GS10":"10-Yr UST","FRED:GS30":"30-Yr UST",
    }),
    ("Real Estate", {
        "STQ:VNQ":"US REITs (VNQ)","STQ:IYR":"US Real Estate (IYR)","FRED:MORTGAGE30US":"30-Yr Mortgage",
    }),
)

# ---------- Helpers ----------

def fetch_series(ident: str, start=START, end=END) -> pd.Series:
    """Fetch a single series from FRED or Stooq and return as a clean Series."""
    src, code = ident.split(":", 1)
    try:
        if src == "FRED":
            df = web.DataReader(code, "fred", start, end)
            s = df.iloc[:,0]
        elif src == "STQ":
            df = web.DataReader(code, "stooq", start, end)
            s = df["Close"].sort_index()
        else:
            return pd.Series(dtype=float)
        s = s.loc[(s.index >= start) & (s.index <= end)].dropna().ffill()
        return s
    except Exception as e:
        warnings.warn(f"Fetch failed for {ident}: {e}")
        return pd.Series(dtype=float)

def pct_return(a, b):
    try:
        return (a/b - 1.0) * 100.0
    except Exception:
        return np.nan

def diff_return(a, b):
    try:
        return a - b
    except Exception:
        return np.nan

def compute_returns(s: pd.Series, end_date: datetime, diff_mode: bool) -> dict:
    """Compute YTD, 1M, 3M, 1Y, 3Y, 5Y, 10Y."""
    horizons = {
        "YTD": datetime(end_date.year,1,1),
        "1M": end_date - timedelta(days=30),
        "3M": end_date - timedelta(days=90),
        "1Y": end_date - timedelta(days=365),
        "3Y": end_date - timedelta(days=3*365),
        "5Y": end_date - timedelta(days=5*365),
        "10Y": end_date - timedelta(days=10*365),
    }
    out = {}
    for k, t0 in horizons.items():
        s0 = s[:t0].iloc[-1] if not s[:t0].empty else np.nan
        s1 = s.iloc[-1] if not s.empty else np.nan
        out[k] = diff_return(s1,s0) if diff_mode else pct_return(s1,s0)
    return out

def normalized_100(s: pd.Series) -> pd.Series:
    if s.empty: return s
    base = s.iloc[0]
    return (s / base) * 100.0 if base != 0 else s * np.nan

def draw_section(pdf, title: str, items: Dict[str,str], data_map: Dict[str,pd.Series]) -> pd.DataFrame:
    """Render a section page (header table + two stacked charts per series)."""
    # Build summary table
    rows = []
    for ident, name in items.items():
        s = data_map.get(ident, pd.Series(dtype=float))
        if s.empty:
            rows.append((name, np.nan, *[np.nan]*7))
            continue
        diff_mode = ident.split(":")[1] in YIELD_OR_SPREAD
        end_date = s.index[-1]
        ret = compute_returns(s, end_date, diff_mode)
        rows.append((name, s.iloc[-1], ret["YTD"], ret["1M"], ret["3M"], ret["1Y"], ret["3Y"], ret["5Y"], ret["10Y"]))

    cols = ["Series","Current","YTD","1M","3M","1Y","3Y","5Y","10Y"]
    tab = pd.DataFrame(rows, columns=cols)

    # Figure
    n = len(items)
    fig = plt.figure(figsize=(11, 8.5), dpi=150)
    fig.suptitle(title, x=0.02, y=0.985, ha="left", va="top", fontsize=16, fontweight="bold")

    # Header table (safe rounding: only numeric cols)
    safe = tab.copy()
    for c in safe.columns[1:]:
        safe[c] = pd.to_numeric(safe[c], errors="coerce")
    cellText = safe.round(2).fillna("").values

    ax_table = fig.add_axes([0.05, 0.87, 0.9, 0.09])
    ax_table.axis("off")
    ax_table.table(
        cellText=cellText,
        colLabels=cols,
        cellLoc="center",
        loc="center",
        colColours=["#E9EEF6"]*len(cols),
    )

    # Charts grid
    gs = gridspec.GridSpec(nrows=2*n, ncols=1, left=0.07, right=0.95, bottom=0.06, top=0.85, hspace=0.18)
    for i, (ident, name) in enumerate(items.items()):
        s = data_map.get(ident, pd.Series(dtype=float))
        a1 = fig.add_subplot(gs[2*i, 0])
        a2 = fig.add_subplot(gs[2*i+1, 0])

        if s.empty:
            a1.text(0.5, 0.5, f"{name}: no data", ha="center", va="center"); a1.axis("off"); a2.axis("off")
            continue

        # normalized
        s_norm = normalized_100(s)
        a1.plot(s_norm.index, s_norm.values)
        a1.set_title(f"{name} — Normalized (2018=100)", fontsize=10)
        a1.grid(True, alpha=0.25)
        a1.set_xlim(s_norm.index.min(), s_norm.index.max())

        # level
        a2.plot(s.index, s.values)
        a2.set_title(f"{name} — {'Level / Spread' if ident.split(':')[1] in YIELD_OR_SPREAD else 'Level'}", fontsize=10)
        a2.grid(True, alpha=0.25)
        a2.set_xlim(s.index.min(), s.index.max())
        if i < n-1: a2.tick_params(labelbottom=False)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return tab

def main():
    from matplotlib.backends.backend_pdf import PdfPages

    # Fetch all data first
    data = {}
    for section, items in SECTIONS:
        for ident in items.keys():
            if ident not in data:
                data[ident] = fetch_series(ident)

    tables = []
    with PdfPages(OUT_PDF) as pdf:
        for section, items in SECTIONS:
            tables.append((section, draw_section(pdf, section, items, data)))

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        # summary sheets
        for section, tab in tables:
            tab.to_excel(writer, sheet_name=section[:31], index=False)
        # raw series
        for ident, s in data.items():
            if s.empty: continue
            s.to_frame(name=ident).to_excel(writer, sheet_name=ident.replace(":","_")[:31])

    print(f"✅ Wrote {OUT_PDF} and {OUT_XLSX}")

if __name__ == "__main__":
    warnings.simplefilter("ignore", category=UserWarning)
    main()
