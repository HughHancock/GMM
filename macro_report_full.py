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
import yfinance as yf

# ---------- Config ----------
# Update end date to ensure we get the most recent data
END = datetime.now()
START = datetime(2018, 1, 1)

OUT_PDF = "macro_monitor.pdf"
OUT_XLSX = "macro_tracker.xlsx"

# Treat these as point changes (not percent)
YIELD_OR_SPREAD = {
    "FEDFUNDS","TB3MS","GS2","GS5","GS10","GS30",
    "AAA","BAA","BAMLH0A0HYM2","BAMLC0A0CMEY","BAMLC0A4CBBBEY",
    "MORTGAGE30US"
}

SECTIONS: Tuple[Tuple[str, Dict[str,str]], ...] = (
    ("Major Indices", {
        "YF:^GSPC": "S&P 500",
        "YF:SPY": "SPY (ETF)",
        "YF:IWM": "Russell 2000 (IWM)",
        "YF:QQQ": "Nasdaq 100 (QQQ)",
        "YF:DIA": "Dow Jones (DIA)",
    }),
    ("Sectors (SPDR)", {
        "YF:XLB":"Materials","YF:XLE":"Energy","YF:XLF":"Financials",
        "YF:XLI":"Industrials","YF:XLK":"Technology","YF:XLP":"Staples",
        "YF:XLU":"Utilities","YF:XLV":"Health Care","YF:XLY":"Discretionary",
        "YF:XLC":"Comm Services","YF:XLRE":"Real Estate",
    }),
    ("Volatility & Credit", {
        "YF:^VIX":"VIX",
        "FRED:BAMLH0A0HYM2":"HY OAS",
        "FRED:AAA":"Moody's AAA",
        "FRED:BAA":"Moody's BAA",
    }),
    ("Commodities & Energy", {
        "YF:CL=F":"WTI Crude",
        "YF:BZ=F":"Brent Crude",
        "YF:NG=F":"Natural Gas",
        "YF:GC=F":"Gold",
        "YF:SI=F":"Silver",
        "YF:XOP":"Oil & Gas E&P",
        "YF:OIH":"Oil Services",
    }),
    ("Rates & Curve", {
        "FRED:FEDFUNDS":"Fed Funds",
        "FRED:TB3MS":"3M T-Bill",
        "YF:^IRX":"13W T-Bill (x10)",
        "YF:^FVX":"5-Yr Treasury",
        "YF:^TNX":"10-Yr Treasury",
        "YF:^TYX":"30-Yr Treasury",
    }),
    ("Real Estate & International", {
        "YF:VNQ":"US REITs (VNQ)",
        "YF:IYR":"US Real Estate",
        "YF:EFA":"Intl Developed (EFA)",
        "YF:EEM":"Emerging Markets",
        "FRED:MORTGAGE30US":"30-Yr Mortgage",
    }),
)

# ---------- Helpers ----------

def fetch_series(ident: str, start=START, end=END) -> pd.Series:
    """Fetch a single series from Yahoo Finance, FRED, or Stooq."""
    src, code = ident.split(":", 1)
    try:
        if src == "YF":
            # Use yfinance for real-time data
            ticker = yf.Ticker(code)
            df = ticker.history(start=start, end=end)
            if df.empty:
                # Fallback to download method
                df = yf.download(code, start=start, end=end, progress=False)
            if not df.empty:
                s = df["Close"]
                # For treasury yields from Yahoo, they're already in percent
                if code in ["^IRX", "^FVX", "^TNX", "^TYX"]:
                    s = s  # Keep as is, they're already yields in percent
            else:
                s = pd.Series(dtype=float)
        elif src == "FRED":
            # For FRED, explicitly set end date to today
            df = web.DataReader(code, "fred", start, end)
            s = df.iloc[:,0]
        elif src == "STQ":
            # For Stooq
            df = web.DataReader(code, "stooq", start, end)
            s = df["Close"].sort_index()
        else:
            return pd.Series(dtype=float)
        
        # Ensure we have recent data
        s = s.loc[(s.index >= start) & (s.index <= end)].dropna()
        if not s.empty:
            s = s.ffill()
        return s
    except Exception as e:
        warnings.warn(f"Fetch failed for {ident}: {e}")
        return pd.Series(dtype=float)

def pct_return(a, b):
    try:
        return (a/b - 1.0) * 100.0
    except:
        return np.nan

def diff_return(a, b):
    try:
        return a - b
    except:
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
        s0 = s[s.index <= t0].iloc[-1] if len(s[s.index <= t0]) > 0 else np.nan
        s1 = s.iloc[-1] if not s.empty else np.nan
        out[k] = diff_return(s1,s0) if diff_mode else pct_return(s1,s0)
    return out

def normalized_100(s: pd.Series) -> pd.Series:
    if s.empty: 
        return s
    base = s.iloc[0]
    return (s / base) * 100.0 if base != 0 else s * np.nan

def draw_section(pdf, title: str, items: Dict[str,str], data_map: Dict[str,pd.Series]) -> pd.DataFrame:
    """Render a section page with better spacing."""
    # Build summary table
    rows = []
    for ident, name in items.items():
        s = data_map.get(ident, pd.Series(dtype=float))
        if s.empty:
            rows.append((name, np.nan, *[np.nan]*7))
            continue
        
        # Check if this should use diff mode
        code = ident.split(":")[1]
        diff_mode = code in YIELD_OR_SPREAD or code in ["^IRX", "^FVX", "^TNX", "^TYX"]
        
        end_date = s.index[-1]
        ret = compute_returns(s, end_date, diff_mode)
        rows.append((name, s.iloc[-1], ret["YTD"], ret["1M"], ret["3M"], ret["1Y"], ret["3Y"], ret["5Y"], ret["10Y"]))

    cols = ["Series","Current","YTD","1M","3M","1Y","3Y","5Y","10Y"]
    tab = pd.DataFrame(rows, columns=cols)

    # Create figure with better layout
    n = len(items)
    # Adjust figure height based on number of items
    fig_height = max(11, 6 + n * 2.5)
    fig = plt.figure(figsize=(11, fig_height), dpi=100)
    
    # Title
    fig.suptitle(title, x=0.5, y=0.98, ha="center", va="top", fontsize=14, fontweight="bold")

    # Calculate positions for table and charts
    table_height = min(0.15, 0.4 / n)  # Scale table height
    table_top = 0.94
    charts_top = table_top - table_height - 0.03
    charts_bottom = 0.02
    
    # Header table with adjusted position
    ax_table = fig.add_axes([0.05, table_top - table_height, 0.9, table_height])
    ax_table.axis("off")
    
    # Format table data
    safe = tab.copy()
    for c in safe.columns[1:]:
        safe[c] = pd.to_numeric(safe[c], errors="coerce")
    cellText = safe.round(2).fillna("").values
    
    # Create table with better formatting
    table = ax_table.table(
        cellText=cellText,
        colLabels=cols,
        cellLoc="center",
        loc="center",
        colColours=["#E9EEF6"]*len(cols),
        colWidths=[0.15] + [0.106]*8  # Adjust column widths
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Charts grid with proper spacing
    chart_height = (charts_top - charts_bottom) / n
    for i, (ident, name) in enumerate(items.items()):
        s = data_map.get(ident, pd.Series(dtype=float))
        
        # Calculate position for this chart pair
        y_pos = charts_top - (i + 1) * chart_height
        
        # Create two subplots side by side for each series
        ax1 = fig.add_axes([0.08, y_pos + chart_height * 0.55, 0.42, chart_height * 0.35])
        ax2 = fig.add_axes([0.55, y_pos + chart_height * 0.55, 0.42, chart_height * 0.35])

        if s.empty:
            ax1.text(0.5, 0.5, f"{name}: no data", ha="center", va="center")
            ax1.axis("off")
            ax2.axis("off")
            continue

        # Plot normalized (left)
        s_norm = normalized_100(s)
        ax1.plot(s_norm.index, s_norm.values, linewidth=0.8)
        ax1.set_title(f"{name} - Normalized (2018=100)", fontsize=8, pad=3)
        ax1.grid(True, alpha=0.25, linewidth=0.5)
        ax1.tick_params(labelsize=7)
        if not s_norm.empty:
            ax1.set_xlim(s_norm.index.min(), s_norm.index.max())

        # Plot level (right)
        ax2.plot(s.index, s.values, linewidth=0.8)
        code = ident.split(':')[1]
        is_yield = code in YIELD_OR_SPREAD or code in ["^IRX", "^FVX", "^TNX", "^TYX"]
        ax2.set_title(f"{name} - {'Yield (%)' if is_yield else 'Level'}", fontsize=8, pad=3)
        ax2.grid(True, alpha=0.25, linewidth=0.5)
        ax2.tick_params(labelsize=7)
        if not s.empty:
            ax2.set_xlim(s.index.min(), s.index.max())

    # Add data date at bottom
    fig.text(0.99, 0.01, f"Data as of: {END.strftime('%Y-%m-%d %H:%M')}", 
             ha='right', va='bottom', fontsize=8, style='italic')

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return tab

def main():
    from matplotlib.backends.backend_pdf import PdfPages
    
    print(f"Fetching real-time data from {START.strftime('%Y-%m-%d')} to {END.strftime('%Y-%m-%d %H:%M')}...")

    # Fetch all data first
    data = {}
    for section, items in SECTIONS:
        for ident in items.keys():
            if ident not in data:
                print(f"  Fetching {ident}...")
                data[ident] = fetch_series(ident)
                if not data[ident].empty:
                    last_date = data[ident].index[-1]
                    last_value = data[ident].iloc[-1]
                    print(f"    -> Latest: {last_value:.2f} on {last_date.strftime('%Y-%m-%d')}")

    tables = []
    with PdfPages(OUT_PDF) as pdf:
        for section, items in SECTIONS:
            print(f"Creating page: {section}")
            tables.append((section, draw_section(pdf, section, items, data)))

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        # Summary sheets
        for section, tab in tables:
            sheet_name = section[:31]  # Excel sheet name limit
            tab.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Raw series sheets
        for ident, s in data.items():
            if s.empty: 
                continue
            sheet_name = ident.replace(":", "_").replace("^", "")[:31]
            df = s.to_frame(name=ident)
            df.to_excel(writer, sheet_name=sheet_name)

    print(f"✅ Wrote {OUT_PDF} and {OUT_XLSX}")
    print(f"📊 Generated report with real-time data through {END.strftime('%Y-%m-%d %H:%M')}")

if __name__ == "__main__":
    warnings.simplefilter("ignore", category=UserWarning)
    main()
