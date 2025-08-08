#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
from datetime import datetime, timedelta
from typing import Dict, Tuple
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pandas_datareader import data as web
import yfinance as yf

# Suppress yfinance warnings
import logging
logging.getLogger('yfinance').setLevel(logging.ERROR)

# ---------- Config ----------
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
        "YF:SPY": "S&P 500 (SPY)",
        "YF:QQQ": "Nasdaq 100 (QQQ)",
        "YF:IWM": "Russell 2000 (IWM)",
        "YF:DIA": "Dow Jones (DIA)",
        "FRED:SP500": "S&P 500 Index",
    }),
    ("Sectors (SPDR)", {
        "YF:XLB":"Materials","YF:XLE":"Energy","YF:XLF":"Financials",
        "YF:XLI":"Industrials","YF:XLK":"Technology","YF:XLP":"Staples",
        "YF:XLU":"Utilities","YF:XLV":"Health Care","YF:XLY":"Discretionary",
        "YF:XLC":"Comm Services","YF:XLRE":"Real Estate",
    }),
    ("Volatility & Credit", {
        "YF:VIXY":"VIX ETF (VIXY)",
        "FRED:VIXCLS":"VIX Index",
        "FRED:BAMLH0A0HYM2":"HY OAS",
        "FRED:AAA":"Moody's AAA",
        "FRED:BAA":"Moody's BAA",
    }),
    ("Commodities & Energy", {
        "YF:USO":"Oil ETF (USO)",
        "YF:GLD":"Gold ETF (GLD)",
        "YF:SLV":"Silver ETF (SLV)",
        "YF:UNG":"Nat Gas ETF (UNG)",
        "YF:XOP":"Oil & Gas E&P",
        "YF:OIH":"Oil Services",
        "FRED:DCOILWTICO":"WTI Crude (FRED)",
    }),
    ("Rates & Curve", {
        "FRED:FEDFUNDS":"Fed Funds",
        "FRED:TB3MS":"3M T-Bill",
        "FRED:GS2":"2-Yr Treasury",
        "FRED:GS5":"5-Yr Treasury",
        "FRED:GS10":"10-Yr Treasury",
        "FRED:GS30":"30-Yr Treasury",
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

def fetch_yf_robust(ticker: str, start: datetime, end: datetime, retries: int = 3) -> pd.Series:
    """Fetch Yahoo Finance data with retries and better error handling."""
    for attempt in range(retries):
        try:
            # Method 1: Try download first (more reliable)
            df = yf.download(ticker, start=start, end=end, progress=False, show_errors=False)
            if not df.empty and 'Close' in df.columns:
                return df['Close'].dropna()
            
            # Method 2: Try Ticker object
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start, end=end)
            if not df.empty and 'Close' in df.columns:
                return df['Close'].dropna()
                
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)  # Wait before retry
                continue
            warnings.warn(f"YF fetch failed for {ticker} after {retries} attempts: {e}")
    
    return pd.Series(dtype=float)

def fetch_series(ident: str, start=START, end=END) -> pd.Series:
    """Fetch a single series from Yahoo Finance or FRED."""
    src, code = ident.split(":", 1)
    try:
        if src == "YF":
            # Use robust Yahoo Finance fetcher
            s = fetch_yf_robust(code, start, end)
            if not s.empty:
                # Ensure index is datetime
                s.index = pd.to_datetime(s.index)
                s = s.loc[(s.index >= start) & (s.index <= end)]
            return s
            
        elif src == "FRED":
            # FRED data
            df = web.DataReader(code, "fred", start, end)
            s = df.iloc[:,0]
            s = s.loc[(s.index >= start) & (s.index <= end)].dropna()
            if not s.empty:
                s = s.ffill()
            return s
            
        elif src == "STQ":
            # Stooq data (backup option)
            df = web.DataReader(code, "stooq", start, end)
            s = df["Close"].sort_index()
            s = s.loc[(s.index >= start) & (s.index <= end)].dropna()
            if not s.empty:
                s = s.ffill()
            return s
            
    except Exception as e:
        warnings.warn(f"Fetch failed for {ident}: {e}")
    
    return pd.Series(dtype=float)

def pct_return(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        return (a/b - 1.0) * 100.0
    except:
        return np.nan

def diff_return(a, b):
    try:
        if pd.isna(a) or pd.isna(b):
            return np.nan
        return a - b
    except:
        return np.nan

def compute_returns(s: pd.Series, end_date: datetime, diff_mode: bool) -> dict:
    """Compute YTD, 1M, 3M, 1Y, 3Y, 5Y, 10Y returns."""
    horizons = {
        "YTD": datetime(end_date.year, 1, 1),
        "1M": end_date - timedelta(days=30),
        "3M": end_date - timedelta(days=90),
        "1Y": end_date - timedelta(days=365),
        "3Y": end_date - timedelta(days=3*365),
        "5Y": end_date - timedelta(days=5*365),
        "10Y": end_date - timedelta(days=10*365),
    }
    
    out = {}
    for k, t0 in horizons.items():
        # Find closest available date
        mask = s.index <= t0
        if mask.any():
            s0 = s.loc[mask].iloc[-1]
        else:
            s0 = np.nan
        
        s1 = s.iloc[-1] if not s.empty else np.nan
        out[k] = diff_return(s1, s0) if diff_mode else pct_return(s1, s0)
    
    return out

def normalized_100(s: pd.Series) -> pd.Series:
    if s.empty: 
        return s
    base = s.iloc[0]
    if base == 0 or pd.isna(base):
        return pd.Series(index=s.index, dtype=float)
    return (s / base) * 100.0

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
        diff_mode = code in YIELD_OR_SPREAD
        
        end_date = s.index[-1] if not s.empty else datetime.now()
        ret = compute_returns(s, end_date, diff_mode)
        current_val = s.iloc[-1] if not s.empty else np.nan
        
        rows.append((name, current_val, ret["YTD"], ret["1M"], ret["3M"], 
                    ret["1Y"], ret["3Y"], ret["5Y"], ret["10Y"]))

    cols = ["Series","Current","YTD","1M","3M","1Y","3Y","5Y","10Y"]
    tab = pd.DataFrame(rows, columns=cols)

    # Create figure with better layout
    n = len(items)
    fig_height = max(11, 6 + n * 2.5)
    fig = plt.figure(figsize=(11, fig_height), dpi=100)
    
    # Title
    fig.suptitle(title, x=0.5, y=0.98, ha="center", va="top", fontsize=14, fontweight="bold")

    # Calculate positions
    table_height = min(0.15, 0.4 / n)
    table_top = 0.94
    charts_top = table_top - table_height - 0.03
    charts_bottom = 0.02
    
    # Header table
    ax_table = fig.add_axes([0.05, table_top - table_height, 0.9, table_height])
    ax_table.axis("off")
    
    # Format table data - handle NaN values properly
    safe = tab.copy()
    for c in safe.columns[1:]:
        safe[c] = pd.to_numeric(safe[c], errors="coerce")
        # Round numeric columns
        if c == "Current":
            # Keep more precision for current values
            safe[c] = safe[c].round(2)
        else:
            # Returns rounded to 2 decimal places
            safe[c] = safe[c].round(2)
    
    # Replace NaN with empty string for display
    cellText = safe.fillna("").astype(str).values
    
    # Create table
    table = ax_table.table(
        cellText=cellText,
        colLabels=cols,
        cellLoc="center",
        loc="center",
        colColours=["#E9EEF6"]*len(cols),
        colWidths=[0.15] + [0.106]*8
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Charts grid
    chart_height = (charts_top - charts_bottom) / n
    for i, (ident, name) in enumerate(items.items()):
        s = data_map.get(ident, pd.Series(dtype=float))
        
        # Calculate position
        y_pos = charts_top - (i + 1) * chart_height
        
        # Create two subplots side by side
        ax1 = fig.add_axes([0.08, y_pos + chart_height * 0.55, 0.42, chart_height * 0.35])
        ax2 = fig.add_axes([0.55, y_pos + chart_height * 0.55, 0.42, chart_height * 0.35])

        if s.empty:
            ax1.text(0.5, 0.5, f"{name}: no data", ha="center", va="center")
            ax1.axis("off")
            ax2.axis("off")
            continue

        # Plot normalized (left)
        s_norm = normalized_100(s)
        if not s_norm.empty and not s_norm.isna().all():
            ax1.plot(s_norm.index, s_norm.values, linewidth=0.8)
            ax1.set_title(f"{name} - Normalized (2018=100)", fontsize=8, pad=3)
            ax1.grid(True, alpha=0.25, linewidth=0.5)
            ax1.tick_params(labelsize=7)
            ax1.set_xlim(s_norm.index.min(), s_norm.index.max())
        else:
            ax1.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            ax1.axis("off")

        # Plot level (right)
        if not s.empty and not s.isna().all():
            ax2.plot(s.index, s.values, linewidth=0.8)
            code = ident.split(':')[1]
            is_yield = code in YIELD_OR_SPREAD
            ax2.set_title(f"{name} - {'Yield (%)' if is_yield else 'Level'}", fontsize=8, pad=3)
            ax2.grid(True, alpha=0.25, linewidth=0.5)
            ax2.tick_params(labelsize=7)
            ax2.set_xlim(s.index.min(), s.index.max())
        else:
            ax2.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            ax2.axis("off")

    # Add timestamp
    fig.text(0.99, 0.01, f"Data as of: {END.strftime('%Y-%m-%d %H:%M')}", 
             ha='right', va='bottom', fontsize=8, style='italic')

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return tab

def main():
    from matplotlib.backends.backend_pdf import PdfPages
    
    print(f"Fetching data from {START.strftime('%Y-%m-%d')} to {END.strftime('%Y-%m-%d %H:%M')}...")
    print("Note: Using ETFs for real-time pricing where available, FRED for economic data")
    print("-" * 60)

    # Fetch all data
    data = {}
    success_count = 0
    fail_count = 0
    
    for section, items in SECTIONS:
        for ident in items.keys():
            if ident not in data:
                print(f"Fetching {ident}...", end=" ")
                data[ident] = fetch_series(ident)
                if not data[ident].empty:
                    last_date = data[ident].index[-1]
                    last_value = data[ident].iloc[-1]
                    print(f"✓ {last_value:.2f} ({last_date.strftime('%Y-%m-%d')})")
                    success_count += 1
                else:
                    print("✗ No data")
                    fail_count += 1

    print("-" * 60)
    print(f"Fetched {success_count} series successfully, {fail_count} failed")
    
    if success_count == 0:
        print("ERROR: No data was fetched. Check your internet connection.")
        return

    # Generate report
    tables = []
    with PdfPages(OUT_PDF) as pdf:
        for section, items in SECTIONS:
            print(f"Creating page: {section}")
            tables.append((section, draw_section(pdf, section, items, data)))

    # Generate Excel
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        # Summary sheets
        for section, tab in tables:
            sheet_name = section[:31]
            tab.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Raw data sheets
        for ident, s in data.items():
            if s.empty: 
                continue
            sheet_name = ident.replace(":", "_").replace("^", "").replace("=", "")[:31]
            df = s.to_frame(name=ident)
            df.to_excel(writer, sheet_name=sheet_name)

    print("-" * 60)
    print(f"✅ Generated {OUT_PDF}")
    print(f"✅ Generated {OUT_XLSX}")
    print(f"📊 Report complete with data through {END.strftime('%Y-%m-%d %H:%M')}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
