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
END = datetime.now()
START = datetime(2018, 1, 1)

OUT_PDF = "macro_monitor.pdf"
OUT_XLSX = "macro_tracker.xlsx"

# Treat these as point changes (not percent)
YIELD_OR_SPREAD = {
    "DFF","FEDFUNDS","TB3MS","DGS3MO","DGS2","DGS5","DGS10","DGS30",
    "GS2","GS5","GS10","GS30",
    "AAA","BAA","BAMLH0A0HYM2","BAMLC0A0CMEY","BAMLC0A4CBBBEY",
    "MORTGAGE30US"
}

# Using FRED and Stooq which are more reliable
SECTIONS: Tuple[Tuple[str, Dict[str,str]], ...] = (
    ("Major Indices", {
        "FRED:SP500": "S&P 500",
        "FRED:DJIA": "Dow Jones",
        "FRED:NASDAQCOM": "NASDAQ Composite",
        "STQ:IWM": "Russell 2000 (IWM)",
        "STQ:SPY": "SPY ETF",
        "STQ:QQQ": "Nasdaq 100 (QQQ)",
    }),
    ("Sectors (SPDR ETFs)", {
        "STQ:XLB":"Materials","STQ:XLE":"Energy","STQ:XLF":"Financials",
        "STQ:XLI":"Industrials","STQ:XLK":"Technology","STQ:XLP":"Staples",
        "STQ:XLU":"Utilities","STQ:XLV":"Health Care","STQ:XLY":"Discretionary",
        "STQ:XLC":"Comm Services",
    }),
    ("Volatility & Credit", {
        "FRED:VIXCLS":"VIX",
        "FRED:BAMLH0A0HYM2":"HY Spread",
        "FRED:AAA":"AAA Yield",
        "FRED:BAA":"BAA Yield",
        "FRED:BAMLC0A0CMEY":"IG Spread",
    }),
    ("Commodities & Energy", {
        "FRED:DCOILWTICO":"WTI Crude",
        "FRED:DCOILBRENTEU":"Brent Crude",
        "FRED:DHHNGSP":"Natural Gas",
        "FRED:GOLDAMGBD228NLBM":"Gold",
        "STQ:XLE":"Energy Sector",
        "STQ:XOP":"Oil & Gas E&P",
    }),
    ("Interest Rates", {
        "FRED:FEDFUNDS":"Fed Funds",
        "FRED:TB3MS":"3M T-Bill",
        "FRED:GS2":"2-Yr Treasury",
        "FRED:GS5":"5-Yr Treasury",
        "FRED:GS10":"10-Yr Treasury",
        "FRED:GS30":"30-Yr Treasury",
    }),
    ("Real Estate & Intl", {
        "STQ:VNQ":"US REITs",
        "STQ:IYR":"US Real Estate",
        "STQ:EFA":"Intl Developed",
        "STQ:EEM":"Emerging Markets",
        "FRED:MORTGAGE30US":"30-Yr Mortgage",
    }),
)

# ---------- Helpers ----------

def fetch_series(ident: str, start=START, end=END) -> pd.Series:
    """Fetch a single series from FRED or Stooq."""
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
        
        # Clean and filter data
        s = s.loc[(s.index >= start) & (s.index <= end)].dropna()
        if not s.empty:
            s = s.ffill()  # Forward fill missing values
        return s
    except Exception as e:
        warnings.warn(f"Fetch failed for {ident}: {e}")
        return pd.Series(dtype=float)

def pct_return(a, b):
    """Calculate percentage return."""
    try:
        if pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        return (a/b - 1.0) * 100.0
    except:
        return np.nan

def diff_return(a, b):
    """Calculate difference return (for yields/spreads)."""
    try:
        if pd.isna(a) or pd.isna(b):
            return np.nan
        return a - b
    except:
        return np.nan

def compute_returns(s: pd.Series, end_date: datetime, diff_mode: bool) -> dict:
    """Compute returns for various time periods."""
    horizons = {
        "YTD": datetime(end_date.year, 1, 1),
        "1M": end_date - timedelta(days=30),
        "3M": end_date - timedelta(days=90),
        "1Y": end_date - timedelta(days=365),
        "3Y": end_date - timedelta(days=3*365),
        "5Y": end_date - timedelta(days=5*365),
    }
    
    out = {}
    for k, t0 in horizons.items():
        mask = s.index <= t0
        if mask.any():
            s0 = s.loc[mask].iloc[-1]
        else:
            s0 = np.nan
        
        s1 = s.iloc[-1] if not s.empty else np.nan
        out[k] = diff_return(s1, s0) if diff_mode else pct_return(s1, s0)
    
    return out

def normalized_100(s: pd.Series) -> pd.Series:
    """Normalize series to 100 at start."""
    if s.empty: 
        return s
    base = s.iloc[0]
    if base == 0 or pd.isna(base):
        return pd.Series(index=s.index, dtype=float)
    return (s / base) * 100.0

def draw_section(pdf, title: str, items: Dict[str,str], data_map: Dict[str,pd.Series]) -> pd.DataFrame:
    """Render a section page."""
    # Build summary table
    rows = []
    for ident, name in items.items():
        s = data_map.get(ident, pd.Series(dtype=float))
        if s.empty:
            rows.append((name, np.nan, *[np.nan]*6))
            continue
        
        # Check if this should use diff mode (for yields/spreads)
        code = ident.split(":")[1]
        diff_mode = code in YIELD_OR_SPREAD
        
        end_date = s.index[-1] if not s.empty else datetime.now()
        ret = compute_returns(s, end_date, diff_mode)
        current_val = s.iloc[-1] if not s.empty else np.nan
        
        rows.append((name, current_val, ret["YTD"], ret["1M"], ret["3M"], 
                    ret["1Y"], ret["3Y"], ret["5Y"]))

    cols = ["Series", "Current", "YTD", "1M", "3M", "1Y", "3Y", "5Y"]
    tab = pd.DataFrame(rows, columns=cols)

    # Create figure
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
    
    # Format table data
    safe = tab.copy()
    for c in safe.columns[1:]:
        safe[c] = pd.to_numeric(safe[c], errors="coerce")
        safe[c] = safe[c].round(2)
    
    cellText = safe.fillna("").astype(str).values
    
    # Create table
    table = ax_table.table(
        cellText=cellText,
        colLabels=cols,
        cellLoc="center",
        loc="center",
        colColours=["#E9EEF6"]*len(cols),
        colWidths=[0.15] + [0.121]*7
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Charts
    chart_height = (charts_top - charts_bottom) / n
    for i, (ident, name) in enumerate(items.items()):
        s = data_map.get(ident, pd.Series(dtype=float))
        
        y_pos = charts_top - (i + 1) * chart_height
        
        # Two charts side by side
        ax1 = fig.add_axes([0.08, y_pos + chart_height * 0.55, 0.42, chart_height * 0.35])
        ax2 = fig.add_axes([0.55, y_pos + chart_height * 0.55, 0.42, chart_height * 0.35])

        if s.empty:
            ax1.text(0.5, 0.5, f"{name}: no data", ha="center", va="center", fontsize=9)
            ax1.axis("off")
            ax2.axis("off")
            continue

        # Normalized chart (left)
        s_norm = normalized_100(s)
        if not s_norm.empty and not s_norm.isna().all():
            ax1.plot(s_norm.index, s_norm.values, linewidth=0.8, color='#1f77b4')
            ax1.set_title(f"{name} - Normalized (2018=100)", fontsize=8, pad=3)
            ax1.grid(True, alpha=0.25, linewidth=0.5)
            ax1.tick_params(labelsize=7)
            ax1.set_xlim(s_norm.index.min(), s_norm.index.max())
            
            # Add min/max markers
            if len(s_norm) > 0:
                max_idx = s_norm.idxmax()
                min_idx = s_norm.idxmin()
                ax1.plot(max_idx, s_norm[max_idx], 'g^', markersize=4)
                ax1.plot(min_idx, s_norm[min_idx], 'rv', markersize=4)

        # Level chart (right)
        if not s.empty and not s.isna().all():
            ax2.plot(s.index, s.values, linewidth=0.8, color='#ff7f0e')
            code = ident.split(':')[1]
            is_yield = code in YIELD_OR_SPREAD
            ax2.set_title(f"{name} - {'Yield (%)' if is_yield else 'Level'}", fontsize=8, pad=3)
            ax2.grid(True, alpha=0.25, linewidth=0.5)
            ax2.tick_params(labelsize=7)
            ax2.set_xlim(s.index.min(), s.index.max())
            
            # Add latest value annotation
            if len(s) > 0:
                latest_val = s.iloc[-1]
                latest_date = s.index[-1]
                ax2.annotate(f'{latest_val:.2f}', 
                           xy=(latest_date, latest_val),
                           xytext=(5, 5), 
                           textcoords='offset points',
                           fontsize=6,
                           bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))

    # Add timestamp and data note
    fig.text(0.99, 0.01, f"Data as of: {END.strftime('%Y-%m-%d %H:%M')}", 
             ha='right', va='bottom', fontsize=8, style='italic')
    fig.text(0.01, 0.01, "Note: FRED data may have 1-2 day lag", 
             ha='left', va='bottom', fontsize=7, style='italic', color='gray')

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return tab

def main():
    from matplotlib.backends.backend_pdf import PdfPages
    
    print(f"Macro Monitor Report Generator")
    print(f"=" * 60)
    print(f"Fetching data from {START.strftime('%Y-%m-%d')} to {END.strftime('%Y-%m-%d')}")
    print(f"Using FRED (economic data) and Stooq (ETF prices)")
    print(f"-" * 60)

    # Fetch all data
    data = {}
    success_count = 0
    fail_count = 0
    
    for section, items in SECTIONS:
        for ident in items.keys():
            if ident not in data:
                print(f"  {ident:<20}", end=" ")
                data[ident] = fetch_series(ident)
                if not data[ident].empty:
                    last_date = data[ident].index[-1]
                    last_value = data[ident].iloc[-1]
                    days_old = (END - last_date).days
                    print(f"✓ {last_value:>10.2f} ({days_old}d old)")
                    success_count += 1
                else:
                    print(f"✗ Failed")
                    fail_count += 1

    print(f"-" * 60)
    print(f"Results: {success_count} successful, {fail_count} failed")
    
    if success_count == 0:
        print("ERROR: No data fetched. Check internet connection.")
        return

    # Generate PDF
    print(f"\nGenerating PDF report...")
    tables = []
    with PdfPages(OUT_PDF) as pdf:
        for section, items in SECTIONS:
            print(f"  Creating section: {section}")
            tables.append((section, draw_section(pdf, section, items, data)))

    # Generate Excel
    print(f"\nGenerating Excel workbook...")
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        # Summary sheets
        for section, tab in tables:
            sheet_name = section[:31]
            tab.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  Added sheet: {sheet_name}")
        
        # Raw data sheets
        print(f"\nAdding raw data sheets...")
        for ident, s in data.items():
            if s.empty: 
                continue
            sheet_name = ident.replace(":", "_")[:31]
            df = s.to_frame(name=ident)
            df.to_excel(writer, sheet_name=sheet_name)

    print(f"\n" + "=" * 60)
    print(f"✅ Report generated successfully!")
    print(f"📄 PDF:   {OUT_PDF}")
    print(f"📊 Excel: {OUT_XLSX}")
    print(f"📅 Data through: {END.strftime('%Y-%m-%d %H:%M')}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
