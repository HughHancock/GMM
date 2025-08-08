#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
from datetime import datetime, timedelta
from typing import Dict, Tuple
import json
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pandas_datareader import data as web

# ---------- Config ----------
END = datetime.now()
START = datetime(2018, 1, 1)

OUT_HTML = "index.html"
OUT_JSON = "data.json"

# Treat these as point changes (not percent)
YIELD_OR_SPREAD = {
    "DFF","FEDFUNDS","TB3MS","DGS3MO","DGS2","DGS5","DGS10","DGS30",
    "GS2","GS5","GS10","GS30",
    "AAA","BAA","BAMLH0A0HYM2","BAMLC0A0CMEY","BAMLC0A4CBBBEY",
    "MORTGAGE30US"
}

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
        "FRED:DFF": "Fed Funds Rate",
        "FRED:DGS3MO": "3-Month Treasury",
        "FRED:DGS2": "2-Year Treasury",
        "FRED:DGS5": "5-Year Treasury",
        "FRED:DGS10": "10-Year Treasury",
        "FRED:DGS30": "30-Year Treasury",
    }),
    ("Real Estate & Intl", {
        "STQ:VNQ":"US REITs",
        "STQ:IYR":"US Real Estate",
        "STQ:EFA":"Intl Developed",
        "STQ:EEM":"Emerging Markets",
        "FRED:MORTGAGE30US":"30-Yr Mortgage",
    }),
)

# ---------- Data Fetching (same as before) ----------

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
    if s.empty: 
        return s
    base = s.iloc[0]
    if base == 0 or pd.isna(base):
        return pd.Series(index=s.index, dtype=float)
    return (s / base) * 100.0

def create_mini_chart(s: pd.Series, title: str, is_normalized: bool = False) -> str:
    """Create a small sparkline chart and return as base64 image."""
    if s.empty or s.isna().all():
        return ""
    
    fig, ax = plt.subplots(figsize=(3.5, 1.5), dpi=100)
    
    if is_normalized:
        s = normalized_100(s)
    
    ax.plot(s.index, s.values, linewidth=1, color='#2E86AB')
    ax.fill_between(s.index, s.values, alpha=0.1, color='#2E86AB')
    
    # Remove all labels and ticks for sparkline effect
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add min/max markers
    if len(s) > 0:
        max_idx = s.idxmax()
        min_idx = s.idxmin()
        ax.plot(max_idx, s[max_idx], 'o', markersize=3, color='#28a745')
        ax.plot(min_idx, s[min_idx], 'o', markersize=3, color='#dc3545')
    
    plt.tight_layout(pad=0)
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    
    return f"data:image/png;base64,{image_base64}"

def generate_html_report():
    """Generate the HTML dashboard."""
    print(f"Generating HTML Dashboard...")
    print(f"Fetching data from {START.strftime('%Y-%m-%d')} to {END.strftime('%Y-%m-%d')}")
    
    # Fetch all data
    data = {}
    for section, items in SECTIONS:
        for ident in items.keys():
            if ident not in data:
                print(f"  Fetching {ident}...")
                data[ident] = fetch_series(ident)
    
    # Generate HTML
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="300">
    <title>Macro Monitor - Live Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #0a0e27;
            color: #e0e0e0;
            line-height: 1.6;
        }
        
        .header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 30px;
            text-align: center;
            box-shadow: 0 2px 20px rgba(0,0,0,0.3);
        }
        
        h1 {
            font-size: 2.5em;
            font-weight: 300;
            letter-spacing: 2px;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .last-updated {
            font-size: 1.1em;
            opacity: 0.9;
            color: #B8D4E3;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .section {
            background: #1a1f3a;
            border-radius: 15px;
            margin-bottom: 30px;
            overflow: hidden;
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
            border: 1px solid #2a3f5f;
        }
        
        .section-header {
            background: linear-gradient(135deg, #2a3f5f 0%, #1e2936 100%);
            padding: 20px;
            font-size: 1.4em;
            font-weight: 500;
            border-bottom: 2px solid #3a4f6f;
            letter-spacing: 1px;
        }
        
        .data-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .data-table th {
            background: #2a3450;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #B8D4E3;
            border-bottom: 2px solid #3a4f6f;
        }
        
        .data-table td {
            padding: 12px;
            border-bottom: 1px solid #2a3450;
            font-size: 0.95em;
        }
        
        .data-table tr:hover {
            background: #252a45;
        }
        
        .series-name {
            font-weight: 500;
            color: #A8DADC;
        }
        
        .current-value {
            font-weight: 600;
            color: #F1FAEE;
            font-size: 1.05em;
        }
        
        .positive {
            color: #52D681;
            font-weight: 500;
        }
        
        .negative {
            color: #FF6B6B;
            font-weight: 500;
        }
        
        .neutral {
            color: #95A5C6;
        }
        
        .chart-cell {
            padding: 5px !important;
            text-align: center;
        }
        
        .mini-chart {
            height: 40px;
            opacity: 0.9;
        }
        
        .download-buttons {
            position: fixed;
            top: 20px;
            right: 20px;
            display: flex;
            gap: 10px;
            z-index: 1000;
        }
        
        .btn {
            padding: 10px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 25px;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        .auto-update {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(26, 31, 58, 0.95);
            padding: 10px 20px;
            border-radius: 25px;
            border: 1px solid #3a4f6f;
            font-size: 0.9em;
        }
        
        .pulse {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #52D681;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.1); }
            100% { opacity: 1; transform: scale(1); }
        }
        
        .footer {
            text-align: center;
            padding: 30px;
            color: #6B7AA1;
            font-size: 0.9em;
        }
        
        @media (max-width: 768px) {
            .container { padding: 10px; }
            .data-table { font-size: 0.85em; }
            .download-buttons { position: static; margin: 20px; }
            h1 { font-size: 1.8em; }
        }
    </style>
</head>
<body>
    <div class="download-buttons">
        <a href="macro_monitor.pdf" class="btn" download>📄 PDF</a>
        <a href="macro_tracker.xlsx" class="btn" download>📊 Excel</a>
    </div>
    
    <div class="header">
        <h1>📊 MACRO MONITOR</h1>
        <div class="last-updated">Last Updated: """ + END.strftime('%B %d, %Y at %I:%M %p ET') + """</div>
    </div>
    
    <div class="container">
"""
    
    # Generate sections
    for section_name, items in SECTIONS:
        html += f"""
        <div class="section">
            <div class="section-header">{section_name}</div>
            <table class="data-table">
                <thead>
                    <tr>
                        <th style="width: 20%">Series</th>
                        <th style="width: 10%">Current</th>
                        <th style="width: 8%">YTD</th>
                        <th style="width: 8%">1M</th>
                        <th style="width: 8%">3M</th>
                        <th style="width: 8%">1Y</th>
                        <th style="width: 8%">3Y</th>
                        <th style="width: 8%">5Y</th>
                        <th style="width: 11%">Trend</th>
                        <th style="width: 11%">Normalized</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for ident, name in items.items():
            s = data.get(ident, pd.Series(dtype=float))
            
            if s.empty:
                html += f"""
                    <tr>
                        <td class="series-name">{name}</td>
                        <td colspan="10" style="text-align: center; color: #6B7AA1;">No data available</td>
                    </tr>
"""
                continue
            
            # Calculate returns
            code = ident.split(":")[1]
            diff_mode = code in YIELD_OR_SPREAD
            end_date = s.index[-1]
            ret = compute_returns(s, end_date, diff_mode)
            current_val = s.iloc[-1]
            
            # Format values
            def format_val(val, is_diff=False):
                if pd.isna(val):
                    return '<span class="neutral">-</span>'
                if is_diff:
                    # For yields/spreads, just show the difference
                    color = "positive" if val > 0 else "negative" if val < 0 else "neutral"
                    return f'<span class="{color}">{val:+.2f}</span>'
                else:
                    # For returns, show percentage
                    color = "positive" if val > 0 else "negative" if val < 0 else "neutral"
                    return f'<span class="{color}">{val:+.1f}%</span>'
            
            # Generate mini charts
            recent_data = s.last('2Y')  # Last 2 years for sparkline
            trend_chart = create_mini_chart(recent_data, name, False)
            norm_chart = create_mini_chart(recent_data, name, True)
            
            html += f"""
                    <tr>
                        <td class="series-name">{name}</td>
                        <td class="current-value">{current_val:.2f}</td>
                        <td>{format_val(ret['YTD'], diff_mode)}</td>
                        <td>{format_val(ret['1M'], diff_mode)}</td>
                        <td>{format_val(ret['3M'], diff_mode)}</td>
                        <td>{format_val(ret['1Y'], diff_mode)}</td>
                        <td>{format_val(ret['3Y'], diff_mode)}</td>
                        <td>{format_val(ret['5Y'], diff_mode)}</td>
                        <td class="chart-cell"><img src="{trend_chart}" class="mini-chart" alt="trend"></td>
                        <td class="chart-cell"><img src="{norm_chart}" class="mini-chart" alt="normalized"></td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
        </div>
"""
    
    html += """
    </div>
    
    <div class="auto-update">
        <span class="pulse"></span>
        Auto-refreshes every 5 minutes
    </div>
    
    <div class="footer">
        <p>Data sources: FRED (Federal Reserve) & Stooq | Updates 5x daily during market hours</p>
        <p>Note: FRED data may have 1-2 day lag</p>
    </div>
    
    <script>
        // Add countdown timer
        let seconds = 300;
        setInterval(() => {
            seconds--;
            if (seconds <= 0) {
                location.reload();
            }
        }, 1000);
        
        // Fade in animation
        document.querySelectorAll('.section').forEach((el, i) => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(20px)';
            setTimeout(() => {
                el.style.transition = 'all 0.5s ease';
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            }, i * 100);
        });
    </script>
</body>
</html>
"""
    
    # Save HTML
    with open(OUT_HTML, 'w') as f:
        f.write(html)
    
    # Save data as JSON for potential API use
    json_data = {
        "updated": END.isoformat(),
        "sections": {}
    }
    
    for section_name, items in SECTIONS:
        json_data["sections"][section_name] = {}
        for ident, name in items.items():
            s = data.get(ident, pd.Series(dtype=float))
            if not s.empty:
                code = ident.split(":")[1]
                diff_mode = code in YIELD_OR_SPREAD
                end_date = s.index[-1]
                ret = compute_returns(s, end_date, diff_mode)
                
                json_data["sections"][section_name][name] = {
                    "current": float(s.iloc[-1]),
                    "returns": {k: float(v) if not pd.isna(v) else None for k, v in ret.items()},
                    "last_date": end_date.isoformat()
                }
    
    with open(OUT_JSON, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"✅ Generated {OUT_HTML}")
    print(f"✅ Generated {OUT_JSON}")
    print(f"📊 Dashboard ready at index.html")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    generate_html_report()
