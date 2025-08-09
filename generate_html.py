#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import json
import base64
from io import BytesIO
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web

# Configuration
# Use naive datetime for data fetching (pandas_datareader requirement)
END = datetime.now()
START = datetime(2018, 1, 1)

# Calculate ET time for display only
utc_offset = -5  # EST
if 3 <= END.month <= 11:  # Rough DST calculation (March-November)
    utc_offset = -4  # EDT
END_ET = END + timedelta(hours=utc_offset)

OUT_HTML = "index.html"
OUT_JSON = "data.json"

# Treat these as point changes (not percent)
YIELD_OR_SPREAD = {
    "DFF","FEDFUNDS","TB3MS","DGS3MO","DGS2","DGS5","DGS10","DGS30",
    "GS2","GS5","GS10","GS30",
    "AAA","BAA","BAMLH0A0HYM2","BAMLC0A0CMEY","BAMLC0A4CBBBEY",
    "MORTGAGE30US"
}

SECTIONS = (
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

# Helper functions
def fetch_series(ident, start=START, end=END):
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

def compute_returns(s, end_date, diff_mode):
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

def normalized_100(s):
    if s.empty: 
        return s
    base = s.iloc[0]
    if base == 0 or pd.isna(base):
        return pd.Series(index=s.index, dtype=float)
    return (s / base) * 100.0

def create_chart(s, title, is_normalized=False, is_yield=False):
    """Create a chart similar to the PDF style."""
    if s.empty or s.isna().all():
        return ""
    
    fig, ax = plt.subplots(figsize=(5, 2), dpi=100)
    
    if is_normalized:
        s_plot = normalized_100(s)
        title_suffix = " - Normalized (2018=100)"
    else:
        s_plot = s
        title_suffix = f" - {'Yield (%)' if is_yield else 'Level'}"
    
    # Plot with style similar to PDF
    ax.plot(s_plot.index, s_plot.values, linewidth=1, color='#1e40af')
    ax.fill_between(s_plot.index, s_plot.values, alpha=0.1, color='#1e40af')
    
    # Add grid
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Style the plot
    ax.set_title(f"{title}{title_suffix}", fontsize=9, pad=5)
    ax.tick_params(labelsize=7)
    
    # Format x-axis to show years
    years = pd.date_range(start=s_plot.index[0], end=s_plot.index[-1], freq='YS')
    ax.set_xticks(years)
    ax.set_xticklabels([d.strftime('%Y') for d in years], fontsize=7)
    
    # Add min/max markers
    if len(s_plot) > 0:
        max_idx = s_plot.idxmax()
        min_idx = s_plot.idxmin()
        ax.plot(max_idx, s_plot[max_idx], '^', markersize=5, color='#16a34a')
        ax.plot(min_idx, s_plot[min_idx], 'v', markersize=5, color='#dc2626')
        
        # Add current value annotation
        current_val = s_plot.iloc[-1]
        ax.annotate(f'{current_val:.1f}', 
                   xy=(s_plot.index[-1], current_val),
                   xytext=(5, 0), 
                   textcoords='offset points',
                   fontsize=7,
                   bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', transparent=False, facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    
    return f"data:image/png;base64,{image_base64}"

def generate_html_report():
    """Generate the HTML dashboard that looks like the PDF."""
    print(f"Generating HTML Dashboard...")
    print(f"Fetching data from {START.strftime('%Y-%m-%d')} to {END.strftime('%Y-%m-%d')}")
    
    # Fetch all data
    data = {}
    for section, items in SECTIONS:
        for ident in items.keys():
            if ident not in data:
                print(f"  Fetching {ident}...")
                data[ident] = fetch_series(ident)
    
    # Generate HTML that looks like the PDF
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="300">
    <title>Macro Monitor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: Arial, Helvetica, sans-serif;
            background: white;
            color: #000;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #333;
        }
        
        h1 {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .timestamp {
            font-size: 12px;
            color: #666;
        }
        
        .section {
            margin-bottom: 50px;
            page-break-inside: avoid;
        }
        
        .section-title {
            font-size: 18px;
            font-weight: bold;
            background: #f0f0f0;
            padding: 8px;
            margin-bottom: 0;
            border: 1px solid #ccc;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
            margin-bottom: 5px;
        }
        
        th {
            background: #E9EEF6;
            padding: 6px;
            text-align: center;
            font-weight: bold;
            border: 1px solid #ccc;
            font-size: 10px;
        }
        
        td {
            padding: 4px 6px;
            border: 1px solid #ddd;
            text-align: right;
        }
        
        td:first-child {
            text-align: left;
            font-weight: 500;
            background: #fafafa;
        }
        
        tr:hover {
            background: #f9f9f9;
        }
        
        .positive {
            color: #008000;
        }
        
        .negative {
            color: #cc0000;
        }
        
        .current-value {
            font-weight: bold;
        }
        
        .charts-container {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            padding: 15px;
            background: #fafafa;
            border: 1px solid #ddd;
            border-top: none;
            margin-bottom: 20px;
        }
        
        .chart-pair {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            padding: 10px;
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
        }
        
        .chart-wrapper {
            text-align: center;
        }
        
        .chart-wrapper img {
            width: 100%;
            max-width: 350px;
            height: auto;
            border: 1px solid #e0e0e0;
        }
        
        .series-label {
            font-weight: bold;
            font-size: 11px;
            margin-bottom: 5px;
            color: #333;
            text-align: center;
        }
        
        .nav-buttons {
            position: fixed;
            top: 20px;
            right: 20px;
            display: flex;
            gap: 10px;
            z-index: 1000;
        }
        
        .btn {
            padding: 8px 16px;
            background: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-size: 12px;
            border: none;
            cursor: pointer;
        }
        
        .btn:hover {
            background: #45a049;
        }
        
        .btn-secondary {
            background: #008CBA;
        }
        
        .btn-secondary:hover {
            background: #007399;
        }
        
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ccc;
            text-align: center;
            font-size: 10px;
            color: #666;
        }
        
        .auto-refresh {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #f0f0f0;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 11px;
            border: 1px solid #ccc;
        }
        
        @media print {
            .nav-buttons, .auto-refresh {
                display: none;
            }
            .section {
                page-break-inside: avoid;
            }
        }
        
        @media (max-width: 768px) {
            body {
                padding: 10px;
            }
            table {
                font-size: 9px;
            }
            .charts-container {
                grid-template-columns: 1fr;
            }
            .chart-pair {
                grid-template-columns: 1fr;
            }
            .nav-buttons {
                position: static;
                margin-bottom: 20px;
                justify-content: center;
            }
        }
    </style>
</head>
<body>
    <div class="nav-buttons">
        <a href="macro_monitor.pdf" class="btn" download>📄 Download PDF</a>
        <a href="macro_tracker.xlsx" class="btn btn-secondary" download>📊 Download Excel</a>
    </div>
    
    <div class="header">
        <h1>MACRO MONITOR</h1>
        <div class="timestamp">Data as of: """ + END_ET.strftime('%B %d, %Y at %I:%M %p') + """ ET</div>
    </div>
"""
    
    # Generate sections with prominent charts like the PDF
    for section_name, items in SECTIONS:
        html += f"""
    <div class="section">
        <div class="section-title">{section_name}</div>
        <table>
            <thead>
                <tr>
                    <th style="width: 25%">Series</th>
                    <th style="width: 12.5%">Current</th>
                    <th style="width: 10.5%">YTD</th>
                    <th style="width: 10.5%">1M</th>
                    <th style="width: 10.5%">3M</th>
                    <th style="width: 10.5%">1Y</th>
                    <th style="width: 10.5%">3Y</th>
                    <th style="width: 10.5%">5Y</th>
                </tr>
            </thead>
            <tbody>
"""
        
        # First pass: generate table rows
        for ident, name in items.items():
            s = data.get(ident, pd.Series(dtype=float))
            
            if s.empty:
                html += f"""
                <tr>
                    <td>{name}</td>
                    <td colspan="7" style="text-align: center; color: #999;">No data</td>
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
                    return ""
                if val > 0:
                    return f'<span class="positive">{val:.2f}</span>'
                elif val < 0:
                    return f'<span class="negative">{val:.2f}</span>'
                else:
                    return f'{val:.2f}'
            
            html += f"""
                <tr>
                    <td>{name}</td>
                    <td class="current-value">{current_val:.2f}</td>
                    <td>{format_val(ret['YTD'], diff_mode)}</td>
                    <td>{format_val(ret['1M'], diff_mode)}</td>
                    <td>{format_val(ret['3M'], diff_mode)}</td>
                    <td>{format_val(ret['1Y'], diff_mode)}</td>
                    <td>{format_val(ret['3Y'], diff_mode)}</td>
                    <td>{format_val(ret['5Y'], diff_mode)}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
"""
        
        # Second pass: generate charts grid
        html += """
        <div class="charts-container">
"""
        
        for ident, name in items.items():
            s = data.get(ident, pd.Series(dtype=float))
            
            if not s.empty:
                code = ident.split(":")[1]
                is_yield = code in YIELD_OR_SPREAD
                
                # Generate both charts
                level_chart = create_chart(s, name, False, is_yield)
                norm_chart = create_chart(s, name, True, is_yield)
                
                html += f"""
            <div class="chart-pair">
                <div class="chart-wrapper">
                    <img src="{norm_chart}" alt="{name} Normalized">
                </div>
                <div class="chart-wrapper">
                    <img src="{level_chart}" alt="{name} Level">
                </div>
            </div>
"""
        
        html += """
        </div>
    </div>
"""
    
    html += """
    <div class="footer">
        <p>Data sources: FRED (Federal Reserve Economic Data) & Stooq</p>
        <p>Note: FRED data may have 1-2 day lag | Updates 5x daily during market hours</p>
    </div>
    
    <div class="auto-refresh">
        <span id="refresh-text">Auto-refresh in <span id="countdown">300</span>s</span>
    </div>
    
    <script>
        // Countdown timer
        let seconds = 300;
        const countdownEl = document.getElementById('countdown');
        
        setInterval(() => {
            seconds--;
            if (countdownEl) {
                countdownEl.textContent = seconds;
            }
            if (seconds <= 0) {
                location.reload();
            }
        }, 1000);
    </script>
</body>
</html>
"""
    
    # Save HTML
    with open(OUT_HTML, 'w') as f:
        f.write(html)
    
    # Save data as JSON
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
