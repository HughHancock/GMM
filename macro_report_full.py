"""
macro_report_full.py
====================

This script creates a detailed macro report intended to mirror the
structure and visual density of the user's original Global Market
Monitor document.  For each group of instruments (major indices,
sector ETFs, style factors, valuation ratios, volatility & credit,
commodities & energy stocks, interest rates, and real‑estate proxies)
the report produces a summary table of returns and a grid of charts
showing both price performance and a valuation proxy for each
instrument.  The valuation proxy is defined as the ratio of the
instrument's price level to its long‑run median, with ±1 standard
deviation bands to approximate expensive and cheap regimes.  When
available, additional valuation series such as the Shiller CAPE or
Tobin Q are plotted directly.

To generate the report, run:

    python macro_report_full.py

The resulting PDF (``macro_monitor_full.pdf``) and Excel workbook
(``global_macro_tracker_full.xlsx``) are saved in the current
directory.

"""

import pandas as pd
import numpy as np
import pandas_datareader.data as web
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import os

# -----------------------------------------------------------------------------
# Data retrieval utilities (reuse from extended report)
# -----------------------------------------------------------------------------


def load_cape_series() -> pd.Series:
    """Load the Shiller CAPE (PE10) series.

    The original implementation loaded the CAPE ratio from a static CSV file
    distributed with this repository.  That file was based on Robert
    Shiller's historical data and only extended through September 2023.
    In order to provide up‑to‑date values through the current month, we
    instead scrape the monthly table from multpl.com.  This site
    publishes the Shiller PE ratio on a monthly basis and includes
    observations through the present day (e.g., August 2025 at the time
    of writing).  If the scrape fails for any reason (e.g., network
    issues), we fall back to the historical CSV file to avoid raising
    exceptions during report generation.

    Returns
    -------
    pandas.Series
        A monthly time series of the Shiller CAPE ratio indexed by
        datetime.  Missing values are forward‑filled.
    """
    import pandas as pd
    import numpy as np

    # Attempt to fetch the monthly Shiller PE ratio table from multpl.com.
    url = 'https://www.multpl.com/shiller-pe/table/by-month'
    try:
        tables = pd.read_html(url)
        if tables:
            df = tables[0].copy()
            # The table has columns ['Date', 'Value'] as strings.  Convert
            # the dates and values appropriately.
            df['Date'] = pd.to_datetime(df['Date'])
            # Remove commas and convert to float
            df['Value'] = df['Value'].astype(str).str.replace(',', '')
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            df.set_index('Date', inplace=True)
            cape = df['Value'].sort_index()
            cape = cape.replace({0.0: np.nan}).ffill()
            return cape
    except Exception:
        # Scrape failed; fall back to local CSV file.
        pass

    # Fallback: load from the local CSV packaged with this project.
    csv_path = os.path.join(os.path.dirname(__file__), 'data.csv')
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    cape = df['PE10'].copy()
    cape.replace({0.0: np.nan}, inplace=True)
    cape = cape.ffill()
    return cape


def fetch_series_any(identifier: str, start: datetime, end: datetime) -> pd.Series:
    """Fetch a series from FRED, stooq, or local dataset."""
    stooq_tickers = {
        'SPY', 'IWB', 'IWM', 'IWF', 'IWD', 'IWO', 'IWN',
        'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XLC',
        'XOP', 'OIH', 'VNQ', 'IYR'
    }
    ident = identifier.upper()
    try:
        if ident == 'CAPE':
            series = load_cape_series()
        elif ident in stooq_tickers:
            df = web.DataReader(ident, 'stooq', start, end)
            df = df.sort_index()
            if 'Close' in df.columns:
                series = df['Close']
            else:
                series = df.iloc[:, 0]
        else:
            df = web.DataReader(ident, 'fred', start, end)
            series = df.iloc[:, 0]
        series = series.loc[(series.index >= start) & (series.index <= end)].dropna()
        return series
    except Exception as exc:
        print(f"Warning: failed to fetch {identifier}: {exc}")
        return pd.Series(dtype=float)


def compute_returns_for_series(series: pd.Series, end_date: datetime,
                               use_difference: bool = False) -> dict:
    """Compute returns or differences for the given series across several horizons.

    When ``use_difference`` is False (default) the function computes percentage
    returns: (end / start - 1) * 100.0.  When ``use_difference`` is True the
    function instead computes simple differences: end - start.  Yield and
    spread series are better represented as differences rather than relative
    returns because the denominator can be near zero, resulting in absurdly
    large percentage changes (e.g., a jump from 0.1 to 4.3 becomes 4,200%).

    Parameters
    ----------
    series : pandas.Series
        The time series to analyse.
    end_date : datetime
        The end date for computing returns.  Values beyond this date are
        ignored.
    use_difference : bool, default False
        Whether to compute differences instead of percentage returns.

    Returns
    -------
    dict
        A mapping from horizon labels (YTD, 1M, etc.) to the computed
        return or difference.
    """
    horizons = {
        'YTD': datetime(end_date.year, 1, 1),
        '1M': end_date - timedelta(days=30),
        '3M': end_date - timedelta(days=90),
        '1Y': end_date - timedelta(days=365),
        '3Y': end_date - timedelta(days=365 * 3),
        '5Y': end_date - timedelta(days=365 * 5),
        '10Y': end_date - timedelta(days=365 * 10),
    }
    if series.empty:
        return {h: np.nan for h in horizons}
    end_val = series.iloc[-1]
    returns = {}
    for label, start_dt in horizons.items():
        sub = series.loc[(series.index >= start_dt) & (series.index <= end_date)]
        if sub.empty:
            returns[label] = np.nan
            continue
        start_val = sub.iloc[0]
        # If computing percentage returns and start_val is zero, result is undefined.
        if not use_difference and start_val == 0:
            returns[label] = np.nan
        else:
            if use_difference:
                returns[label] = end_val - start_val
            else:
                returns[label] = (end_val / start_val - 1.0) * 100.0
    return returns


def build_summary_table(section_series: dict, start: datetime, end: datetime) -> pd.DataFrame:
    # Define which series should use simple differences rather than percentage returns.
    # Yield and spread series often start near zero, so percentage changes can be
    # misleadingly large.  Use plain differences for these identifiers.
    YIELD_SERIES = {
        # Short‑term yields and longer‑term treasury rates
        'FEDFUNDS', 'TB3MS', 'GS2', 'GS5', 'GS10', 'GS30',
        # Corporate and high‑yield spreads/yields
        'AAA', 'BAA', 'BAMLH0A0HYM2', 'BAMLC0A0CMEY', 'BAMLC0A4CBBB',
        # Valuation ratios (treat changes as differences rather than percentage returns)
        'CAPE', 'QUSR628BIS', 'DDDM01USA156NWDB'
    }
    records = []
    for ident, name in section_series.items():
        series = fetch_series_any(ident, start, end)
        if series.empty:
            continue
        end_date = series.index[-1]
        current_val = series.iloc[-1]
        use_diff = ident.upper() in YIELD_SERIES
        returns = compute_returns_for_series(series, end_date, use_difference=use_diff)
        rec = {'Series': name, 'Current': current_val}
        rec.update(returns)
        records.append(rec)
    df = pd.DataFrame(records)
    return df


def build_valuation_summary(section_series: dict, start: datetime, end: datetime) -> pd.DataFrame:
    """Build a summary table for valuation ratios.

    Instead of computing percentage returns over various horizons, this
    summary reports the current value, the long‑run median (computed on
    the full history available for each series), the percentage
    discount or premium relative to that long‑run median, and the
    standard deviation of the valuation ratio.  These statistics
    mirror the presentation in the user's original report for the
    Shiller CAPE Ratio, Tobin Q Ratio and Market Cap to GDP ratio.

    Parameters
    ----------
    section_series : dict
        Mapping from series identifiers (e.g., 'CAPE') to display names.
    start : datetime
        Start date for retrieving the series.  Only values after this
        date are used to determine the current value.
    end : datetime
        End date for retrieving the series.  Only values up to this
        date are used.

    Returns
    -------
    pandas.DataFrame
        A table with columns: 'Series', 'Current', 'Long‑run Median',
        'Discount/Premium', 'Sigma'.  If a series cannot be fetched
        or lacks sufficient data, it is omitted from the result.
    """
    records = []
    for ident, name in section_series.items():
        try:
            # Fetch the full history to compute median and sigma
            full_series = fetch_series_any(ident, datetime(1900, 1, 1), end)
            # Fetch the recent series to get the current value
            series_recent = fetch_series_any(ident, start, end)
        except Exception:
            continue
        if full_series.empty or series_recent.empty:
            continue
        current_val = series_recent.iloc[-1]
        # Compute long‑run median from full series
        long_run_median = np.nanmedian(full_series.values)
        discount_premium = np.nan
        if long_run_median != 0 and not np.isnan(long_run_median):
            discount_premium = (current_val / long_run_median - 1.0) * 100.0
        # Compute sigma as the standard deviation of the valuation ratio
        ratio_full = full_series / long_run_median if long_run_median not in (0, np.nan) else full_series * np.nan
        finite = ratio_full.replace([np.inf, -np.inf], np.nan).dropna()
        sigma = finite.std() if not finite.empty else np.nan
        records.append({
            'Series': name,
            'Current': current_val,
            'Long-run Median': long_run_median,
            'Discount/Premium': discount_premium,
            'Sigma': sigma,
        })
    df = pd.DataFrame(records)
    # Format numeric values similar to other tables
    for col in ['Current', 'Long-run Median', 'Discount/Premium', 'Sigma']:
        df[col] = df[col].astype(float)
    return df


# -----------------------------------------------------------------------------
# Chart generation helpers
# -----------------------------------------------------------------------------

def compute_normalized(series: pd.Series) -> pd.Series:
    """Normalize a series to 100 at the first observation."""
    if series.empty:
        return series
    base = series.iloc[0]
    if base == 0:
        return series * np.nan
    return series / base * 100.0


def compute_valuation_ratio(series: pd.Series) -> pd.Series:
    """Compute the valuation ratio (series divided by its long‑run median).

    The median is computed over the entire input series.  A constant
    median of zero results in a NaN ratio.
    """
    if series.empty:
        return series
    median = np.nanmedian(series.values)
    if median == 0 or np.isnan(median):
        return pd.Series(index=series.index, data=np.nan)
    return series / median


def compute_valuation_stats(series: pd.Series) -> tuple:
    """Return the median and standard deviation of the valuation ratio."""
    if series.empty:
        return np.nan, np.nan
    ratio = compute_valuation_ratio(series)
    # Use finite values only
    finite = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return np.nan, np.nan
    return finite.median(), finite.std()


def draw_series_panels(
    fig,
    axes,
    series: pd.Series,
    title: str,
    *,
    median_override: float | None = None,
    std_override: float | None = None,
) -> None:
    """Plot two panels for a given series: normalized price and valuation ratio.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing the axes.
    axes : tuple
        Two axes (left for price, right for valuation).
    series : pandas.Series
        Series to plot (trimmed to the display period).
    title : str
        Title to display on both charts.
    median_override : float, optional
        If provided, this value is used as the long‑run median for the valuation
        ratio instead of computing the median from the trimmed series.  This is
        useful for measures like the Shiller CAPE Ratio, where the
        long‑run median should be based on over a century of data rather than a
        short window.  When ``median_override`` is specified, the ratio is
        computed as ``series / median_override``.
    std_override : float, optional
        If provided, this value is used as the standard deviation of the
        valuation ratio instead of computing it from the trimmed series.
    """
    price_ax, val_ax = axes
    # Plot normalized price
    if not series.empty:
        norm_series = compute_normalized(series)
        price_ax.plot(norm_series.index, norm_series.values, color='tab:blue')
    price_ax.set_title(f"{title} – Normalized", fontsize=8)
    price_ax.tick_params(axis='both', labelsize=6)
    price_ax.grid(True, alpha=0.3)
    # Plot valuation ratio
    if not series.empty:
        if median_override is not None:
            # Compute ratio using the supplied long‑run median
            ratio = series / median_override
        else:
            # Use trimmed median
            ratio = compute_valuation_ratio(series)
        # Determine standard deviation
        if std_override is not None:
            std_val = std_override
        else:
            _, std_val = compute_valuation_stats(series)
        val_ax.plot(ratio.index, ratio.values, color='tab:green')
        # Horizontal lines at median and ±1 standard deviation
        val_ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
        if std_val is not None and not np.isnan(std_val):
            val_ax.axhline(1.0 + std_val, color='gray', linestyle=':', linewidth=0.6)
            val_ax.axhline(1.0 - std_val, color='gray', linestyle=':', linewidth=0.6)
    val_ax.set_title(f"{title} – Valuation Ratio", fontsize=8)
    val_ax.tick_params(axis='both', labelsize=6)
    val_ax.grid(True, alpha=0.3)


def draw_full_section(fig, section_title: str, summary_df: pd.DataFrame,
                      series_map: dict, start: datetime, end: datetime,
                      table_height: float = 0.22, header_height: float = 0.07):
    """Draw a detailed page with improved spacing for a subset of series.

    Each page consists of a header, a summary table for the included
    series, and vertically stacked panels for each series.  For
    presentation‑ready spacing, each series occupies a full row with
    two stacked plots (price and valuation) across the width of the
    page.  The heights of the header and table can be tuned via
    ``header_height`` and ``table_height``.
    """
    n_series = len(series_map)
    # Base axes for header and background
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    # Draw header bar
    ax.add_patch(patches.Rectangle((0, 1 - header_height), 1, header_height,
                                   transform=ax.transAxes, color='#003366'))
    ax.text(0.02, 1 - header_height / 2, section_title, color='white',
            fontsize=18, va='center', weight='bold', transform=ax.transAxes)
    # Summary table area
    # Adapt table height based on number of rows.  A taller table is used for more
    # series; a smaller table leaves more space for charts when there are only
    # one or two series.  Bound the height to a reasonable range.
    n_rows = len(summary_df)
    dynamic_table_height = 0.08 + 0.04 * max(n_rows - 1, 0)
    table_height_eff = min(table_height, dynamic_table_height)
    table_top = 1 - header_height - table_height_eff
    table_ax = fig.add_axes([0.05, table_top, 0.9, table_height_eff])
    table_ax.axis('off')
    # Format summary_df for display
    display_df = summary_df.copy()
    # If the summary is empty (e.g., because all series in the subset failed
    # to fetch), insert a placeholder row so that the table call does not
    # raise an IndexError.  This also lets the user know that data was
    # unavailable for the requested series.
    if display_df.empty:
        display_df = pd.DataFrame({'Series': ['No data available']})
    for col in display_df.columns:
        if col != 'Series':
            display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else 'NA')
    col_labels = list(display_df.columns)
    cell_text = display_df.values.tolist()
    table = table_ax.table(cellText=cell_text, colLabels=col_labels,
                           cellLoc='right', loc='upper center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(col_labels))))
    # -----------------------------------------------------------------------
    # Series panels
    #
    # We allocate the remaining vertical space (between the summary table and
    # a small bottom margin) to a grid of 2*n_series rows.  Each series
    # occupies two stacked rows: the top for its normalized price and the
    # bottom for its valuation ratio.  By using a GridSpec we ensure the
    # panels fill the available space uniformly and avoid manual height
    # calculations that can leave excess white space or cause overlap.
    #
    # Define boundaries for the panel area.
    # Leave virtually no gap between the summary table and the charts.  The
    # subsequent GridSpec hspace will manage spacing between individual
    # subplots.  A zero gap here maximizes vertical real estate for
    # charts and eliminates the large band of white space seen in
    # earlier versions.
    gap_after_table = 0.0
    panel_top = table_top - gap_after_table
    panel_bottom = 0.05      # bottom margin
    # Create a GridSpec with 2*n_series rows spanning the panel area.
    import matplotlib.gridspec as gridspec
    if n_series > 0:
        gs = gridspec.GridSpec(
            nrows=2 * n_series,
            ncols=1,
            left=0.05,
            right=0.95,
            bottom=panel_bottom,
            top=panel_top,
            # A moderate hspace ensures separation between subplots without
            # wasting vertical space.  Increase to 0.18 to prevent
            # overlapping titles and x‑tick labels between panels.
            hspace=0.18
        )
        for idx, (ident, display_name) in enumerate(series_map.items()):
            # Each series uses two consecutive rows in the GridSpec
            price_ax = fig.add_subplot(gs[2 * idx, 0])
            val_ax = fig.add_subplot(gs[2 * idx + 1, 0])
            # Fetch the series for the displayed period
            series = fetch_series_any(ident, start, end)
            # Compute long‑run median and standard deviation using the full history
            # available for this series.  Use a broad start date far in the past to
            # capture as much history as possible.  Wrap in try/except to avoid
            # failures for series that cannot be fetched in the extended range.
            median_override = None
            std_override = None
            try:
                full_series = fetch_series_any(ident, datetime(1900, 1, 1), end)
                if not full_series.empty:
                    med_full = np.nanmedian(full_series.values)
                    if med_full != 0 and not np.isnan(med_full):
                        median_override = med_full
                        ratio_full = full_series / med_full
                        finite = ratio_full.replace([np.inf, -np.inf], np.nan).dropna()
                        if not finite.empty:
                            std_override = finite.std()
            except Exception:
                pass
            draw_series_panels(
                fig,
                (price_ax, val_ax),
                series,
                display_name,
                median_override=median_override,
                std_override=std_override,
            )
            # Hide x‑axis tick labels on the price panels to save vertical space.
            price_ax.tick_params(axis='x', labelbottom=False)
            # For all but the last series, hide x‑axis labels on the valuation panels too.
            if idx < n_series - 1:
                val_ax.tick_params(axis='x', labelbottom=False)
    else:
        # No series: nothing to draw
        pass


def build_full_report(start: datetime, end: datetime, pdf_path: str, excel_path: str,
                      max_series_per_page: int = 4):
    """Generate a comprehensive macro report with two‑panel charts for each series.

    If a section contains more series than ``max_series_per_page``, the section
    will be split across multiple pages to improve spacing and readability.
    Each page shows only the subset of series assigned to it, along with a
    corresponding summary table.  The Excel workbook retains the full
    summary table for each section.
    """
    sections = [
        {
            'title': 'Major Equity Indices',
            'series': {
                'SP500': 'S&P 500 Index',
                'IWB': 'Russell 1000 ETF',
                'IWM': 'Russell 2000 ETF',
            }
        },
        {
            'title': 'S&P Sector ETFs',
            'series': {
                'XLB': 'Materials',
                'XLE': 'Energy',
                'XLF': 'Financials',
                'XLI': 'Industrials',
                'XLK': 'Technology',
                'XLP': 'Consumer Staples',
                'XLU': 'Utilities',
                'XLV': 'Health Care',
                'XLY': 'Consumer Discretionary',
                'XLC': 'Communication Services',
            }
        },
        {
            'title': 'Growth vs Value',
            'series': {
                'IWF': 'Russell 1000 Growth',
                'IWD': 'Russell 1000 Value',
                'IWO': 'Russell 2000 Growth',
                'IWN': 'Russell 2000 Value',
            }
        },
        {
            'title': 'Valuation Ratios',
            'series': {
                'CAPE': 'Shiller CAPE Ratio',
                'QUSR628BIS': 'Tobin Q Ratio',
                'DDDM01USA156NWDB': 'Market Cap to GDP',
            }
        },
        {
            'title': 'Volatility & Credit',
            'series': {
                'VIXCLS': 'CBOE VIX Index',
                'BAMLH0A0HYM2': 'High‑Yield Bond Spread',
                'BAMLC0A0CMEY': 'AAA Corporate Yield',
                'BAMLC0A4CBBB': 'BBB Corporate Yield',
            }
        },
        {
            'title': 'Commodities & Energy Stocks',
            'series': {
                'DCOILWTICO': 'WTI Crude (USD/bbl)',
                'DCOILBRENTEU': 'Brent Crude (USD/bbl)',
                'DHHNGSP': 'Natural Gas (USD/MMBtu)',
                'XOP': 'Oil & Gas E&P ETF',
                'OIH': 'Oil Services ETF',
            }
        },
        {
            'title': 'Interest Rates & Spreads',
            'series': {
                'FEDFUNDS': 'Fed Funds Rate',
                'TB3MS': '3‑Month T‑Bill',
                'GS2': '2‑Year Treasury',
                'GS5': '5‑Year Treasury',
                'GS10': '10‑Year Treasury',
                'GS30': '30‑Year Treasury',
                'AAA': 'AAA Corporate Yield',
                'BAA': 'BAA Corporate Yield',
            }
        },
        {
            'title': 'Real Estate & Mortgage',
            'series': {
                'VNQ': 'US Real Estate ETF (VNQ)',
                'IYR': 'US Real Estate ETF (IYR)',
                'CSUSHPINSA': 'Case‑Shiller Home Price Index',
                'MORTGAGE30US': '30Y Mortgage Rate',
            }
        },
    ]
    # Create Excel writer for summary tables
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        with PdfPages(pdf_path) as pdf:
            for section in sections:
                title = section['title']
                series_map = section['series']
                # Build full summary table for the section.  Use a special
                # summary for valuation ratios that reports the long‑run
                # median, discount/premium and sigma instead of time‑horizon
                # returns.
                if title == 'Valuation Ratios':
                    full_summary_df = build_valuation_summary(series_map, start, end)
                else:
                    full_summary_df = build_summary_table(series_map, start, end)
                sheet_name = title[:31]
                # If no data was fetched for any series in this section, write a
                # placeholder sheet and skip generating report pages.
                if full_summary_df.empty:
                    placeholder_df = pd.DataFrame({'Notice': ['No data available for this section']})
                    placeholder_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    continue
                # Write the populated summary to Excel
                full_summary_df.to_excel(writer, sheet_name=sheet_name, index=False)
                # Determine which of the requested series actually returned data
                series_items = list(series_map.items())
                available_items = [(k, v) for (k, v) in series_items
                                   if not fetch_series_any(k, start, end).empty]
                if not available_items:
                    continue
                num_series = len(available_items)
                num_pages = int(np.ceil(num_series / max_series_per_page))
                for page_idx in range(num_pages):
                    # Determine subset of series for this page
                    start_idx = page_idx * max_series_per_page
                    end_idx = start_idx + max_series_per_page
                    subset_items = available_items[start_idx:end_idx]
                    subset_map = dict(subset_items)
                    # Filter summary data accordingly
                    subset_summary = full_summary_df[full_summary_df['Series'].isin(subset_map.values())]
                    # Create figure for this page
                    fig = plt.figure(figsize=(8.5, 11))
                    page_title = title
                    if num_pages > 1:
                        page_title = f"{title} (Page {page_idx + 1}/{num_pages})"
                    draw_full_section(
                        fig, page_title, subset_summary, subset_map, start, end,
                        table_height=0.18, header_height=0.07)
                    pdf.savefig(fig)
                    plt.close(fig)
    print(f"Full macro report saved to {pdf_path}")
    print(f"Full Excel tracker saved to {excel_path}")


def main():
    start_date = datetime(2018, 1, 1)
    end_date = datetime.now()
    pdf_path = 'macro_monitor_full.pdf'
    excel_path = 'global_macro_tracker_full.xlsx'
    # Use 3 series per page for improved spacing
    # Use 2 series per page for generous spacing and to avoid overlapping
    build_full_report(start_date, end_date, pdf_path, excel_path, max_series_per_page=2)


if __name__ == '__main__':
    main()