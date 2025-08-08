"""
macro_update_scheduler.py
=================================

This script automates the generation of the macro monitor report during
U.S. market trading hours.  When run, it continuously checks the
current time in the user's local timezone (America/Los_Angeles) and,
if the time falls within regular trading hours (Monday through Friday,
6:30 AM to 1:00 PM Pacific Time), it regenerates the macro report and
Excel workbook every ten minutes.  Outside of trading hours the
script sleeps without generating reports.

Usage
-----

Run this script in a Python environment that has the same
dependencies as `macro_report_full.py`.  For example:

```
python macro_update_scheduler.py
```

Notes
-----

* This script calls `macro_report_full.py` as a subprocess on each
  iteration.  Ensure that the report script is in the same directory
  or adjust the path accordingly.
* The schedule uses the local timezone to determine trading hours.
* If you wish to run the script as a background service or cron job,
  wrap it in appropriate process management (e.g., systemd, supervisord,
  or a cron entry that launches at system boot).
* The generated report files (PDF and Excel) will be overwritten on
  each run.
"""

import subprocess
import time
from datetime import datetime, time as dtime
try:
    # Python 3.9+ standard library for timezone handling
    from zoneinfo import ZoneInfo
except ImportError:
    # Fall back to pytz if zoneinfo is unavailable
    from pytz import timezone as ZoneInfo


def within_market_hours(now: datetime) -> bool:
    """Return True if the given datetime falls within U.S. market hours.

    Market hours are defined as Monday through Friday from 6:30 AM to
    1:00 PM (inclusive) in the user's local timezone (Pacific Time).

    Parameters
    ----------
    now : datetime
        A timezone-aware datetime object representing the current
        time.

    Returns
    -------
    bool
        True if now is a weekday and the time is within trading hours;
        otherwise False.
    """
    # Monday=0, Sunday=6
    if now.weekday() >= 5:
        return False
    start = dtime(6, 30)  # 6:30 AM
    end = dtime(13, 0)    # 1:00 PM
    return start <= now.time() <= end


def run_report():
    """Invoke the macro report script to generate updated files."""
    print(f"[{datetime.now()}] Running macro report generation...")
    result = subprocess.run(['python', 'macro_report_full.py'])
    if result.returncode != 0:
        print(f"[{datetime.now()}] Report generation failed with status {result.returncode}")
    else:
        print(f"[{datetime.now()}] Report generation completed.")


def main():
    # Use the America/Los_Angeles timezone
    try:
        tz = ZoneInfo('America/Los_Angeles')
    except Exception:
        # If zoneinfo fails, fallback to using pytz via ZoneInfo alias
        tz = ZoneInfo('America/Los_Angeles')
    print("Starting macro update scheduler... Press Ctrl+C to stop.")
    while True:
        now = datetime.now(tz)
        if within_market_hours(now):
            run_report()
        else:
            print(f"[{now}] Outside market hours; skipping report generation.")
        # Sleep for 10 minutes
        time.sleep(600)


if __name__ == '__main__':
    main()