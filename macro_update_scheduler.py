name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install --no-cache-dir \
      pandas pandas_datareader matplotlib \
      openpyxl xlsxwriter lxml html5lib beautifulsoup4

name: Run macro monitor once (manual)

on:
  workflow_dispatch:  # adds a "Run workflow" button

permissions:
  contents: write     # needed so the job can push the PDF/XLSX back

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install --no-cache-dir \
            pandas pandas_datareader matplotlib \
            openpyxl lxml html5lib beautifulsoup4

      # Option 1 (recommended for a quick one-shot):
      - name: Generate macro report (one-shot)
        run: python macro_report_full.py

      # --- If you prefer to use your scheduler in "run once" mode, 
      # --- replace the step above with this one (and keep the flag):
      # - name: Generate via scheduler (one-shot/force)
      #   run: python macro_update_scheduler.py --once --force

      - name: Commit and push updated reports
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add macro_monitor_full.pdf global_macro_tracker_full.xlsx || true
          git commit -m "Run-once: update reports" || echo "Nothing to commit"
          git push
