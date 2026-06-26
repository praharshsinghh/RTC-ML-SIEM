#!/usr/bin/env python3
"""
run_dashboard.py — RTC SIEM Dashboard Launcher
================================================
Convenience wrapper that starts the Streamlit dashboard.

Usage
-----
    python run_dashboard.py                    # default port 8501
    python run_dashboard.py --port 8080        # custom port
    python run_dashboard.py --no-browser       # headless / server mode

This script simply delegates to `streamlit run` with the correct entry point
(src/dashboard/app.py) and sensible defaults.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_APP = Path(__file__).parent / "src" / "dashboard" / "app.py"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch the RTC SIEM Streamlit dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--port", type=int, default=8501,
                        help="Streamlit server port (default: 8501)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open a browser tab automatically")
    args = parser.parse_args()

    if not _APP.exists():
        print(f"❌ Dashboard entry point not found: {_APP}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(_APP),
        "--server.port", str(args.port),
        "--server.headless", "true" if args.no_browser else "false",
        "--theme.base", "dark",
    ]

    print(f"🛡️  RTC SIEM Dashboard")
    print(f"    URL  : http://localhost:{args.port}")
    print(f"    Entry: {_APP.relative_to(Path.cwd())}")
    print(f"    Press Ctrl+C to stop.\n")

    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
