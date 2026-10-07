#!/usr/bin/env python3
"""Compatibility entry point for the unified matching benchmark.

The maintained implementation is :mod:`matching_comparison_all`. Re-exporting
its public objects keeps older notebooks/tests working while ensuring every
entry point uses the bundled user-provided AdaLAM source.
"""

from matching_comparison_all import *  # noqa: F401,F403
from matching_comparison_all import main as _main


if __name__ == "__main__":
    raise SystemExit(_main())
