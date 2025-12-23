#!/usr/bin/env python3
"""
Small smoke-test script for common ML/viz Python packages.

Usage examples:
  python lib_port_test.py                  # test defaults: tensorflow, plotly
  python lib_port_test.py --libs tensorflow plotly
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from typing import Callable, Dict, List


def check_tensorflow() -> Dict[str, object]:
    import tensorflow as tf

    devices = [d.name for d in tf.config.list_physical_devices("GPU")]
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0], [6.0]])
    result = tf.matmul(a, b)

    return {
        "version": tf.__version__,
        "gpu_devices": devices,
        "matmul_result": result.numpy().tolist(),
    }


def check_plotly() -> Dict[str, object]:
    import plotly
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Bar(x=["A", "B"], y=[1, 2])])
    fig_dict = fig.to_dict()

    return {
        "version": plotly.__version__,
        "figure_keys": list(fig_dict.keys()),
    }


LIB_CHECKS: Dict[str, Callable[[], Dict[str, object]]] = {
    "tensorflow": check_tensorflow,
    "plotly": check_plotly,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick import/run checks for popular libs.")
    parser.add_argument(
        "--libs",
        nargs="+",
        default=list(LIB_CHECKS.keys()),
        help=f"Which libraries to check. Known: {', '.join(LIB_CHECKS)}",
    )
    args = parser.parse_args()

    results: List[Dict[str, object]] = []
    for lib in args.libs:
        fn = LIB_CHECKS.get(lib)
        if fn is None:
            results.append(
                {"library": lib, "status": "skipped", "error": "unknown library name"}
            )
            continue

        try:
            payload = fn()
            results.append({"library": lib, "status": "ok", "details": payload})
        except Exception:  # pragma: no cover - diagnostic output
            results.append(
                {
                    "library": lib,
                    "status": "failed",
                    "error": traceback.format_exc(),
                }
            )

    json.dump(results, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

