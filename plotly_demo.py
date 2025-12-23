"""
Small, self-contained Plotly demo you can run inside the devcontainer.

Usage:
  python plotly_demo.py

It writes an HTML file and opens it in your default browser (inside the
container it may print the file path instead).
"""

import math
from pathlib import Path

import plotly.express as px


def make_data(num_points: int = 200):
    """Create a simple sine/cosine dataset."""
    xs, sin_y, cos_y = [], [], []
    for i in range(num_points):
        x = i / 10.0
        xs.append(x)
        sin_y.append(math.sin(x))
        cos_y.append(math.cos(x))
    return xs, sin_y, cos_y


def main():
    xs, sin_y, cos_y = make_data()
    fig = px.line(
        x=xs,
        y=[sin_y, cos_y],
        labels={"x": "x", "value": "value", "variable": "series"},
        title="Sine vs Cosine",
    )
    fig.update_layout(template="plotly_dark")

    out_path = Path("plotly_demo.html").resolve()
    fig.write_html(out_path)
    print(f"Wrote {out_path}")
    try:
        fig.show()
    except Exception as exc:  # pragma: no cover - best-effort display
        print(f"Could not open browser: {exc}")


if __name__ == "__main__":
    main()






