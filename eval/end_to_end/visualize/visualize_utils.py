import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import json

plt.rcParams["font.size"] = 27
matplotlib.rcParams['pdf.fonttype'] = 42

method2name = {
    "prefill": "LMPrefill",
    "chunked": "Chunked Prefill",
    "pp": "Pipeline Parallel",
    "tp": "Tensor Parallel",
    "vanilla": "PagedAttention"
}

colors = {
    "prefill":      "#004daf",  # (0, 77, 175)
    "chunked":      "#33a02c",  # (51, 160, 44)
    "pp":           "#ff9900",  # (255, 153, 0)
    "tp":           "#ed1b3a",  # (237, 27, 58)
    "vanilla":      "#9ACBD0",  # (154, 203, 208)
}

markers = {
    "prefill":      "v",  # 三角形
    "chunked":      "s",  # 正方形
    "pp":           "o",  # 圆形
    "tp":           "^",  # 菱形
    "vanilla":      "*",  # 上三角形
}

marker_sizes = {
    "prefill":      16,
    "chunked":      16,
    "pp":           16,
    "tp":           16,
    "vanilla":      16,
}


line_width = {
    "prefill":      3,
    "chunked":      3,
    "pp":           3,
    "tp":           3,
    "vanilla":      3,
}

legend_sizes = {
    "prefill":      10,
    "chunked":      10,
    "pp":           10,
    "tp":           10,
    "vanilla":      10,
}


e2e_figure_size = (8, 4.3)

def load_results(results_dir):
    results = []
    results_dir = Path(results_dir).resolve()
    for path in Path(results_dir).glob("*.json"):
        stem = path.stem
        args = stem.split("__")
        kwargs = {}

        for i in range(0, len(args), 2):
            kwargs[args[i]] = float(args[i+1])

        with open(results_dir/path, "rb") as f:
            data = json.load(f)
            kwargs.update(data)
        results.append(kwargs)
    return results