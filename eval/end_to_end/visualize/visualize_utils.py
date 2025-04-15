
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import json

plt.rcParams["font.size"] = 25
matplotlib.rcParams['pdf.fonttype'] = 42

method2name = {
    "prefill": "LMPrefill",
    "chunked": "Chunked Prefill",
    "pp": "Pipeline Parallel",
    "tp": "Tensor Parallel",
    "vanilla": "PagedAttention"
}

colors = {
    "prefill": "#004daf",  # (0, 77, 175)
    "chunked":      "#33a02c",  # (51, 160, 44)
    "pp":           "#ff9900",  # (255, 153, 0)
    "tp":           "#ed1b3a",  # (237, 27, 58)
    "vanilla":      "#328E6E"
}

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