
import matplotlib.pyplot as plt
import json
from pathlib import Path
import pandas as pd

from visualize_utils import *






prefill = pd.DataFrame.from_dict(load_results("../results-0413-2-H100/prefill_csjf"))
pp = pd.DataFrame.from_dict(load_results("../results-0414-2-H100-NVLink/pp"))
tp = pd.DataFrame.from_dict(load_results("../results-0414-2-H100-NVLink/tp"))

prefill.sort_values(by="qps", inplace=True)
pp.sort_values(by="qps", inplace=True)
tp.sort_values(by="qps", inplace=True)


fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(prefill["qps"], 
        prefill["p99_ttft_ms"] / 1000, 
        label=method2name["prefill"], 
        color=colors["prefill"],
        linewidth=2,
        marker='v',
        markersize=10)
ax.plot(pp["qps"], 
        pp["p99_ttft_ms"] / 1000, 
        label=method2name["pp"], 
        color=colors["pp"],
        linewidth=2,
        marker='v',
        markersize=10)
ax.plot(tp["qps"], 
        tp["p99_ttft_ms"] / 1000, 
        label=method2name["tp"], 
        color=colors["tp"],
        linewidth=2,
        marker='v',
        markersize=10)
ax.legend(loc="lower right", framealpha=0.7)
ax.grid()
ax.set_xlabel("Query per second")
ax.set_ylabel("P99 latency (s)")
ax.set_ylim(bottom=0)
ax.set_ylim(top=500)

bbox_props = dict(boxstyle="rarrow", fc=(1,1,1), ec="grey", lw=2)
ax.text(0.05, 0.95, "Better", ha="left", 
va="top", rotation=-45, bbox=bbox_props, c='grey',
transform=ax.transAxes
)

fig.tight_layout()
fig.savefig("figures/h100_end2end_workload2_NVLink_p99_ttft.pdf", bbox_inches="tight")
