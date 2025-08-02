import matplotlib.pyplot as plt
import json
from pathlib import Path
import pandas as pd

from visualize_utils import *



DIR = '../results-0801-L4-WL-1'


prefill = pd.DataFrame.from_dict(load_results(f"{DIR}/prefill_csjf"))
vanilla = pd.DataFrame.from_dict(load_results(f"{DIR}/vanilla"))
chunked = pd.DataFrame.from_dict(load_results(f"{DIR}/chunked"))
pp = pd.DataFrame.from_dict(load_results(f"{DIR}/pp"))
tp = pd.DataFrame.from_dict(load_results(f"{DIR}/tp"))

prefill.sort_values(by="qps", inplace=True)
vanilla.sort_values(by="qps", inplace=True)
chunked.sort_values(by="qps", inplace=True)
pp.sort_values(by="qps", inplace=True)
tp.sort_values(by="qps", inplace=True)


fig, ax = plt.subplots(figsize=e2e_figure_size)

ax.plot(prefill["qps"], 
        prefill["mean_ttft_ms"] / 1000, 
        label=method2name["prefill"], 
        color=colors["prefill"],
        linewidth=line_width["prefill"],
        marker=markers["prefill"],
        markersize=marker_sizes["prefill"])
ax.plot(vanilla["qps"], 
        vanilla["mean_ttft_ms"] / 1000, 
        label=method2name["vanilla"], 
        color=colors["vanilla"],
        linewidth=line_width["vanilla"],
        marker=markers["vanilla"],
        markersize=marker_sizes["vanilla"])
ax.plot(chunked["qps"], 
        chunked["mean_ttft_ms"] / 1000, 
        label=method2name["chunked"], 
        color=colors["chunked"],
        linewidth=line_width["chunked"],
        marker=markers["chunked"],
        markersize=marker_sizes["chunked"])
ax.plot(pp["qps"], 
        pp["mean_ttft_ms"] / 1000, 
        label=method2name["pp"], 
        color=colors["pp"],
        linewidth=line_width["pp"],
        marker=markers["pp"],
        markersize=marker_sizes["pp"])
ax.plot(tp["qps"], 
        tp["mean_ttft_ms"] / 1000, 
        label=method2name["tp"], 
        color=colors["tp"],
        linewidth=line_width["tp"],
        marker=markers["tp"],
        markersize=marker_sizes["tp"])


# ax.legend(loc="upper right", framealpha=0.7)
ax.grid()
ax.set_xlabel("Query per second")
ax.set_ylabel("Avg latency (s)")
ax.set_ylim(bottom=0)
ax.set_ylim(top=85)
ax.set_xlim(left=0)
bbox_props = dict(boxstyle="rarrow", fc=(1,1,1), ec="grey", lw=2)
ax.text(0.05, 0.95, "Better", ha="left", 
va="top", rotation=-45, bbox=bbox_props, c='grey',
transform=ax.transAxes
)

fig.tight_layout()
fig.savefig("../eval_figures/l4_end2end_workload1_mean_ttft.pdf", bbox_inches="tight")
