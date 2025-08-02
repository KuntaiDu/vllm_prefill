import matplotlib.pyplot as plt
import json
from pathlib import Path
import pandas as pd

from visualize_utils import *



DIR = '../results-0801-L4-WL-2'


prefill = pd.DataFrame.from_dict(load_results(f"{DIR}/prefill_csjf"))
pp = pd.DataFrame.from_dict(load_results(f"{DIR}/pp"))
tp = pd.DataFrame.from_dict(load_results(f"{DIR}/tp"))

prefill.sort_values(by="qps", inplace=True)
pp.sort_values(by="qps", inplace=True)
tp.sort_values(by="qps", inplace=True)

fig, ax = plt.subplots(figsize=(7.5, 5))


ax.plot(prefill["qps"], 
        prefill["request_throughput"], 
        label=method2name["prefill"], 
        color=colors["prefill"],
        linewidth=line_width["prefill"],
        marker=markers["prefill"],
        markersize=marker_sizes["prefill"])
ax.plot(pp["qps"], 
        pp["request_throughput"], 
        label=method2name["pp"], 
        color=colors["pp"],
        linewidth=line_width["pp"],
        marker=markers["pp"],
        markersize=marker_sizes["pp"])
ax.plot(tp["qps"], 
        tp["request_throughput"], 
        label=method2name["tp"], 
        color=colors["tp"],
        linewidth=line_width["tp"],
        marker=markers["tp"],
        markersize=marker_sizes["tp"])
ax.legend(loc="lower right", framealpha=0.7)
ax.grid()
ax.set_xlabel("Query per second")
ax.set_ylabel("Request throughput (req/s)")
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)
# ax.set_ylim(top=105)

fig.tight_layout()
fig.savefig("../eval_figures/l4_end2end_workload2_tput.pdf", bbox_inches="tight")
