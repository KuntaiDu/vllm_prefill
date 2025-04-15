
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

ax.bar(
    [method2name["prefill"], '\n'.join(method2name["pp"].split(" ")), '\n'.join(method2name["tp"].split(" "))],
    [prefill[prefill.qps==float("inf")].request_throughput.iloc[0], pp[pp.qps==float("inf")].request_throughput.iloc[0], tp[tp.qps==float("inf")].request_throughput.iloc[0]],
    color=[colors["prefill"], colors["pp"], colors["tp"]]
)

# ax.legend(loc="lower right", framealpha=0.7)
ax.grid(axis='y')
# ax.set_xlabel("Query per second")
ax.set_ylabel("Request Tput (req/s)")
ax.set_ylim(bottom=0)
# ax.set_ylim(top=140))
ax.set_xlim(left=-0.9)

bbox_props = dict(boxstyle="rarrow", fc=(1,1,1), ec="grey", lw=2)
ax.text(0.05, 0.9, "Better", ha="left", 
va="top", rotation=90, bbox=bbox_props, c='grey',
transform=ax.transAxes
)

fig.tight_layout()
fig.savefig("figures/h100_end2end_workload2_NVLink_tput.pdf", bbox_inches="tight")
