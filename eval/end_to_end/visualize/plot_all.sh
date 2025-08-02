mkdir -p ../eval_figures

echo "Plotting WL1 L4 mean ttft..."
python l4_end2end_workload1_mean_ttft.py

echo "Plotting WL1 L4 p99 ttft..."
python l4_end2end_workload1_p99_ttft.py

echo "Plotting WL1 L4 tput..."
python l4_end2end_workload1_tput.py

echo "Plotting WL2 L4 mean ttft..."
python l4_end2end_workload2_mean_ttft.py

echo "Plotting WL2 L4 p99 ttft..."
python l4_end2end_workload2_p99_ttft.py

echo "Plotting WL2 L4 tput..."
python l4_end2end_workload2_tput.py
