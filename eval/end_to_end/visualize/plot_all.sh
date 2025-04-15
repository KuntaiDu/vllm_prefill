

echo "Plotting WL1 H100 mean ttft..."
python h100_end2end_workload1_mean_ttft.py

echo "Plotting WL1 H100 p99 ttft..."
python h100_end2end_workload1_p99_ttft.py

echo "Plotting WL1 H100 tput..."
python h100_end2end_workload1_tput.py

echo "Plotting WL2 H100 mean ttft..."
python h100_end2end_workload2_mean_ttft.py

echo "Plotting WL2 H100 p99 ttft..."
python h100_end2end_workload2_p99_ttft.py

echo "Plotting WL2 H100 tput..."
python h100_end2end_workload2_tput.py

echo "Plotting WL1 H100 NVLink mean ttft..."
python h100_end2end_workload1_NVLink_mean_ttft.py

echo "Plotting WL1 H100 NVLink p99 ttft..."
python h100_end2end_workload1_NVLink_p99_ttft.py

echo "Plotting WL2 H100 NVLink mean ttft..."
python h100_end2end_workload2_NVLink_mean_ttft.py

echo "Plotting WL2 H100 NVLink p99 ttft..."
python h100_end2end_workload2_NVLink_p99_ttft.py

echo "Plotting WL2 H100 NVLink tput..."
python h100_end2end_workload2_NVLink_tput.py
