export RESULTS_PATH='results-0413-re'

PIDS=()

mkdir -p logs

cleanup() {
    echo "Cleaning up in run-flat..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing PID $pid"
            kill -USR1 $pid
        fi
    done
    for pid in "${PIDS[@]}"; do
        # Waiting for the process to finish
        echo "Waiting $pid"
        wait "$pid"
    done
    # exit 130
}

cleanup_and_exit() {
    cleanup
    exit 130
}

trap cleanup_and_exit INT

# Utility function to wait for a server to start
wait_for_server() {
  local port=$1
  local timeout_seconds=120
  local start_time=$(date +%s)

  echo "Waiting for server on port $port..."

  while true; do
    if curl -s "http://localhost:${port}/name" > /dev/null; then
      return 0
    fi

    local now=$(date +%s)
    if (( now - start_time >= timeout_seconds )); then
      echo "Timeout waiting for server, cleanup"
      cleanup

      return 1
    fi

    sleep 5
  done
}

go() {
    local setting=$1
    local workload=$2
    local qps=$3

    # Clear pid list
    PIDS=()

    export WORKLOAD=$workload
    export qps=$qps
    echo "Running with setting: $setting, workload: $workload, qps: $qps"

    bash server.sh $setting >logs/${RESULTS_PATH}-server_${setting}_${workload}_${qps}.log 2>&1 &
    PIDS+=($!)
    
    # Wait for the server to start
    if ! wait_for_server 5000; then
        echo "Server failed to start."
        continue
    fi

    echo "Server started with PID ${PIDS[-1]}"
    
    # Run the client script
    bash client.sh | tee logs/${RESULTS_PATH}-client_${setting}_${workload}_${qps}.log

    # Done! Stop the server
    cleanup
}

# for setting in vanilla chunked prefill_csjf tp pp; do
#     for workload in 1 2; do
#         for qps in inf; do
#             go $setting $workload $qps
#         done
#     done
# done

export RESULTS_PATH='results-0413-re-1'

for workload in 1; do
    for setting in vanilla chunked prefill_csjf tp pp; do
        for qps in 1.94 3.88 7.76 15.52 31.04; do
            go $setting $workload $qps
        done
    done
done

export RESULTS_PATH='results-0413-re-2'

for workload in 2; do
    for setting in prefill_csjf tp pp; do
        for qps in 0.0225 0.045 0.09 0.18 0.36; do
            go $setting $workload $qps
        done
    done
done
