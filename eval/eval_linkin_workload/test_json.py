import json

with open('/root/Eamin/vllm_prefill/eval/eval_linkin_workload/test_data_modify.json') as f:
    data = json.load(f)

print(len(data))
print(data[0]['conversations'][0]['value'])