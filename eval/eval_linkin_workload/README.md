就写中文了哈

主要修改都在`/root/Eamin/vllm_prefill/eval/eval_linkin_workload`

1.生成数据集
先cd到`/root/Eamin/vllm_prefill/eval/eval_linkin_workload`

原始文件

`benchmark_recommendation_modify.py`

修改了生成数据的逻辑

`bm_prom.py`

摘出来生成数据，其他都去掉

`clean_gen.py`

但是这个只能生成一个numusers的数据，dirty fix了一下循环调用脚本

`generate_json.sh`

跑的话
```
chmod +x  generate_json.sh
bash generate_json.sh
```


修改num users 范围
```
#!/bin/bash

# save as: generate_dataset.sh
mkdir -p link_json

for num_users in $(seq 2 100); do   《----这里2到100代表n users
  python clean_gen.py \
    --prefill-only \
    --shuffle-seed 10 \
    --output-len 10 \
    --num-documents 50 \               《----这里是documents
    --num-users $num_users \           《===不用改
    --user-history-length 20000 \         《----这里是2w个前缀
    --output-json link_json/test_data_${num_users}.json 《会在这个文件夹里生成json数据集 比如test_data_3.json num=3 等等
done
```
修改150长度后缀在generate_json.sh
这一行
```
 doc_text = " ".join([doc_chunk] * 150)
```








2.跑数据

cd 到vllm_prefill/benchmarks

benchmark_serving_linkin.py这个

修改的点是数据集有filter 会去掉对话轮数少 我把这个去掉了


跑的话

先启vllm serve

再
```
python benchmark_serving_linkin.py \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path /root/Eamin/vllm_prefill/eval/eval_linkin_workload/link_json/test_data_2.json \
    --request-rate 5 \
    --num-prompts 100\
    --save-result \
    --result-dir /root/Eamin/vllm_prefill/eval/eval_linkin_workload/link_result_json\
    --result-filename my_benchmark.json
```



扫描频率   `/root/Eamin/vllm_prefill/eval/eval_linkin_workloa`
直接跑`qps.sh`

```
# 手动指定你要用的数据编号
data_num=30

# Define the directory containing the JSON datasets
DATA_DIR="/root/Eamin/vllm_prefill/eval/eval_linkin_workload/link_json"
# 直接拼接 json_file
json_file="$DATA_DIR/test_data_${data_num}.json"

# 手动指定结果目录
RESULT_DIR="/root/Eamin/vllm_prefill/eval/eval_linkin_workload/qps/${data_num}"
mkdir -p "$RESULT_DIR"

# 固定 num_prompts
num_prompts=$((data_num * 50))

# QPS 从0.1 到 2.0 循环
for qps in $(seq 0.1 0.1 2.0 ); do
```


修改一下上边文件名啥的

```
chmod +x
bash
```
跑一下


数据存在
```
eval/eval_linkin_workload/qps/2
eval/eval_linkin_workload/qps/3
```
最后一个数字是用户数量

里面的文件夹类似
```
result_qps_1_data_2.json
```
画图的话用qps文件夹里的`draw.py`
