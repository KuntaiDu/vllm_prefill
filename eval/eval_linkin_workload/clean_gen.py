import dataclasses
import random
import string
import json
from vllm.utils import FlexibleArgumentParser

def generate_dataset(args):
    random.seed(args.shuffle_seed)

    doc_alphas = [''.join(random.choices(string.ascii_lowercase, k=2)) for _ in range(args.num_documents)]
    user_alphas = [''.join(random.choices(string.ascii_lowercase, k=2)) for _ in range(args.num_users)]
    documents = [f"{i}{alpha}" for i, alpha in enumerate(doc_alphas)]

    records = []
    for user_idx in range(args.num_users):
        user_alpha = user_alphas[user_idx]
        user_name = f"user{user_idx}"
        user_history = ' '.join([f"{user_idx}{user_alpha}"] * args.user_history_length)

        for doc_chunk in documents:
            doc_text = ' '.join([doc_chunk] * 150)
            user_text = f"{user_name}:\n{user_history} document:\n{doc_text}"
            record = {
                "id": user_name,
                "conversations": [
                    {"from": "human", "value": user_text},
                    {"from": "gpt", "value": ""}
                ]
            }
            records.append(record)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Dataset saved to {args.output_json}")

if __name__ == "__main__":
    parser = FlexibleArgumentParser(description='Generate Long Document QA dataset.')
    parser.add_argument('--user-history-length', type=int, default=9000)
    parser.add_argument('--num-users', type=int, default=6)
    parser.add_argument('--document-length', type=int, default=10000)
    parser.add_argument('--num-documents', type=int, default=8)
    parser.add_argument('--output-len', type=int, default=1)
    parser.add_argument('--shuffle-seed', type=int, default=0)
    parser.add_argument('--prefill-only', action='store_true')
    parser.add_argument('--output-json', type=str, default='test_data_modify.json')
    args = parser.parse_args()
    generate_dataset(args)