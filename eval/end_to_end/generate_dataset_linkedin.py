
import dataclasses
import random
import string
import json
from vllm.utils import FlexibleArgumentParser

def generate_dataset(args):
    random.seed(args.shuffle_seed)

    records = []
    for user_idx in range(args.num_users):
        user_name = f"user{user_idx}"

        user_history_length =int(random.gauss(
            args.user_history_mean, 
            args.user_history_std
        ))
        user_history_length = max(user_history_length, args.user_history_min)
        user_history_length = min(user_history_length, args.user_history_max)
        user_history = ' '.join(["Hi"] * user_history_length)

        for doc_idx in range(args.num_documents):
            doc_name = f"document{doc_idx}"
            document_length = int(random.gauss(
                args.document_mean, 
                args.document_std
            ))
            document_length = max(document_length, args.document_min)
            document_length = min(document_length, args.document_max)
            doc_text = ' '.join(["Hi"] * document_length)
            user_text = f"{user_name}:\n{user_history}\n\n{doc_name}:\n{doc_text}"
            record = {
                "id": f"{user_idx}",
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
    parser.add_argument('--user-history-mean', type=float, default=20000)
    parser.add_argument('--user-history-std', type=float, default=3000)
    parser.add_argument('--user-history-min', type=int, default=10)
    parser.add_argument('--user-history-max', type=int, default=40000)
    parser.add_argument('--num-users', type=int, default=8)
    parser.add_argument('--num-documents', type=int, default=50)
    parser.add_argument('--document-mean', type=int, default=1500)
    parser.add_argument('--document-std', type=int, default=500)
    parser.add_argument('--document-min', type=int, default=100)
    parser.add_argument('--document-max', type=int, default=3000)
    parser.add_argument('--shuffle-seed', type=int, default=0)
    parser.add_argument('--output-json', type=str, default='test_data_modify.json')
    args = parser.parse_args()
    generate_dataset(args)