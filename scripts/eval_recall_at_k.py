# Evaluate recall@k numbers

import json
import pickle
import os
import sys

import numpy as np
import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# We use our data because we are doing eval
data_dir = "/data/user_data/thomaszh/Mathlib"
mathlib_only = False
model_names = [
    # "leandojo-lean4-retriever-byt5-small-hammer",
    # for sanity check, these should equal the wandb-logged value during training
    # "all-MiniLM-L6-v2-lr2e-4-bs256-nneg3-ml-ne5",
    "all-MiniLM-L12-v2-lr2e-4-bs256-nneg3-ml-ne5",
    # "all-MiniLM-L12-v2-lr2e-4-bs256-nneg3-ml-ne5-nameless",
    # "all-MiniLM-L12-v2-lr2e-4-bs256-nneg3-ne5",
    # "all-MiniLM-L12-v2-lr2e-4-bs256-nneg3-ml-ne5-naive",
    # "all-distilroberta-v1-lr2e-4-bs256-nneg3-ml",
    # "all-distilroberta-v1-lr2e-4-bs256-nneg3-ml-ne2",
    "all-distilroberta-v1-lr2e-4-bs256-nneg3-ml-ne5",
    # "all-mpnet-base-v2-lr1e-4-bs256-nneg3-ml-ne5",
    # "all-MiniLM-L12-v2-lr2e-4-bs256-nneg0-ml-ne5",
    # "all-MiniLM-L12-v2-lr2e-4-bs256-nneg0-ne5",
    # "knn",
    # "rf",
    # "mepo-p0.5-c0.9",
    # "mepo-p0.5-c1.2",
    # "mepo-p0.5-c2.4",
    # "mepo-p0.5-c3.6",
    # "mepo-p0.6-c0.9",
    # "mepo-p0.6-c1.2",
    # "mepo-p0.6-c2.4",
    # "mepo-p0.6-c3.6",
    # "mepo-p0.7-c0.9",
    # "mepo-p0.7-c1.2",
    # "mepo-p0.7-c2.4",
    # "mepo-p0.7-c3.6",
    # "mepo-p0.8-c0.9",
    # "mepo-p0.8-c1.2",
    # "mepo-p0.8-c2.4",
    # "mepo-p0.8-c3.6",
    # "mepo-p0.9-c0.9",
    # "mepo-p0.9-c1.2",
    # "mepo-p0.9-c2.4",
    # "mepo-p0.9-c3.6",
]
eval_decls_tag = "apr25"

# dataset_train, dataset_valid, dataset_test = load_data(
#     data_dir,
#     mathlib_only=mathlib_only,
#     num_negatives_per_state=0
# )

for model_name in model_names:
    print(f"Evaluating {model_name}")

    for name in ["valid", "test"]:
        print(f"=== {name} ===")
        for k in [16, 32]:
            decls_path = f"retrieved_premises/{name}_decls_{eval_decls_tag}.json"
            if model_name not in ["rf", "knn"] and not model_name.startswith("mepo"):
                retrieved_premises_path = f"retrieved_premises/{name}-{model_name}.json"
            elif model_name.startswith("mepo"):
                retrieved_premises_path = f"results_mar30-test/{model_name.split('-')[0]}/{model_name}.json"
            else:
                retrieved_premises_path = f"results_mar11-test/{model_name.split('-')[0]}/{model_name}.json"

            with open(decls_path) as f:
                decls = json.load(f)
                name2relevant = {decl["decl_name"]: decl["gt_premises"] for decl in decls}
            with open(retrieved_premises_path) as f:
                entries = json.load(f)

            recalls = []
            precisions = []
            f1s = []
            full_recalls = []
            not_found = []
            for entry in entries:
                if model_name not in ["rf", "knn"] and not model_name.startswith("mepo"):
                    assert len(entry["premises"]) == 1024
                decl_name = entry["decl_name"]
                if decl_name not in name2relevant:
                    print(f"Warning: {decl_name} not found in {decls_path}")
                    not_found.append(decl_name)
                    continue

                relevant = set(name2relevant[decl_name])
                retrieved = {e["corpus_id"] for e in sorted(entry["premises"], key=lambda e: e["score"], reverse=True)[:k]}
                # print(decl_name, retrieved, relevant)
                # assert len(retrieved) == k  # apparently this fails in exceptional circumstances, due to errors? in leandojo's extraction

                tp = relevant & retrieved
                precision = len(tp) / k
                precisions.append(precision)
                full_recalls.append(relevant <= retrieved)

                if len(relevant) > 0:
                    recall = len(tp) / len(relevant)
                else:
                    recall = 1.
                recalls.append(recall)

                f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
                f1s.append(f1)

            print(f"{len(not_found)} / {len(entries)} not found")
            print(f"Recall@{k} = {np.mean(recalls) * 100.} %")
            print(f"Precision@{k} = {np.mean(precisions) * 100.} %")
            print(f"F1@{k} = {np.mean(f1s) * 100.} %")
            print(f"FullRecall@{k} = {np.mean(full_recalls) * 100.} %")
            print()
