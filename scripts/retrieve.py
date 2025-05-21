# Testing file for train.py

import argparse
import json
import logging
from typing import Dict, Any, List, Optional, Literal, Tuple
import os
import sys
from dataclasses import dataclass, field

from sentence_transformers import (
    SentenceTransformer,
)
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import Corpus, read_states, base_dir
from data import RetrievalDataset, load_data, PremiseRetrievalDataCollator
from evaluator import PremiseRetrievalEvaluator


logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--mathlib_only", action="store_true")
parser.add_argument("--model_name", type=str, default=None)
parser.add_argument("--model_path", type=str, default=None)
args = parser.parse_args()

dataset_train, dataset_valid, dataset_test = load_data(
    input("data dir (default /data/user_data/thomaszh/Mathlib): ") or "/data/user_data/thomaszh/Mathlib",
    mathlib_only=args.mathlib_only,
    num_negatives_per_state=3  # not used
)

mini_batch_size = 32
evaluator_kwargs: Dict[str, Any] = dict(
    batch_size=mini_batch_size,
    corpus_chunk_size=50000,
    show_progress_bar=True,
    mrr_at_k=[256],
    ndcg_at_k=[],
    accuracy_at_k=[],
    precision_recall_at_k=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    map_at_k=[],
    full_recall_at_k=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
)

if args.model_name is not None:
    model_path = args.model_path or f"/data/user_data/thomaszh/models/{args.model_name}/final"
    model = SentenceTransformer(model_path)
    with torch.inference_mode():
        corpus_premises = dataset_test.corpus.premises
        corpus_embeddings = model.encode(
            [premise.to_string() for premise in corpus_premises],
            show_progress_bar=True,
            batch_size=mini_batch_size,
            convert_to_tensor=True,
        )

        for (name, dataset) in [("valid", dataset_valid), ("test", dataset_test)]:
            print("=" * 100)
            print(f"{name} evaluation")

            # Evaluate on the test set
            evaluator = PremiseRetrievalEvaluator(
                dataset=dataset,
                name=f"{name}-{args.model_name}",
                retrieved_premises_save_dir="retrieved_premises",
                **evaluator_kwargs,
            )
            results = evaluator(model, corpus_embeddings=corpus_embeddings)

            # Evaluate on next tactics only
            # NB: in the current setup, this should be the same as the evaluation above (see the evaluation set generation,
            # which only includes the initial states) so it is not included here.
            # dataset.generate_relevant_premises_and_pairs(mode="next_tactic_hammer_recommendation")
            # evaluator_next_tactic = PremiseRetrievalEvaluator(
            #     dataset=dataset,
            #     name=f"{name}-next-tactic",
            #     **evaluator_kwargs,
            # )
            # results_next_tactic = evaluator_next_tactic(model, corpus_embeddings=corpus_embeddings)

    # (Optional) Push it to the Hugging Face Hub
    # model.push_to_hub(model_name_or_path)
