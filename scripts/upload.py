# Uploading the model, premise corpus, and pre-computed embeddings to Hugging Face

from datetime import datetime
import json
import os
import sys
import tarfile

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from huggingface_hub import HfApi, login

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import load_data

data_dir = "/data/user_data/thomaszh/mathlib"
mathlib_only = False
model_name = "all-distilroberta-v1-lr2e-4-bs256-nneg3-ml-ne5"
model_path = f"/data/user_data/thomaszh/models/{model_name}/final"
model_push_repo = f"{model_name}"
data_archive_path = "premises.tar.gz"
data_push_repo = "l3lab/lean-premises"
with open(os.path.join(data_dir, "revision")) as f:
    revision = f.read().strip()

os.makedirs("embeddings", exist_ok=True)
export_embeddings_path = f"embeddings/embeddings_{model_name}.npy"
embeddings_path_in_repo = f"embeddings/{model_name}.npy"


### Export decls
dataset_train, dataset_valid, dataset_test = load_data(
    data_dir,
    mathlib_only=mathlib_only,
    num_negatives_per_state=0
)


### Push model to hub
model = SentenceTransformer(model_path)
model.eval()
torch.inference_mode()

model.push_to_hub(model_push_repo, revision=revision, exist_ok=True)


### Export embeddings
mini_batch_size = 32
corpus_premises = dataset_test.corpus.premises
corpus_embeddings = model.encode(
    [premise.to_string() for premise in corpus_premises],
    show_progress_bar=True,
    batch_size=mini_batch_size,
    convert_to_tensor=True,
)
np.save(export_embeddings_path, corpus_embeddings.cpu().numpy())

api = HfApi()

api.create_branch(
    repo_id=data_push_repo, repo_type="dataset", branch=revision,
    exist_ok=True,
)

# Upload pre-computed embeddings
api.upload_file(
    repo_id=data_push_repo, repo_type="dataset", revision=revision,
    path_or_fileobj=export_embeddings_path,
    path_in_repo=embeddings_path_in_repo,
)

# Upload premises
with tarfile.open(data_archive_path, "w:gz") as tar:
    for item in ["revision", "Modules.jsonl", "HammerBlacklist.jsonl", "Declarations", "Imports"]:
        item_path = os.path.join(data_dir, item)
        tar.add(item_path, arcname=item)
api.upload_file(
    repo_id=data_push_repo, repo_type="dataset", revision=revision,
    path_or_fileobj=data_archive_path,
    path_in_repo="premises.tar.gz",
)

# Too many small files is not compatible with Hugging Face's rate limiting
# api.upload_folder(
#     repo_id=data_push_repo, repo_type="dataset", revision=revision,
#     folder_path=data_dir,
#     path_in_repo=".",
#     allow_patterns=[
#         "revision",
#         "Modules.jsonl",
#         "HammerBlacklist.jsonl",
#         "Declarations/*.jsonl",
#         "Imports/*.jsonl",
#     ]
# )
