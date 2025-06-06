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

# System-specific configs
data_dir = "/data/user_data/thomaszh/mathlib"
model_name = "all-distilroberta-v1-lr2e-4-bs256-nneg3-ml-ne2"
model_path = f"/data/user_data/thomaszh/models/{model_name}/final"

# Data & model upload configs
mathlib_only = False
model_push_repo = f"{model_name}"
data_archive_path = "premises.tar.gz"
data_archive_path_in_repo = "premises.tar.gz"
data_push_repo = "l3lab/lean-premises"

# Embeddings upload configs
os.makedirs("embeddings", exist_ok=True)
export_embeddings_path = f"embeddings/embeddings_{model_name}.npy"
embeddings_path_in_repo = f"embeddings/{model_name}.npy"

# Load model
print(f"Loading model from {model_path}")
model = SentenceTransformer(model_path)
model.eval()
torch.inference_mode()

# Data and model revision
with open(os.path.join(data_dir, "revision")) as f:
    data_revision = f.read().strip()
try:
    with open(os.path.join(model_path, "revision")) as f:
        model_revision = f.read().strip()
except FileNotFoundError as e:
    print(f"Warning: cannot infer model revision from saved model. Using data revision {data_revision} as model revision.")
    model_revision = data_revision


# Load data
dataset_train, dataset_valid, dataset_test = load_data(
    data_dir,
    mathlib_only=mathlib_only,
    num_negatives_per_state=0
)


### Push model to hub
print(f"Pushing model to {model_push_repo} @ {model_revision}")
model.push_to_hub(model_push_repo, revision=model_revision, exist_ok=True)


### Export embeddings
print(f"Embedding premises to {export_embeddings_path}")
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
    repo_id=data_push_repo, repo_type="dataset", branch=data_revision,
    exist_ok=True,
)

# Upload pre-computed embeddings
print(f"Uploading embeddings to {data_push_repo} @ {data_revision} / {embeddings_path_in_repo}")
api.upload_file(
    repo_id=data_push_repo, repo_type="dataset", revision=data_revision,
    path_or_fileobj=export_embeddings_path,
    path_in_repo=embeddings_path_in_repo,
)

# Upload premises
with tarfile.open(data_archive_path, "w:gz") as tar:
    for item in ["revision", "Modules.jsonl", "HammerBlacklist.jsonl", "Declarations", "Imports"]:
        item_path = os.path.join(data_dir, item)
        tar.add(item_path, arcname=item)
print(f"Uploading premises to {data_push_repo} @ {data_revision} / {data_archive_path_in_repo}")
api.upload_file(
    repo_id=data_push_repo, repo_type="dataset", revision=data_revision,
    path_or_fileobj=data_archive_path,
    path_in_repo=data_archive_path_in_repo,
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
