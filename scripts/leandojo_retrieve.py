# (Statically) let leandojo reprover retrieve premises

import json
import pickle
import os
import sys

import numpy as np
from transformers import AutoTokenizer, T5EncoderModel
import torch
import tqdm
from huggingface_hub import HfApi, login

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import load_data

sys.path.insert(0, "/data/user_data/thomaszh/ReProver")

import lean_dojo, common  # unused imports to test that correct env is loaded

# We use "naive" data because leandojo uses it
data_dir = "/data/user_data/thomaszh/ntp-toolkit-naive/Examples/Mathlib"
mathlib_only = False
corpus_path = "/data/user_data/thomaszh/leandojo_benchmark_4/corpus.jsonl"
corpus_embeddings_pickle_path = "/data/user_data/thomaszh/ReProver/corpus-hammer.pickle"
model_name = "leandojo-lean4-retriever-byt5-small-hammer"
model_path = f"/data/user_data/thomaszh/ReProver/{model_name}"  # huggingface format
model_push_repo = f"{model_name}"
embeddings_path = "embeddings/leandojo_hammer_embeddings.npy"
dictionary_path = "embeddings/leandojo_hammer_dictionary.json"


dataset_train, dataset_valid, dataset_test = load_data(
    data_dir,
    mathlib_only=mathlib_only,
    num_negatives_per_state=0
)


### Unpickle premises
# `corpus.pickle` is produced by `retrieval/index.py` in [ReProver](https://github.com/lean-dojo/ReProver).
with open(corpus_embeddings_pickle_path, "rb") as f:
    indexed_corpus = pickle.load(f)

embeddings_tensor = indexed_corpus.embeddings
embeddings_array = embeddings_tensor.numpy()
embeddings_array_64 = embeddings_array.astype(np.float64)

np.save(embeddings_path, embeddings_array_64)
print(f"Embeddings saved to {embeddings_path}")

all_premises = indexed_corpus.corpus.all_premises

premise_dict = {
    index: {"full_name": premise.full_name, "path": premise.path, "code": premise.code}
    for index, premise in enumerate(all_premises)
}

with open(dictionary_path, "wt") as f:
    json.dump(premise_dict, f, indent=4)
print(f"Dictionary saved to {dictionary_path}")


## Retrieve for valid and train
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5EncoderModel.from_pretrained(model_path)
model.eval()
torch.inference_mode()

model.push_to_hub(model_push_repo)

@torch.no_grad()
def encode(s: str) -> torch.Tensor:
    """Encode texts into feature vectors."""
    s = [s]
    should_squeeze = True
    tokenized_s = tokenizer(s, return_tensors="pt", padding=True)
    hidden_state = model(tokenized_s.input_ids).last_hidden_state
    lens = tokenized_s.attention_mask.sum(dim=1)
    features = (hidden_state * tokenized_s.attention_mask.unsqueeze(2)).sum(
        dim=1
    ) / lens.unsqueeze(1)
    if should_squeeze:
        features = features.squeeze()
    return features

k = 1024
for name, dataset in [("valid", dataset_valid), ("test", dataset_test)]:
    retrieved_premises_path = f"retrieved_premises/dot_{name}-{model_name}.json"
    # `dataset.states` are the "root" states of the valid/test declarations
    states_serialized = []
    for state in tqdm.tqdm(dataset.states, desc=f"Retrieving for {name}"):
        state_embedding = encode(state.to_string())
        probs = torch.matmul(embeddings_tensor, state_embedding)
        assert probs.shape == (len(premise_dict),)

        accessible_premises = dataset.get_accessible_premises(state)
        for i in premise_dict:
            premise_name = premise_dict[i]["full_name"]
            if premise_name not in dataset.corpus.name2premise:
                probs[i] = -torch.inf
            else:
                premise = dataset.corpus.name2premise[premise_name]
                if premise not in accessible_premises:
                    probs[i] = -torch.inf

        scores, indices = torch.topk(probs, k)
        state_serialized = {
            "state_name": state.name, "decl_name": state.decl_name, "module": state.module, "idx_in_module": state.idx_in_module,
            "premises": [
                {"corpus_id": premise_dict[int(i)]["full_name"], "score": score.item()}
                for score, i in zip(scores, indices)
            ]
        }
        states_serialized.append(state_serialized)
    with open(retrieved_premises_path, "w") as f:
        json.dump(states_serialized, f)
