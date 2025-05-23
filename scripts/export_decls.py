import json
import sys
import os

import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import load_data

tag = ""
# # naive data
# tag = "_naive"
# tag = "_naive-blacklist"

if "naive" in tag:
    data_dir = "/data/user_data/thomaszh/ntp-toolkit-naive/Examples/Mathlib"
else:
    data_dir = "/data/user_data/thomaszh/mathlib"

mathlib_only = False
dataset_train, dataset_valid, dataset_test = load_data(
    data_dir,
    mathlib_only=mathlib_only,
    filter=tag != "_naive",
    num_negatives_per_state=0
)

for name, dataset in [("valid", dataset_valid), ("test", dataset_test)]:
    filename = f"retrieved_premises/{name}_decls{tag}.json"
    # `dataset.states` are the "root" states of the valid/test declarations
    states = [
        {"decl_name": state.decl_name, "module": state.module, "idx_in_module": state.idx_in_module, "gt_premises": list(dataset.relevant_premises[state.name]), "gt_hints": state.simp_all_hints}
        for state in dataset.states
    ]
    with open(filename, "w") as f:
        json.dump(states, f)

    print(f"Output saved to {filename}")
