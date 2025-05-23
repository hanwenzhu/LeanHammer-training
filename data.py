# Defines `RetrievalDataset` and `PremiseRetrievalDataCollator` for use during training

from dataclasses import dataclass, field
import logging
from typing import Dict, List, Literal, Optional, Sequence, Set, Tuple, Any, Callable

from sentence_transformers.data_collator import SentenceTransformerDataCollator
import datasets  # for baseline
# from torch.utils.data import Dataset
import torch
import torch.utils.data
import numpy as np
import tqdm

from models import StateWithTactic, Corpus, Premise, read_states, PremiseSet, BaseInfo

logger = logging.getLogger(__name__)


class RetrievalDataset:  # (torch.utils.data.Dataset):
    def __init__(
        self, corpus: Corpus, states: List[StateWithTactic],
        num_negatives_per_state: int, train: bool
    ):
        self.corpus = corpus
        """The `Corpus` object storing premises to retrieve from."""
        self.states = states
        """All states in the dataset."""
        self.revision = corpus.revision
        """Revision (Lean version) of the data extracted."""

        self.train = train
        """Whether the dataset is for training."""

        self.name2state = {state.name: state for state in self.states}

        self.num_negatives_per_state = num_negatives_per_state
        """Number of negative premises to sample for each state"""

        self.generator = torch.Generator(device="cpu")
        """Controls negative premise selection (which is currently random)"""

        # These are set in `generate_relevant_premises_and_pairs``
        self.relevant_premises: Dict[str, Set[str]] = {}
        """Mapping state names to set of names of used premises"""
        self.pairs: List[Tuple[StateWithTactic, Premise]] = []
        """Pairs of (state, premise) where premise is relevant to state"""
        self.generate_relevant_premises_and_pairs()

    def generate_relevant_premises_and_pairs(
        self,
        mode: Literal["hammer_recommendation", "next_tactic_hammer_recommendation"] = "hammer_recommendation"
    ):
        # for state in tqdm.tqdm(self.states, desc="Preparing states"):
        self.pairs = []
        self.relevant_premises = {}
        for state in self.states:
            self.relevant_premises[state.name] = set()
            match mode:
                case "hammer_recommendation":
                    state_relevant_premises = state.hammer_recommendation
                case "next_tactic_hammer_recommendation":
                    state_relevant_premises = state.next_tactic_hammer_recommendation
            for premise_name in state_relevant_premises:
                if premise_name in self.corpus.name2premise:
                    premise = self.corpus.name2premise[premise_name]
                    self.relevant_premises[state.name].add(premise_name)
                    self.pairs.append((state, premise))

    # def __len__(self):
    #     return len(self.pairs)

    # def __getitem__(self, index) -> Dict[str, Any]:
    #     state, premise = self.pairs[index]
    #     negative_premises = self.sample_negatives(state)

    #     example = {}
    #     example["state"] = state.to_string()
    #     example["premise"] = premise.to_string()
    #     for i, negative_premise in enumerate(negative_premises):
    #         example[f"negative_premise{i}"] = negative_premise.to_string()
    #     return example

    def to_huggingface_dataset(self) -> datasets.Dataset:
        """Convert RetrievalDataset to a datasets.Dataset with (state, premise) pairs
        Note: I would like to use torch.utils.data.Dataset, but this does not (?) work with sentence-transformers.
        """
        examples = [
            # We relegate all logic to the collator
            {
                "state_name": state.name,
                "premise_name": premise.name,
                # The actual negative premise sampling is done by `sample_negatives`,
                # but is deferred to data collator because doing it here makes the negative premises
                # the same in different accesses to the same index.
                # FIXME: adding this hangs the dataset (!?)
                # "should_sample_negatives": self.train,
            }
            for state, premise in self.pairs
        ]
        features = datasets.Features({
            "state_name": datasets.Value(dtype="string"),
            "premise_name": datasets.Value(dtype="string"),
            # "should_sample_negatives": datasets.Value(dtype="bool"),
        })
        dataset: datasets.Dataset = datasets.Dataset.from_list(examples, features=features)  # type: ignore
        return dataset

    def get_accessible_premises(self, state: StateWithTactic) -> PremiseSet:
        return self.corpus.get_accessible_premises(state)

    def get_negative_premises(self, state: StateWithTactic) -> PremiseSet:
        return self.corpus.get_negative_premises(state)

    def get_accessible_negative_premises(self, state: StateWithTactic) -> PremiseSet:
        return self.corpus.get_accessible_negative_premises(state)

    def sample_negatives(self, state: StateWithTactic) -> List[Premise]:
        """Sample a list of negative premises of size self.num_negatives_per_state for the given state"""
        # TODO, implement hard negative and/or stratified sampling
        if self.num_negatives_per_state == 0:
            return []  # avoid calculating negative premise set
        negatives = self.get_accessible_negative_premises(state)
        return negatives.sample(self.num_negatives_per_state, generator=self.generator)

    # This is not used and probably not needed
    def set_epoch(self, epoch: int):
        self.generator.manual_seed(self.generator.initial_seed() + epoch)


def load_data(
    data_dir: str,
    mathlib_only: bool = False,
    filter: bool = True,
    nameless: bool = False,
    num_negatives_per_state: int = 0,
    # This is probably not ideal split size?
    valid_num_decl: int = 500,
    test_num_decl: int = 500,
    **dataset_kwargs
) -> Tuple[RetrievalDataset, RetrievalDataset, RetrievalDataset]:
    """Get the train, valid, test datasets"""

    logger.info(f"Loading data from {data_dir} (mathlib_only={mathlib_only})")

    states = read_states(data_dir, mathlib_only=mathlib_only)
    corpus = Corpus.from_ntp_toolkit(data_dir, filter=filter, mathlib_only=mathlib_only, nameless=nameless)

    # Split states to train, valid, test
    all_decl_names = sorted({state.decl_name for state in states})
    np.random.seed(0)  # TODO
    np.random.shuffle(all_decl_names)
    def is_eligible_for_valid_test(name):
        if not name:
            return False
        if name not in corpus.name2premise:
            return False
        premise = corpus.name2premise[name]
        return premise.can_benchmark
    eligible_names_for_valid_test = [name for name in all_decl_names if is_eligible_for_valid_test(name)]
    valid_decl_names = set(eligible_names_for_valid_test[test_num_decl : test_num_decl + valid_num_decl])
    test_decl_names = set(eligible_names_for_valid_test[:test_num_decl])
    train_decl_names = set(all_decl_names) - valid_decl_names - test_decl_names

    logger.info(f"Total {len(states)} states from {len(all_decl_names)} declarations")

    train_states = [state for state in states if state.decl_name in train_decl_names]
    # For valid and test states, only include the "root" state
    valid_decl_name2states: Dict[str, StateWithTactic] = {}
    test_decl_name2states: Dict[str, StateWithTactic] = {}
    for state in states:
        if state.decl_name in valid_decl_names and (
                state.decl_name not in valid_decl_name2states or
                state.idx_in_module < valid_decl_name2states[state.decl_name].idx_in_module):
            valid_decl_name2states[state.decl_name] = state
        if state.decl_name in test_decl_names and (
                state.decl_name not in test_decl_name2states or
                state.idx_in_module < test_decl_name2states[state.decl_name].idx_in_module):
            test_decl_name2states[state.decl_name] = state
    valid_states = list(valid_decl_name2states.values())
    test_states = list(test_decl_name2states.values())

    np.random.shuffle(train_states)  # type: ignore
    # np.random.shuffle(valid_states)  # type: ignore
    # np.random.shuffle(test_states)  # type: ignore

    dataset_train = RetrievalDataset(corpus=corpus, states=train_states, num_negatives_per_state=num_negatives_per_state, train=True, **dataset_kwargs)
    dataset_valid = RetrievalDataset(corpus=corpus, states=valid_states, num_negatives_per_state=0, train=False, **dataset_kwargs)
    dataset_test = RetrievalDataset(corpus=corpus, states=test_states, num_negatives_per_state=0, train=False, **dataset_kwargs)

    logger.info("Training data summary:")
    logger.info(f"Total # premises: {len(corpus.premises)} (filtered from {len(corpus.unfiltered_premises)})")
    logger.info(f"Avgerage # ground truth premises: {np.mean([len(ps) for ps in dataset_train.relevant_premises.values()])}")
    logger.info(f"Total # training pairs: {len(dataset_train.pairs)}")
    logger.info(f"Total # training pairs (+ negative pairs): {len(dataset_train.pairs) * (1 + num_negatives_per_state)}")

    return dataset_train, dataset_valid, dataset_test


@dataclass(kw_only=True)
class PremiseRetrievalDataCollator(SentenceTransformerDataCollator):
    """Collating function for premise retrieval, that also recognizes negatives.
    Implemented as a minimal wrapper around SentenceTransformerDataCollator
    """

    # A better design choice is to have a single dataset field, and
    # use a train collator for train dataset, eval collator for eval dataset.
    # But this is incompatible with transformers Trainer logic, and we could circumvent by subclassing
    # SentenceTransformerTrainer and rewriting get_eval_collator etc., but chose not to for simplicity
    train_dataset: RetrievalDataset
    all_datasets: List[RetrievalDataset]
    retrieval_mask_mode: Literal["none", "no_positive", "only_negative"] = "no_positive"
    def __post_init__(self):
        self.corpus = self.train_dataset.corpus
        self.name2state: Dict[str, StateWithTactic] = self.train_dataset.name2state
        self.relevant_premises: Dict[str, Set[str]] = self.train_dataset.relevant_premises
        for dataset in self.all_datasets:
            self.name2state.update(dataset.name2state)
            self.relevant_premises.update(dataset.relevant_premises)

    def is_false_negative(self, state: StateWithTactic, premise: Premise) -> bool:
        """If the (state, premise) pair is labelled negative, determine if it is false negative."""
        if self.retrieval_mask_mode == "no_positive":
            return premise.name in self.relevant_premises[state.name] or premise.name == state.decl_name
        elif self.retrieval_mask_mode == "only_negative":
            return premise not in self.train_dataset.get_negative_premises(state)
        else:
            return False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        column_names = list(features[0].keys())

        # We should always be able to return a loss, label or not:
        batch = {}

        if "dataset_name" in column_names:
            column_names.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        if tuple(column_names) not in self._warned_columns:
            self.maybe_warn_about_column_order(column_names)

        # Extract the label column if it exists
        for label_column in self.valid_label_columns:
            if label_column in column_names:
                batch["label"] = torch.tensor([row[label_column] for row in features])
                column_names.remove(label_column)
                break

        # Extract the feature columns
        infos: Dict[str, List[BaseInfo]] = {}
        infos["state"] = [self.name2state[row["state_name"]] for row in features]
        infos["premise"] = [self.corpus.name2premise[row["premise_name"]] for row in features]
        premise_column_names = ["premise"]  # all column names corresponding to (positive or negative) premises

        # If training, also sample negatives
        # if features[0]["should_sample_negatives"]:
        if features[0]["state_name"] in self.train_dataset.name2state:
            negatives = [self.train_dataset.sample_negatives(state) for state in infos["state"]]  # type: ignore
            for i in range(len(negatives[0])):
                infos[f"negative_premise_{i}"] = [negative[i] for negative in negatives]
                premise_column_names.append(f"negative_premise_{i}")

        for column_name in ["state"] + premise_column_names:
            tokenized = self.tokenize_fn([info.to_string() for info in infos[column_name]])
            for key, value in tokenized.items():
                batch[f"{column_name}_{key}"] = value

        # I don't know when I should do this
        # batch.update(infos)

        batch_size = len(features)
        num_premise_columns = len(premise_column_names)  # including positive premises

        # (bsz, (1 + nneg) * bsz)
        retrieval_mask = torch.ones(
            (batch_size, num_premise_columns, batch_size),
            dtype=torch.bool
        )
        for i, state in enumerate(infos["state"]):  # type: ignore
            state: StateWithTactic
            for j, premise_column_name in enumerate(premise_column_names):
                for k, premise in enumerate(infos[premise_column_name]):  # type: ignore
                    premise: Premise
                    # If this (state, premise) pair is labelled negative
                    if (j, k) != (0, i):
                        # Identify if this pair is false negative; if so don't include it in loss
                        if self.is_false_negative(state, premise):
                            retrieval_mask[i, j, k] = 0
        batch["state_retrieval_mask"] = retrieval_mask.view(batch_size, -1)

        # Note: the `batch` now goes into `SentenceTransformerTrainer.collect_features`
        # And then into `(Masked)(Cached)MultipleNegativesRankingLoss.forward`
        return batch
