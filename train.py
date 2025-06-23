# Contrastive learning for premise retrieval

import argparse
import logging
from typing import Dict, Any, List, Optional, Literal, Tuple
import os
from dataclasses import dataclass, field

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.util import cos_sim, dot_score
from sentence_transformers.models import Transformer, Pooling, Normalize
from sentence_transformers.losses import MultipleNegativesRankingLoss, CachedMultipleNegativesRankingLoss, CosineSimilarityLoss
from sentence_transformers.training_args import BatchSamplers
import transformers
from transformers import AutoTokenizer, AutoModel, TrainerCallback
from transformers.integrations import WandbCallback
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from models import Corpus, read_states, base_dir
from data import RetrievalDataset, load_data, PremiseRetrievalDataCollator
from evaluator import PremiseRetrievalEvaluator
from loss import MaskedCachedMultipleNegativesRankingLoss


logging.basicConfig(level=logging.DEBUG)


@dataclass
class ModelArguments:
    model_name_or_path: str # field(default="all-distilroberta-v1") # "l3lab/ntp-mathlib-st-deepseek-coder-1.3b")
    is_sentence_transformer: bool = field(
        default=False, metadata={"help": "If true, the model is SentenceTransformer(model_name_or_path). Otherwise build a SentenceTransformer from scratch from AutoModel.from_pretrained(model_name_or_path), followed by a pooling layer."}
    )
    pooling_mode: Literal["cls", "lasttoken", "max", "mean", "mean_sqrt_len_tokens", "weightedmean"] = field(
        default="lasttoken", metadata={"help": "If not is_sentence_transformer, choice of pooling method for the pooling layer to build the embedding model"}
    )
    # state_prompt: str = field(default="", metadata={"help": "Prompt for encoding a state."})
    # premise_prompt: str = field(default="", metadata={"help": "Prompt for encoding a premise."})
    # This is put in model arguments because it only depends on how much can be fit in memory and does not affect training other than speed
    mini_batch_size: int = field(default=32, metadata={"help": "Mini-batch size in MultipleNegativesRankingLoss, when calculating embeddings. Does not affect final loss."})

@dataclass
class DataArguments:
    data_dir: str = field(metadata={"help": "Path to base directory of training data (extracted from ntp-toolkit)."})
    mathlib_only: bool = field(default=False, metadata={"help": "Use only Mathlib premises to train data"})
    filter: bool = field(default=True, metadata={"help": "Filter out by explicit blacklist and name/module/type-based blacklist"})
    nameless: bool = field(default=False, metadata={"help": "If true, the pretty-printed string of a premise will not contain the premise name."})
    retrieval_mask_mode: Literal["none", "no_positive", "only_negative"] = field(default="no_positive", metadata={"help": "Use a mask to eliminate false negatives in contrastive learning loss"})
    num_negatives_per_state: int = field(default=3, metadata={"help": "Number of negatives per training sample"})
    use_qlora: bool = field(default=False)

if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, SentenceTransformerTrainingArguments))  # type: ignore
    dataclasses: Tuple[ModelArguments, DataArguments, SentenceTransformerTrainingArguments] = parser.parse_args_into_dataclasses()  # type: ignore
    model_args, data_args, train_args = dataclasses

    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # using >1 data collator workers with prefetch
    # os.environ["KMP_AFFINITY"] = "disabled"  # (see "should_sample_negatives")

    if train_args.local_rank == 0:
        print("=" * 100)
        print(model_args)
        print(data_args)
        print(train_args)


    ## Prepare data (mathlib only)
    dataset_train, dataset_valid, dataset_test = load_data(
        data_args.data_dir,
        mathlib_only=data_args.mathlib_only,
        filter=data_args.filter,
        nameless=data_args.nameless,
        num_negatives_per_state=data_args.num_negatives_per_state
    )


    ## Prepare model
    if model_args.is_sentence_transformer:
        model_name_or_path = model_args.model_name_or_path
        modules = None
    else:
        # Use a decoder-only model to generate embeddings, by taking the last hidden state of the last token (</s>)
        # model_name_or_path = "l3lab/ntp-mathlib-st-deepseek-coder-1.3b"
        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if data_args.use_qlora:
            raise NotImplementedError("PEFT not implemented")
            from transformers import BitsAndBytesConfig
            from peft import LoraConfig, prepare_model_for_kbit_training  # type: ignore
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if train_args.bf16 else torch.float16,
            )
            peft_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=64,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules="all-linear",
            )
            model_kwargs["quantization_config"] = bnb_config
        transformer_model = Transformer(
            model_args.model_name_or_path,
            model_args=model_kwargs,
            tokenizer_args={"trust_remote_code": True, "add_eos_token": True},
            config_args={"trust_remote_code": True},
        )
        pooling_layer = Pooling(
            transformer_model.get_word_embedding_dimension(),
            pooling_mode=model_args.pooling_mode
        )
        normalize_layer = Normalize()
        model_name_or_path = None
        modules = [transformer_model, pooling_layer, normalize_layer]
    model = SentenceTransformer(
        model_name_or_path=model_name_or_path,
        modules=modules,
        model_card_data=SentenceTransformerModelCardData(
            # NB: this does not seem to have any effect
            train_datasets=[{"name": type(dataset_train).__name__, "revision": dataset_train.revision}]
        )
        # prompts={"state": model_args.state_prompt, "premise": model_args.premise_prompt}
    )
    # This is a hot fix to comply with HfTrainerDeepSpeedConfig
    # which needs to read model.config.hidden_size to set "auto" params
    model.config = model._first_module().auto_model.config  # type: ignore


    ## Training
    evaluator_kwargs: Dict[str, Any] = dict(
        batch_size=model_args.mini_batch_size,
        corpus_chunk_size=50000,
        show_progress_bar=True,
        mrr_at_k=[256],
        ndcg_at_k=[],
        accuracy_at_k=[],
        precision_recall_at_k=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        map_at_k=[],
        full_recall_at_k=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    )
    valid_evaluator = PremiseRetrievalEvaluator(
        dataset=dataset_valid,
        name="valid",
        **evaluator_kwargs,
    )

    loss = MaskedCachedMultipleNegativesRankingLoss(model, mini_batch_size=model_args.mini_batch_size)

    dataset_train_huggingface = dataset_train.to_huggingface_dataset()
    dataset_valid_huggingface = dataset_valid.to_huggingface_dataset()

    data_collator = PremiseRetrievalDataCollator(
        train_dataset=dataset_train,
        all_datasets=[dataset_train, dataset_valid, dataset_test],
        tokenize_fn=model.tokenize,
        retrieval_mask_mode=data_args.retrieval_mask_mode,
    )

    callbacks: List[TrainerCallback] = []
    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset_train_huggingface,
        eval_dataset=dataset_valid_huggingface,
        data_collator=data_collator,
        loss=loss,
        # Note: native evaluator is bad because this is in-batch negative loss which is slow & random & dependent on batch size
        # evaluator=valid_evaluator,
        callbacks=callbacks,
    )

    if train_args.resume_from_checkpoint in ["True", "False"]:
        resume_from_checkpoint = train_args.resume_from_checkpoint == "True"
    else:
        resume_from_checkpoint = train_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_state()
    model_save_path = os.path.join(train_args.output_dir, "final")
    model.save_pretrained(model_save_path)
    with open(os.path.join(model_save_path, "revision"), "w") as f:
        f.write(dataset_train.revision + "\n")

    ## Test evaluation
    model.eval()
    with torch.inference_mode():
        corpus_premises = dataset_test.corpus.premises
        corpus_embeddings = model.encode(
            [premise.to_string() for premise in corpus_premises],
            show_progress_bar=True,
            batch_size=model_args.mini_batch_size,
            convert_to_tensor=True,
        )

        for (name, dataset) in [("valid", dataset_valid), ("test", dataset_test)]:
            if train_args.local_rank == 0:
                print("=" * 100)
                print(f"{name} evaluation")

            # Evaluate on the evaluation set
            evaluator = PremiseRetrievalEvaluator(
                dataset=dataset,
                name=f"{name}-{train_args.run_name}",
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
