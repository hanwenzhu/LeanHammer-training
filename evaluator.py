import json
import logging
import os
from contextlib import nullcontext
from typing import List, Callable, Optional
import heapq

import numpy as np
from sentence_transformers.evaluation import SentenceEvaluator, InformationRetrievalEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.util import cos_sim, dot_score
from sentence_transformers import SentenceTransformer
import torch
from torch import Tensor
from tqdm import trange

from data import RetrievalDataset

logger = logging.getLogger(__name__)


class PremiseRetrievalEvaluator(SentenceEvaluator):
    """Adapted from InformationRetrievalEvaluator with almost no changes, except:
    * Added FullRecall@k.
    * Retrieve only from accessible premises (and not the declaration itself).
    * By default, only do cosine similarity
    * Allow empty k lists
    """

    def __init__(
        self,
        dataset: RetrievalDataset,
        restrict_to_accessible: bool = True,
        corpus_chunk_size: int = 50000,
        mrr_at_k: list[int] = [],
        ndcg_at_k: list[int] = [],
        accuracy_at_k: list[int] = [],
        precision_recall_at_k: list[int] = [],
        map_at_k: list[int] = [],
        full_recall_at_k: List[int] = [],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = "",
        write_csv: bool = True,
        truncate_dim: int | None = None,
        score_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
            # SimilarityFunction.COSINE.value: cos_sim,
            SimilarityFunction.DOT_PRODUCT.value: dot_score,
        },  # Score function, higher=more similar
        main_score_function: str | SimilarityFunction | None = None,
        retrieved_premises_save_dir: Optional[str] = None,
    ):
        super().__init__()

        self.states = dataset.states
        self.corpus = dataset.corpus
        self.relevant_docs = {n: set(ps) for n, ps in dataset.relevant_premises.items()}
        self.accessible_premises = {state.name: dataset.get_accessible_premises(state) for state in dataset.states}
        for state in dataset.states:
            if state.decl_name in dataset.corpus.name2premise:
                self.accessible_premises[state.name].remove(dataset.corpus.name2premise[state.decl_name])
        self.restrict_to_accessible = restrict_to_accessible

        self.corpus_chunk_size = corpus_chunk_size
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k
        self.full_recall_at_k = full_recall_at_k

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys()))
        self.main_score_function = SimilarityFunction(main_score_function) if main_score_function else None
        self.truncate_dim = truncate_dim
        self.retrieved_premises_save_dir = retrieved_premises_save_dir

        if name:
            name = "_" + name

        self.csv_file: str = "Premise-Retrieval_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]

        for score_name in self.score_function_names:
            for k in accuracy_at_k:
                self.csv_headers.append(f"{score_name}-Accuracy@{k}")

            for k in precision_recall_at_k:
                self.csv_headers.append(f"{score_name}-Precision@{k}")
                self.csv_headers.append(f"{score_name}-Recall@{k}")

            for k in mrr_at_k:
                self.csv_headers.append(f"{score_name}-MRR@{k}")

            for k in ndcg_at_k:
                self.csv_headers.append(f"{score_name}-NDCG@{k}")

            for k in map_at_k:
                self.csv_headers.append(f"{score_name}-MAP@{k}")

            for k in full_recall_at_k:
                self.csv_headers.append(f"{score_name}-FullRecall@{k}")

    def __call__(
        self, model: SentenceTransformer, output_path: Optional[str] = None, epoch: int = -1, steps: int = -1, *args, **kwargs
    ) -> dict[str, float]:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(f"Premise Retrieval Evaluation of the model on the {self.name} dataset{out_txt}:")

        scores = self.compute_metrices(model, *args, **kwargs)

        # Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data: List[int | float] = [epoch, steps]
            for name in self.score_function_names:
                for k in self.accuracy_at_k:
                    output_data.append(scores[name]["accuracy@k"][k])

                for k in self.precision_recall_at_k:
                    output_data.append(scores[name]["precision@k"][k])
                    output_data.append(scores[name]["recall@k"][k])

                for k in self.mrr_at_k:
                    output_data.append(scores[name]["mrr@k"][k])

                for k in self.ndcg_at_k:
                    output_data.append(scores[name]["ndcg@k"][k])

                for k in self.map_at_k:
                    output_data.append(scores[name]["map@k"][k])

                for k in self.full_recall_at_k:
                    output_data.append(scores[name]["fullrecall@k"][k])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        if not self.primary_metric:
            if self.main_score_function is None:
                score_function = max(
                    [(name, scores[name]["recall@k"][max(self.precision_recall_at_k)]) for name in self.score_function_names],
                    key=lambda x: x[1],
                )[0]
                self.primary_metric = f"{score_function}_recall@{max(self.precision_recall_at_k)}"
            else:
                self.primary_metric = f"{self.main_score_function.value}_recall@{max(self.precision_recall_at_k)}"

        metrics = {
            f"{score_function}_{metric_name.replace('@k', '@' + str(k))}": value
            for score_function, values_dict in scores.items()
            for metric_name, values in values_dict.items()
            for k, value in values.items()
        }
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics)
        return metrics

    def compute_metrices(
        self, model: SentenceTransformer, corpus_model=None, corpus_embeddings: Optional[Tensor] = None
    ) -> dict[str, dict[str, dict[int, float]]]:
        if corpus_model is None:
            corpus_model = model

        max_k = max(
            self.mrr_at_k + self.ndcg_at_k + self.accuracy_at_k +
            self.precision_recall_at_k + self.map_at_k + self.full_recall_at_k
        )

        # Compute embedding for the queries
        with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(self.truncate_dim):
            query_embeddings = model.encode(
                [state.to_string() for state in self.states],
                show_progress_bar=self.show_progress_bar,
                batch_size=self.batch_size,
                convert_to_tensor=True,
            )

        queries_result_list: dict[str, list[list[dict[str, str | float] | tuple[float, float]]]] = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

        # Iterate over chunks of the corpus
        for corpus_start_idx in trange(
            0, len(self.corpus.premises), self.corpus_chunk_size, desc="Corpus Chunks", disable=not self.show_progress_bar
        ):
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus.premises))

            # Encode chunk of corpus
            if corpus_embeddings is None:
                with nullcontext() if self.truncate_dim is None else corpus_model.truncate_sentence_embeddings(
                    self.truncate_dim
                ):
                    sub_corpus_embeddings = corpus_model.encode(
                        [premise.to_string() for premise in self.corpus.premises[corpus_start_idx:corpus_end_idx]],
                        show_progress_bar=False,
                        batch_size=self.batch_size,
                        convert_to_tensor=True,
                    )
            else:
                sub_corpus_embeddings = corpus_embeddings[corpus_start_idx:corpus_end_idx]

            # Compute cosine similarites
            for name, score_function in self.score_functions.items():
                pair_scores = score_function(query_embeddings, sub_corpus_embeddings)

                # Set non-accessible premises to have -inf score
                if self.restrict_to_accessible:
                    for query_itr in range(len(query_embeddings)):
                        for premise_idx, premise in enumerate(self.corpus.premises[corpus_start_idx:corpus_end_idx]):
                            if premise not in self.accessible_premises[self.states[query_itr].name]:
                                pair_scores[query_itr][premise_idx] = -torch.inf

                # Get top-k values
                pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(
                    pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False
                )
                pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(query_embeddings)):
                    for sub_corpus_id, score in zip(
                        pair_scores_top_k_idx[query_itr], pair_scores_top_k_values[query_itr]
                    ):
                        corpus_id = self.corpus.premises[corpus_start_idx + sub_corpus_id].name
                        if len(queries_result_list[name][query_itr]) < max_k:
                            heapq.heappush(
                                queries_result_list[name][query_itr], (score, corpus_id)
                            )  # heaqp tracks the quantity of the first element in the tuple
                        else:
                            heapq.heappushpop(queries_result_list[name][query_itr], (score, corpus_id))

        for name in queries_result_list:
            for query_itr in range(len(queries_result_list[name])):
                for doc_itr in range(len(queries_result_list[name][query_itr])):
                    score, corpus_id = queries_result_list[name][query_itr][doc_itr]
                    queries_result_list[name][query_itr][doc_itr] = {"corpus_id": corpus_id, "score": score}

        logger.info(f"Queries: {len(self.states)}")
        logger.info(f"Corpus: {len(self.corpus.premises)}\n")

        # Compute scores
        scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}  # type: ignore
        if self.retrieved_premises_save_dir is not None:
            for name in self.score_functions:
                self.save_premises(self.retrieved_premises_save_dir, name, queries_result_list[name])  # type: ignore

        # Output
        for name in self.score_function_names:
            logger.info(f"Score-Function: {name}")
            self.output_scores(scores[name])

        return scores

    def save_premises(self, save_dir: str, name: str, queries_result_list: list[list[dict[str, str | float]]]):
        results = [
            {"state_name": state.name, "decl_name": state.decl_name, "module": state.module, "idx_in_module": state.idx_in_module, "premises": queries_result}
            for state, queries_result in zip(self.states, queries_result_list)
        ]
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{name}_{self.name}.json"), "w") as f:
            json.dump(results, f)

    def compute_metrics(self, queries_result_list: list[list[dict[str, str | float]]]):
        # Init score computation values
        num_hits_at_k = {k: 0. for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0. for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}
        full_recall_at_k = {k: [] for k in self.full_recall_at_k}

        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            state = self.states[query_itr]

            # Sort scores
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: x["score"], reverse=True)
            query_relevant_docs = self.relevant_docs[state.name]

            # Accuracy@k - We count the result correct, if at least one relevant doc is across the top-k documents
            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break

            # Precision and Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1

                precisions_at_k[k_val].append(num_correct / k_val)
                if len(query_relevant_docs) > 0:
                    # For recall, filter only to states that have > 0 relevant premises
                    recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [
                    1 if top_hit["corpus_id"] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]
                ]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(
                    true_relevances, k_val
                )
                ndcg[k_val].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k[k_val].append(avg_precision)

            # FullRecall@k
            for k_val in self.full_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1
                full_recall_at_k[k_val].append(1. if num_correct == len(query_relevant_docs) else 0.)

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(self.states)

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k]).item()  # type: ignore

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k]).item()  # type: ignore

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k]).item()  # type: ignore

        for k in MRR:
            MRR[k] /= len(self.states)

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k]).item()  # type: ignore

        for k in full_recall_at_k:
            full_recall_at_k[k] = np.mean(full_recall_at_k[k]).item()  # type: ignore

        return {
            "accuracy@k": num_hits_at_k,
            "precision@k": precisions_at_k,
            "recall@k": recall_at_k,
            "ndcg@k": ndcg,
            "mrr@k": MRR,
            "map@k": AveP_at_k,
            "fullrecall@k": full_recall_at_k,
        }

    def output_scores(self, scores):
        for k in scores["accuracy@k"]:
            logger.info("Accuracy@{}: {:.2f}%".format(k, scores["accuracy@k"][k] * 100))

        for k in scores["precision@k"]:
            logger.info("Precision@{}: {:.2f}%".format(k, scores["precision@k"][k] * 100))

        for k in scores["recall@k"]:
            logger.info("Recall@{}: {:.2f}%".format(k, scores["recall@k"][k] * 100))

        for k in scores["mrr@k"]:
            logger.info("MRR@{}: {:.4f}".format(k, scores["mrr@k"][k]))

        for k in scores["ndcg@k"]:
            logger.info("NDCG@{}: {:.4f}".format(k, scores["ndcg@k"][k]))

        for k in scores["map@k"]:
            logger.info("MAP@{}: {:.4f}".format(k, scores["map@k"][k]))

        for k in scores["fullrecall@k"]:
            logger.info("FullRecall@{}: {:.2f}%".format(k, scores["fullrecall@k"][k] * 100))

    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
        return dcg
