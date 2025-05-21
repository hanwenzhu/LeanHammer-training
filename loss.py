# Wrapper around CachedMultipleNegativesRankingLoss

from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from typing import Any, Iterable, Iterator, Callable, Optional

import torch
import tqdm
from torch import Tensor, nn
from torch.utils.checkpoint import get_device_states, set_device_states

from sentence_transformers import SentenceTransformer, util
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.losses.CachedMultipleNegativesRankingLoss import RandContext, _backward_hook

class MaskedCachedMultipleNegativesRankingLoss(CachedMultipleNegativesRankingLoss):
    """Wrapper around CachedMultipleNegativesRankingLoss.

    Allows the input batch to contain a `retrieval_mask` of shape `(batch_size, (1 + num_negatives) * batch_size)`
    such that `retrieval_mask[i, j]` is `true` iff the i-th query should retrieve from
    `{ j-th document | retrieval_mask[i, j] == 1 }`.

    In other words, the only change is adding `scores[~retrieval_mask] = -torch.inf`.

    See `PremiseRetrievalDataCollator` that supplies `retrieval_mask`.
    """

    # def embed_minibatch(
    #     self,
    #     sentence_feature: dict[str, Tensor],
    #     begin: int,
    #     end: int,
    #     with_grad: bool,
    #     copy_random_state: bool,
    #     random_state: RandContext | None = None,
    # ) -> tuple[Tensor, RandContext | None]:
    #     """Do forward pass on a minibatch of the input features and return corresponding embeddings."""
    #     grad_context = nullcontext if with_grad else torch.no_grad
    #     random_state_context = nullcontext() if random_state is None else random_state
    #     sentence_feature_minibatch = {k: v[begin:end] for k, v in sentence_feature.items()}
    #     with random_state_context:
    #         with grad_context():
    #             random_state = RandContext(*sentence_feature_minibatch.values()) if copy_random_state else None
    #             reps = self.model(sentence_feature_minibatch)["sentence_embedding"]  # (mbsz, hdim)
    #     return reps, random_state

    # def embed_minibatch_iter(
    #     self,
    #     sentence_feature: dict[str, Tensor],
    #     with_grad: bool,
    #     copy_random_state: bool,
    #     random_states: list[RandContext] | None = None,
    # ) -> Iterator[tuple[Tensor, RandContext | None]]:
    #     """Do forward pass on all the minibatches of the input features and yield corresponding embeddings."""
    #     input_ids: Tensor = sentence_feature["input_ids"]
    #     bsz, _ = input_ids.shape
    #     for i, b in enumerate(
    #         tqdm.trange(
    #             0,
    #             bsz,
    #             self.mini_batch_size,
    #             desc="Embed mini-batches",
    #             disable=not self.show_progress_bar,
    #         )
    #     ):
    #         e = b + self.mini_batch_size
    #         reps, random_state = self.embed_minibatch(
    #             sentence_feature=sentence_feature,
    #             begin=b,
    #             end=e,
    #             with_grad=with_grad,
    #             copy_random_state=copy_random_state,
    #             random_state=None if random_states is None else random_states[i],
    #         )
    #         yield reps, random_state  # reps: (mbsz, hdim)

    def calculate_loss_and_cache_gradients(self, reps: list[list[Tensor]], retrieval_mask: Optional[Tensor] = None) -> Tensor:
        """Calculate the cross-entropy loss and cache the gradients wrt. the embeddings."""
        embeddings_a = torch.cat(reps[0])  # (bsz, hdim)
        embeddings_b = torch.cat([torch.cat(r) for r in reps[1:]])  # ((1 + nneg) * bsz, hdim)

        batch_size = len(embeddings_a)
        labels = torch.tensor(
            range(batch_size), dtype=torch.long, device=embeddings_a.device
        )  # (bsz, (1 + nneg) * bsz)  Example a[i] should match with b[i]
        losses: list[torch.Tensor] = []
        for b in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Preparing caches",
            disable=not self.show_progress_bar,
        ):
            e = b + self.mini_batch_size
            scores: Tensor = self.similarity_fct(embeddings_a[b:e], embeddings_b) * self.scale
            if retrieval_mask is not None:
                scores[~retrieval_mask[b:e]] = -torch.inf
            loss_mbatch: torch.Tensor = self.cross_entropy_loss(scores, labels[b:e]) * len(scores) / batch_size
            loss_mbatch.backward()
            losses.append(loss_mbatch.detach())

        loss = sum(losses).requires_grad_()  # type: ignore

        self.cache = [[r.grad for r in rs] for rs in reps]  # e.g. 3 * bsz/mbsz * (mbsz, hdim)  # type: ignore

        return loss

    def calculate_loss(self, reps: list[list[Tensor]], retrieval_mask: Optional[Tensor] = None) -> Tensor:
        """Calculate the cross-entropy loss. No need to cache the gradients."""
        embeddings_a = torch.cat(reps[0])  # (bsz, hdim)
        embeddings_b = torch.cat([torch.cat(r) for r in reps[1:]])  # ((1 + nneg) * bsz, hdim)

        batch_size = len(embeddings_a)
        labels = torch.tensor(
            range(batch_size), dtype=torch.long, device=embeddings_a.device
        )  # (bsz, (1 + nneg) * bsz)  Example a[i] should match with b[i]
        losses: list[torch.Tensor] = []
        for b in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Preparing caches",
            disable=not self.show_progress_bar,
        ):
            e = b + self.mini_batch_size
            scores: Tensor = self.similarity_fct(embeddings_a[b:e], embeddings_b) * self.scale
            if retrieval_mask is not None:
                scores[~retrieval_mask[b:e]] = -torch.inf
            loss_mbatch: torch.Tensor = self.cross_entropy_loss(scores, labels[b:e]) * len(scores) / batch_size
            losses.append(loss_mbatch)

        loss = sum(losses)
        return loss  # type: ignore

    # Here, `sentence_features` is:
    # [
    #   {"input_ids": ..., "attention_mask": ..., "retrieval_mask": ...},  # queries
    #   {"input_ids": ..., "attention_mask": ...},  # positives
    #   {"input_ids": ..., "attention_mask": ...},  # negatives 0
    #   {"input_ids": ..., "attention_mask": ...},  # negatives 1 ...
    # ]
    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Step (1): A quick embedding step without gradients/computation graphs to get all the embeddings
        reps = []
        self.random_states = []  # Copy random states to guarantee exact reproduction of the embeddings during the second forward pass, i.e. step (3)
        retrieval_mask = None
        for sentence_feature in sentence_features:
            reps_mbs = []
            random_state_mbs = []
            for reps_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            self.random_states.append(random_state_mbs)

            # Added mask
            if "retrieval_mask" in sentence_feature:
                assert retrieval_mask is None, "multiple retrieval_mask for loss"
                retrieval_mask = sentence_feature["retrieval_mask"]

        if torch.is_grad_enabled():
            # Step (2): Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings
            loss = self.calculate_loss_and_cache_gradients(reps, retrieval_mask)

            # Step (3): A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain
            loss.register_hook(partial(_backward_hook, sentence_features=sentence_features, loss_obj=self))
        else:
            # If grad is not enabled (e.g. in evaluation), then we don't have to worry about the gradients or backward hook
            loss = self.calculate_loss(reps, retrieval_mask)

        return loss
