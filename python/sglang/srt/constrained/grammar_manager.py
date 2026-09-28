from __future__ import annotations

import logging
import time
from concurrent import futures
from typing import TYPE_CHECKING, List

import torch

from sglang.srt.constrained.base_grammar_backend import (
    GrammarStats,
    InvalidGrammarObject,
    create_grammar_backend,
)
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject
from sglang.srt.distributed.communication_tags import P2PTag
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_serving
from sglang.srt.sampling.sampling_params import (
    get_request_reasoning_end_token_ids,
)

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class GrammarManager:
    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self.server_args = scheduler.server_args
        self.grammar_queue: List[Req] = []
        if not get_serving().skip_tokenizer_init:
            self.grammar_backend = create_grammar_backend(
                self.server_args,
                scheduler.tokenizer,
                scheduler.model_config.vocab_size,
                scheduler.model_config.hf_eos_token_id,
                think_end_ids=scheduler.model_config.think_end_ids,
            )
        else:
            self.grammar_backend = None

        self._enable_strict_thinking = (
            self.grammar_backend.enable_strict_thinking
            if self.grammar_backend is not None
            else False
        )

        # When glm_decoding_constraint_module is set, create_grammar_backend skips
        # the ReasonerGrammarBackend wrap. That leaves JSON grammars active from
        # token 0, so reasoning ("<think>...") cannot open. Build a dedicated
        # reasoner backend for JSON only; regex / ebnf / structural_tag keep
        # the GLM-managed path unchanged.
        self.json_reasoner_grammar_backend = None
        if (
            self.grammar_backend is not None
            and get_serving().glm_decoding_constraint_module
            and get_serving().reasoning_parser
            and (
                scheduler.model_config.think_end_ids
                or getattr(scheduler.tokenizer, "think_end_id", None) is not None
            )
        ):
            from sglang.srt.constrained.reasoner_grammar_backend import (
                ReasonerGrammarBackend,
            )
            from sglang.srt.parser.reasoning_parser import ReasoningParser

            reasoning_parser = ReasoningParser(
                model_type=get_serving().reasoning_parser,
                stream_reasoning=False,
                tokenizer=scheduler.tokenizer,
            )
            self.json_reasoner_grammar_backend = ReasonerGrammarBackend(
                self.grammar_backend,
                reasoning_parser,
                scheduler.tokenizer,
                enable_strict_thinking=get_serving().enable_strict_thinking,
            )

        self.grammar_sync_group = scheduler.dp_tp_cpu_group
        self.grammar_sync_size = scheduler.dp_tp_group.world_size
        self.grammar_sync_entry = scheduler.dp_tp_group.first_rank
        self.is_grammar_sync_entry = scheduler.dp_tp_group.is_first_rank
        self.grammar_cp_sync_group = scheduler.attn_cp_cpu_group
        # With DP attention, dp_tp_group contains only attention-TP ranks.
        # CP ranks share requests too and must admit compiled grammars together.
        self.grammar_cp_sync_size = (
            scheduler.ps.attn_cp_size if scheduler.enable_dp_attention else 1
        )
        self.pp_rank = scheduler.ps.pp_rank
        self.pp_size = scheduler.ps.pp_size
        self.pp_group = scheduler.pp_group
        self.grammar_pp_sync_work_list = []

        self.SGLANG_GRAMMAR_POLL_INTERVAL = envs.SGLANG_GRAMMAR_POLL_INTERVAL.get()
        self.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = (
            envs.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS.get()
        )

    def __len__(self):
        return len(self.grammar_queue)

    def _get_grammar_backend(self, key):
        # Route JSON through the reasoner-wrapped backend when it exists;
        # regex / structural_tag / ebnf keep the original grammar_backend.
        if (
            self.json_reasoner_grammar_backend is not None
            and key is not None
            and key[0] == "json"
        ):
            return self.json_reasoner_grammar_backend
        return self.grammar_backend

    def _log_grammar_stats(self, grammar_stats) -> None:
        # GLM NOTE: emitted when the grammar attaches (cache hit / compile
        # done / timeout); tree_traversal_time has no writers yet, so nothing
        # is lost by not waiting for request finish.
        if grammar_stats is not None and self.scheduler.metrics_reporter.enable_metrics:
            self.scheduler.metrics_collector.log_grammar_stats(grammar_stats)

    def get_cache_stats(self):
        if self.grammar_backend is None:
            return 0, 0
        return self.grammar_backend.get_cache_stats()

    def clear(self):
        if self.grammar_backend:
            self.grammar_backend.reset()
        if self.json_reasoner_grammar_backend is not None:
            self.json_reasoner_grammar_backend.reset()

    def has_waiting_grammars(self) -> bool:
        return len(self.grammar_queue) > 0

    def _drain_pp_sync_work(self):
        for p2p_work in self.grammar_pp_sync_work_list:
            p2p_work.work.wait()
        self.grammar_pp_sync_work_list.clear()

    def _pp_sync_ready_failed(
        self,
        ready_req_idxs: set[int],
        failed_req_idxs: set[int],
    ) -> tuple[set[int], set[int]]:
        """
        Synchronize ready/failed grammar request indexes across the PP pipeline.

        PP0 provides the data. Each later PP rank receives it from the previous
        rank and asynchronously forwards it to the next rank.
        """
        if self.pp_size <= 1 or self.pp_group is None:
            return ready_req_idxs, failed_req_idxs

        self._drain_pp_sync_work()
        data = (ready_req_idxs, failed_req_idxs)
        if self.pp_rank > 0:
            data = self.pp_group.recv_object(
                src=self.pp_rank - 1,
                tag=P2PTag.GRAMMAR_PP_SYNC,
            )
        if self.pp_rank + 1 < self.pp_size:
            self.grammar_pp_sync_work_list.extend(
                self.pp_group.send_object(
                    data,
                    dst=self.pp_rank + 1,
                    async_send=True,
                    tag=P2PTag.GRAMMAR_PP_SYNC,
                )
            )
        return data

    def abort_requests(self, recv_req: AbortReq):
        for req in self.grammar_queue:
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                logger.debug(f"Abort grammar queue request. {req.rid=}")
                if isinstance(req.grammar, futures.Future) and req.grammar:
                    req.grammar.cancel()
                req.set_finish_with_abort("Aborted by AbortReq.")

    def _get_request_thinking_budget(self, req: Req) -> int | None:
        custom_params = req.sampling_params.custom_params
        if not isinstance(custom_params, dict):
            return None
        thinking_budget = custom_params.get("thinking_budget")
        return thinking_budget if isinstance(thinking_budget, int) else None

    def _apply_request_reasoning_config(self, req: Req) -> None:
        if not isinstance(req.grammar, ReasonerGrammarObject):
            return
        think_end_ids = get_request_reasoning_end_token_ids(
            req.sampling_params.custom_params,
            allowed_sequences=getattr(
                self.scheduler.model_config,
                "request_selectable_think_end_id_sequences",
                None,
            ),
        )
        if think_end_ids is not None:
            req.grammar.set_request_think_end_ids(think_end_ids)
        thinking_budget = self._get_request_thinking_budget(req)
        if thinking_budget is not None:
            req.grammar.max_think_tokens = thinking_budget

    def process_req_with_grammar(self, req: Req) -> bool:
        # Init grammar cache for this request
        add_to_grammar_queue = False
        if (
            req.sampling_params.json_schema is not None
            or req.sampling_params.regex is not None
            or req.sampling_params.ebnf is not None
            or req.sampling_params.structural_tag is not None
        ):
            if self.grammar_backend is None:
                error_msg = "Grammar-based generation (json_schema, regex, ebnf, structural_tag) is not supported when the server is launched with --grammar-backend none"
                req.set_finish_with_abort(error_msg)
            else:
                if req.sampling_params.json_schema is not None:
                    key = ("json", req.sampling_params.json_schema)
                elif req.sampling_params.regex is not None:
                    key = ("regex", req.sampling_params.regex)
                elif req.sampling_params.ebnf is not None:
                    key = ("ebnf", req.sampling_params.ebnf)
                elif req.sampling_params.structural_tag is not None:
                    key = ("structural_tag", req.sampling_params.structural_tag)

                value, cache_hit = self._get_grammar_backend(
                    key
                ).get_cached_or_future_value(key, req.require_reasoning)
                req.grammar = value

                if not cache_hit:
                    req.grammar_key = key
                    add_to_grammar_queue = True
                else:
                    if isinstance(
                        value, InvalidGrammarObject
                    ):  # We hit a cached invalid grammar.
                        error_msg = (
                            f"Failed to compile {key[0]} grammar: {value.error_message}"
                        )
                        req.set_finish_with_abort(error_msg)
                    else:
                        self._apply_request_reasoning_config(req)
                        self._log_grammar_stats(value.grammar_stats)
        elif self._enable_strict_thinking:
            grammar_obj = self.grammar_backend.init_strict_reasoning_grammar(
                req.require_reasoning
            )
            if grammar_obj is not None:
                req.grammar = grammar_obj
                self._apply_request_reasoning_config(req)

        if add_to_grammar_queue:
            self.grammar_queue.append(req)

        return add_to_grammar_queue

    def get_ready_grammar_requests(self) -> List[Req]:
        """
        Move requests whose grammar objects are ready from grammar_queue to waiting_queue.

        For PP0, each attention TP/CP rank i returns two sets ready_reqs_i,
        failed_reqs_i. Gather both sets across the ranks sharing each request,
        first within the attention TP group, then within the CP group.

        ready_reqs = intersect(ready_reqs_all)
        failed_reqs = union(failed_reqs_all)

        PP0 then propagates the synced result to later PP ranks. Later PP
        ranks receive and apply the propagated ready/failed decision.
        """
        assert self.grammar_backend
        ready_req_idxs: set[int] = set()
        failed_req_idxs: set[int] = set()

        if self.pp_rank == 0:
            # Poll for ready requests
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < self.SGLANG_GRAMMAR_POLL_INTERVAL:
                for i, req in enumerate(self.grammar_queue):
                    if i in ready_req_idxs:
                        continue

                    if (
                        req.finished() or req.grammar is None
                    ):  # It is aborted by AbortReq
                        ready_req_idxs.add(i)
                        continue

                    assert isinstance(req.grammar, futures.Future), f"{req=}"
                    if req.grammar.done():
                        ready_req_idxs.add(i)

                if len(ready_req_idxs) == len(self.grammar_queue):
                    break

                # Sleep a bit to avoid busy waiting
                time.sleep(self.SGLANG_GRAMMAR_POLL_INTERVAL / 10)

            # Check failed requests
            for i, req in enumerate(self.grammar_queue):
                if i not in ready_req_idxs:
                    # grammar_wait_ct is only updated on PP0; later PP ranks
                    # receive PP0's ready/failed decision through PP sync.
                    self.grammar_queue[i].grammar_wait_ct += 1
                    if (
                        self.grammar_queue[i].grammar_wait_ct
                        >= self.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS
                    ):
                        # Timeout after max poll iterations
                        # The actual waiting time is SGLANG_GRAMMAR_MAX_POLL_ITERATIONS * max(SGLANG_GRAMMAR_POLL_INTERVAL, GPU_forward_batch_latency)
                        failed_req_idxs.add(i)

            # Sync within each request's TP/CP shard without crossing DP replicas.
            synced_ready_req_idxs = ready_req_idxs
            synced_failed_req_idxs = failed_req_idxs
            for group, size in (
                (self.grammar_sync_group, self.grammar_sync_size),
                (self.grammar_cp_sync_group, self.grammar_cp_sync_size),
            ):
                if size == 1:
                    continue
                all_gather_output = [None] * size
                torch.distributed.all_gather_object(
                    all_gather_output,
                    (synced_ready_req_idxs, synced_failed_req_idxs),
                    group=group,
                )
                synced_ready_req_idxs = set.intersection(
                    *[x[0] for x in all_gather_output]
                )
                synced_failed_req_idxs = set.union(*[x[1] for x in all_gather_output])
        else:
            synced_ready_req_idxs = ready_req_idxs
            synced_failed_req_idxs = failed_req_idxs

        # Propagate PP0's grammar queue decision to later PP ranks.
        (
            synced_ready_req_idxs,
            synced_failed_req_idxs,
        ) = self._pp_sync_ready_failed(
            synced_ready_req_idxs,
            synced_failed_req_idxs,
        )

        # Return ready requests
        return_reqs: List[Req] = []
        for i in synced_ready_req_idxs:
            req = self.grammar_queue[i]
            return_reqs.append(req)
            if req.finished() or req.grammar is None:  # It is aborted by AbortReq
                continue

            assert isinstance(req.grammar, futures.Future) and req.grammar_key
            try:
                req.grammar = req.grammar.result()
            except Exception as e:
                logger.error(
                    f"Grammar compilation raised an exception: {e}, "
                    f"grammar_key={req.grammar_key}"
                )
                req.grammar = InvalidGrammarObject(f"Grammar compilation failed: {e}")
            self._get_grammar_backend(req.grammar_key).set_cache(
                req.grammar_key, req.grammar.copy()
            )
            self._apply_request_reasoning_config(req)
            if isinstance(req.grammar, InvalidGrammarObject):
                error_msg = f"Failed to compile {req.grammar_key[0]} grammar: {req.grammar.error_message}"
                req.set_finish_with_abort(error_msg)
            else:
                self._log_grammar_stats(req.grammar.grammar_stats)

        # Return failed requests
        for i in synced_failed_req_idxs:
            req = self.grammar_queue[i]
            return_reqs.append(req)

            assert isinstance(req.grammar, futures.Future) and req.grammar_key
            req.grammar.cancel()
            self._get_grammar_backend(req.grammar_key).set_cache(
                req.grammar_key, InvalidGrammarObject("Grammar preprocessing timed out")
            )
            error_msg = f"Grammar preprocessing timed out: {req.grammar_key=}"
            req.set_finish_with_abort(error_msg)
            self._log_grammar_stats(GrammarStats(num_timeout=1))

        # Remove finished requests from grammar_queue
        self.grammar_queue = [
            req
            for i, req in enumerate(self.grammar_queue)
            if i not in synced_ready_req_idxs and i not in synced_failed_req_idxs
        ]
        return return_reqs
