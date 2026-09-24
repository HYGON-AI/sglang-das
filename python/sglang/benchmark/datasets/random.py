import json
import random
from argparse import Namespace
from dataclasses import dataclass
from typing import List

import numpy as np
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import (
    SHAREGPT_FILENAME,
    SHAREGPT_REPO_ID,
    BaseDataset,
    DatasetRow,
    compute_random_lens,
)
from sglang.benchmark.utils import download_and_cache_hf_file, is_file_valid_json


@dataclass
class RandomDataset(BaseDataset):
    input_len: int
    output_len: int
    num_requests: int
    range_ratio: float
    dataset_path: str
    return_text: bool
    random_sample: bool

    @classmethod
    def from_args(cls, args: Namespace) -> "RandomDataset":
        return cls(
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_requests=args.num_prompts,
            range_ratio=args.random_range_ratio,
            dataset_path=args.dataset_path,
            return_text=not getattr(args, "tokenize_prompt", False),
            random_sample=(args.dataset_name == "random"),
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        return sample_random_requests(
            input_len=self.input_len,
            output_len=self.output_len,
            num_prompts=self.num_requests,
            range_ratio=self.range_ratio,
            tokenizer=tokenizer,
            dataset_path=self.dataset_path,
            random_sample=self.random_sample,
            return_text=self.return_text,
        )


def sample_random_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
    random_sample: bool = True,
    return_text: bool = True,
) -> List[DatasetRow]:
    input_lens = compute_random_lens(
        full_len=input_len,
        range_ratio=range_ratio,
        num=num_prompts,
    )
    output_lens = compute_random_lens(
        full_len=output_len,
        range_ratio=range_ratio,
        num=num_prompts,
    )

    if return_text:
        # Need to truncate input_len as server encode will add special token.
        num_special_tokens = int(tokenizer.num_special_tokens_to_add())
        for i in range(num_prompts):
            input_lens[i] = max(1, input_lens[i] - num_special_tokens)

    if random_sample:
        # Sample token ids from ShareGPT and repeat/truncate them to satisfy the input_lens

        # Download sharegpt if necessary
        if not is_file_valid_json(dataset_path):
            dataset_path = download_and_cache_hf_file(
                repo_id=SHAREGPT_REPO_ID,
                filename=SHAREGPT_FILENAME,
            )

        # Load the dataset.
        with open(dataset_path) as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [
            data
            for data in dataset
            if len(data.get("conversations", data.get("conversation", []))) >= 2
        ]
        # Only keep the first two turns of each conversation.
        dataset = [
            (
                data.get("conversations", data.get("conversation", []))[0]["value"],
                data.get("conversations", data.get("conversation", []))[1]["value"],
            )
            for data in dataset
        ]
        # Shuffle the dataset.
        random.shuffle(dataset)

        # Filter out sequences that are too long or too short
        input_requests: List[DatasetRow] = []
        for data in dataset:
            i = len(input_requests)
            if i == num_prompts:
                break

            # Tokenize the prompts and completions.
            prompt = data[0]
            prompt_token_ids = tokenizer.encode(prompt)
            prompt_len = len(prompt_token_ids)

            # Skip empty prompt
            if prompt_len == 0:
                continue

            if prompt_len > input_lens[i]:
                input_ids = prompt_token_ids[: input_lens[i]]
            else:
                ratio = (input_lens[i] + prompt_len - 1) // prompt_len
                input_ids = (prompt_token_ids * ratio)[: input_lens[i]]
            input_content = input_ids
            if return_text:
                input_content = tokenizer.decode(input_content)
            input_requests.append(
                DatasetRow(
                    prompt=input_content,
                    prompt_len=input_lens[i],
                    output_len=output_lens[i],
                )
            )
    else:
        # Sample token ids from random integers. This can cause some NaN issues.
        offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
        input_requests = []
        for i in range(num_prompts):
            # Use int() to convert numpy.int64 to native Python int for JSON serialization
            input_content = [
                int((offsets[i] + i + j) % tokenizer.vocab_size)
                for j in range(input_lens[i])
            ]
            if return_text:
                input_content = tokenizer.decode(input_content)
            input_requests.append(
                DatasetRow(
                    prompt=input_content,
                    prompt_len=input_lens[i],
                    output_len=output_lens[i],
                )
            )

    print(f"#Input tokens: {np.sum(input_lens)}")
    print(f"#Output tokens: {np.sum(output_lens)}")
    return input_requests


@dataclass
class RandomIdsRawDataset(BaseDataset):
    """Sample prompts as raw token id sequences from ShareGPT.

    Unlike `random`, this dataset never round-trips through `tokenizer.decode`,
    so the request length on the server side is exactly `input_len` (no drift
    from encode/decode mismatches). Output length is enforced by the sampling
    params, so this gives precise control over both input and output lengths.

    When `shared_prefix_len > 0`, the first `shared_prefix_len` token ids of
    every request are identical (sampled once from ShareGPT), so the server's
    radix cache hits exactly `shared_prefix_len` tokens after the first
    request. The unique-tail length is `input_len - shared_prefix_len`.
    """

    input_len: int
    output_len: int
    num_requests: int
    range_ratio: float
    dataset_path: str
    shared_prefix_len: int

    @classmethod
    def from_args(cls, args: Namespace) -> "RandomIdsRawDataset":
        return cls(
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_requests=args.num_prompts,
            range_ratio=args.random_range_ratio,
            dataset_path=args.dataset_path,
            shared_prefix_len=getattr(args, "random_ids_raw_shared_prefix_len", 0),
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        return sample_random_ids_raw_requests(
            input_len=self.input_len,
            output_len=self.output_len,
            num_prompts=self.num_requests,
            range_ratio=self.range_ratio,
            tokenizer=tokenizer,
            dataset_path=self.dataset_path,
            shared_prefix_len=self.shared_prefix_len,
        )


def sample_random_ids_raw_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
    shared_prefix_len: int = 0,
) -> List[DatasetRow]:
    """Build prompts as raw token id lists from ShareGPT, never decoded back to text."""
    if shared_prefix_len < 0:
        raise ValueError("shared_prefix_len must be >= 0")
    if shared_prefix_len > input_len:
        raise ValueError(
            f"shared_prefix_len ({shared_prefix_len}) must be <= input_len ({input_len})"
        )
    if shared_prefix_len > 0 and range_ratio != 1.0:
        # Variable-length requests would either truncate the shared prefix or
        # leave the tail empty, breaking the cache-hit guarantee.
        raise ValueError(
            "shared_prefix_len > 0 requires --random-range-ratio 1.0 so every "
            "request has the exact same total length."
        )

    input_lens = compute_random_lens(
        full_len=input_len,
        range_ratio=range_ratio,
        num=num_prompts,
    )
    output_lens = compute_random_lens(
        full_len=output_len,
        range_ratio=range_ratio,
        num=num_prompts,
    )

    if not is_file_valid_json(dataset_path):
        dataset_path = download_and_cache_hf_file(
            repo_id=SHAREGPT_REPO_ID,
            filename=SHAREGPT_FILENAME,
        )

    with open(dataset_path) as f:
        dataset = json.load(f)
    dataset = [
        data
        for data in dataset
        if len(data.get("conversations", data.get("conversation", []))) >= 2
    ]
    dataset = [
        data.get("conversations", data.get("conversation", []))[0]["value"]
        for data in dataset
    ]
    random.shuffle(dataset)

    shared_prefix_ids: List[int] = []
    if shared_prefix_len > 0:
        # Build the shared prefix DETERMINISTICALLY, independent of args.seed,
        # so multiple bench invocations (e.g. a warmup run + a measurement run
        # with different --seed values to avoid full-prompt cache hits) still
        # share the *same* prefix tokens. We pick the longest ShareGPT prompts
        # first to minimize concatenation count.
        prefix_pool = sorted(dataset, key=len, reverse=True)
        for prompt in prefix_pool:
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            shared_prefix_ids.extend(ids)
            if len(shared_prefix_ids) >= shared_prefix_len:
                break
        if len(shared_prefix_ids) < shared_prefix_len:
            ratio = (shared_prefix_len + len(shared_prefix_ids) - 1) // max(
                len(shared_prefix_ids), 1
            )
            shared_prefix_ids = shared_prefix_ids * ratio
        shared_prefix_ids = shared_prefix_ids[:shared_prefix_len]

    # Per-run salt for tail RNG. random.getstate() reflects the global seed
    # set by bench_serving.py (`random.seed(args.seed)`) PLUS the calls made
    # since (e.g. random.shuffle above), so reruns with a different --seed
    # produce different tails — preventing accidental cache hits from a
    # previous run that left identical prompts in the radix tree.
    tail_salt = random.getrandbits(32)

    input_requests: List[DatasetRow] = []
    dataset_iter = iter(dataset)
    for req_idx in range(num_prompts):
        target_len = input_lens[req_idx]
        tail_len = target_len - shared_prefix_len

        if tail_len <= 0:
            tail_ids: List[int] = []
        elif shared_prefix_len > 0:
            # When a shared prefix is requested, sample tails as random vocab
            # ids with a per-request, per-run seed. Per-request guarantees
            # pairwise-distinct tails (collision probability ~ 1/vocab_size at
            # the first token); per-run (tail_salt) guarantees consecutive
            # bench invocations produce different tails so the server's radix
            # cache from a prior run cannot accidentally fully hit.
            rng = np.random.default_rng(
                seed=(hash(("tail", req_idx, tail_salt)) & 0xFFFFFFFF)
            )
            tail_ids = rng.integers(
                low=0, high=tokenizer.vocab_size, size=tail_len, dtype=np.int64
            ).tolist()
        else:
            # Legacy path (no shared prefix): take tail from the next ShareGPT
            # prompt, repeating/truncating to tail_len.
            prompt = None
            while True:
                try:
                    candidate = next(dataset_iter)
                except StopIteration:
                    raise RuntimeError(
                        f"Ran out of ShareGPT prompts after {req_idx} requests "
                        f"(needed {num_prompts})."
                    )
                ids = tokenizer.encode(candidate, add_special_tokens=False)
                if len(ids) > 0:
                    prompt = ids
                    break
            if len(prompt) >= tail_len:
                tail_ids = prompt[:tail_len]
            else:
                ratio = (tail_len + len(prompt) - 1) // len(prompt)
                tail_ids = (prompt * ratio)[:tail_len]

        input_ids = shared_prefix_ids + list(tail_ids)
        assert len(input_ids) == target_len, (
            f"built prompt len {len(input_ids)} != target {target_len}"
        )

        input_requests.append(
            DatasetRow(
                prompt=[int(t) for t in input_ids],
                prompt_len=target_len,
                output_len=output_lens[req_idx],
            )
        )

    print(f"#Input tokens: {np.sum(input_lens)}")
    print(f"#Output tokens: {np.sum(output_lens)}")
    if shared_prefix_len > 0:
        print(
            f"#Shared prefix tokens per request: {shared_prefix_len} "
            f"(expect cache hit on all requests after the first)"
        )
    return input_requests
