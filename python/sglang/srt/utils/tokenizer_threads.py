from sglang.srt.environ import envs


def cap_torch_intraop_threads() -> None:
    """Cap torch intra-op parallelism for the calling tokenizer thread."""
    num_threads = envs.GLM_TOKENIZER_TORCH_NUM_THREADS.get()
    if num_threads <= 0:
        return

    # Keep torch out of this module's import graph until the worker starts.
    import torch

    torch.set_num_threads(num_threads)
