"""Row views over independently received EPD parts."""

import torch


class EmbeddingChunks:
    def __init__(self, tensors):
        self.tensors = tuple(tensors)
        if not self.tensors:
            raise ValueError("EmbeddingChunks requires at least one part")
        tail = self.tensors[0].shape[1:]
        self.dtype = self.tensors[0].dtype
        self.device = self.tensors[0].device
        if any(
            t.shape[1:] != tail or t.dtype != self.dtype or t.device != self.device
            for t in self.tensors
        ):
            raise ValueError("Incompatible EPD embedding parts")
        self.shape = (sum(t.shape[0] for t in self.tensors), *tail)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, rows):
        if not isinstance(rows, slice) or rows.step not in (None, 1):
            raise TypeError("EPD embedding chunks only support contiguous row slices")
        start, stop, _ = rows.indices(len(self))
        pieces = []
        offset = 0
        for tensor in self.tensors:
            lo, hi = max(0, start - offset), min(tensor.shape[0], stop - offset)
            if lo < hi:
                pieces.append(tensor[lo:hi])
            offset += tensor.shape[0]
        if not pieces:
            return self.tensors[0][:0]
        # Per-image slices normally stay in one part: preserve the pool view.
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
