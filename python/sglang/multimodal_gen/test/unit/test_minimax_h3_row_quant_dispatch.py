"""CPU-only metadata guard tests; no GPU numerical or performance claim.

Extract the real helper without importing SGLang, Torch, or Triton. Fake tensors
carry metadata only; even the offset boundary cases allocate no tensor storage.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


class MetadataTensor:
    def __init__(
        self, shape, dtype="bf16", *, strides=None, layout="strided", is_cuda=True
    ):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.layout = layout
        self.is_cuda = is_cuda
        self._strides = strides or ((shape[1], 1) if self.ndim == 2 else (1,))

    def stride(self, dimension):
        if self.layout != "strided":
            raise AssertionError("Non-strided layout must be rejected before stride()")
        return self._strides[dimension]


class TestMiniMaxH3RowQuantDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (
            Path(__file__).resolve().parents[2]
            / "runtime/layers/quantization/minimax_h3_quant.py"
        )
        names = {
            "MAX_ROW_QUANT_BLOCK_SIZE",
            "MAX_ROW_QUANT_LINEAR_OFFSET",
            "_row_quant_block_size",
        }
        nodes = []
        for node in ast.parse(source.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                nodes.append(node)
            elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in names
                for target in node.targets
            ):
                nodes.append(node)
        if len(nodes) != len(names):
            raise AssertionError("Expected the real helper and both resource constants")
        cls.namespace = {
            "torch": SimpleNamespace(
                bfloat16="bf16", float16="fp16", float32="fp32", strided="strided"
            ),
            "triton": SimpleNamespace(next_power_of_2=lambda k: 1 << (k - 1).bit_length()),
        }
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"),
            cls.namespace,
        )

    def block(self, tensor, precision):
        return self.namespace["_row_quant_block_size"](tensor, precision)

    def test_resource_constants(self):
        self.assertEqual(self.namespace["MAX_ROW_QUANT_BLOCK_SIZE"], 16384)
        self.assertEqual(self.namespace["MAX_ROW_QUANT_LINEAR_OFFSET"], (1 << 31) - 1)

    def test_h3_shapes_and_k_boundaries(self):
        expected = {
            1: 1,
            16: 16,
            5376: 8192,
            7168: 8192,
            8192: 8192,
            8193: 16384,
            12288: 16384,
            14336: 16384,
            16384: 16384,
            16385: None,
        }
        for precision in ("int8", "fp8"):
            for k, block in expected.items():
                with self.subTest(precision=precision, k=k):
                    tensor = MetadataTensor((33, k))
                    self.assertEqual(self.block(tensor, precision), block)

    def test_precision_and_dtype(self):
        for precision in ("int8", "fp8"):
            for dtype in ("bf16", "fp16", "fp32"):
                supported = dtype == "bf16" or (precision == "fp8" and dtype == "fp16")
                expected = 32 if supported else None
                with self.subTest(precision=precision, dtype=dtype):
                    tensor = MetadataTensor((3, 32), dtype)
                    self.assertEqual(self.block(tensor, precision), expected)
        with self.assertRaises(ValueError):
            self.block(MetadataTensor((3, 32)), "unknown")

    def test_row_stride_and_last_dimension(self):
        for precision in ("int8", "fp8"):
            for strides in ((12288, 1), (12352, 1), (0, 1)):
                with self.subTest(precision=precision, strides=strides):
                    tensor = MetadataTensor((33, 12288), strides=strides)
                    self.assertEqual(self.block(tensor, precision), 16384)
            for strides in ((24576, 2), (-12288, 1)):
                tensor = MetadataTensor((33, 12288), strides=strides)
                self.assertIsNone(self.block(tensor, precision))

    def test_device_layout_and_rank_rejections(self):
        invalid = (
            MetadataTensor((3, 32), is_cuda=False),
            MetadataTensor((3, 32), layout="sparse"),
            MetadataTensor((32,)),
            MetadataTensor((1, 3, 32)),
        )
        for precision in ("int8", "fp8"):
            for tensor in invalid:
                with self.subTest(
                    precision=precision, shape=tensor.shape, layout=tensor.layout
                ):
                    self.assertIsNone(self.block(tensor, precision))

    def test_empty_dimensions(self):
        for precision in ("int8", "fp8"):
            for shape in ((0, 32), (3, 0), (0, 0)):
                with self.subTest(precision=precision, shape=shape):
                    self.assertIsNone(self.block(MetadataTensor(shape), precision))

    def test_input_offset_boundary(self):
        limit = (1 << 31) - 1
        for precision in ("int8", "fp8"):
            at_limit = MetadataTensor((2, 32), strides=(limit - 31, 1))
            over_limit = MetadataTensor((2, 32), strides=(limit - 30, 1))
            self.assertEqual(self.block(at_limit, precision), 32)
            self.assertIsNone(self.block(over_limit, precision))

    def test_padded_output_offset_boundary(self):
        # Zero row stride isolates output indexing from the input offset guard.
        for precision in ("int8", "fp8"):
            for m, expected in ((131072, 16384), (131073, None)):
                tensor = MetadataTensor((m, 16384), strides=(0, 1))
                self.assertEqual(self.block(tensor, precision), expected)
            # The unpadded size fits int32, but rounding M to 16 rows exceeds it.
            self.assertLessEqual(174753 * 12288 - 1, (1 << 31) - 1)
            at_limit = MetadataTensor((174752, 12288), strides=(0, 1))
            over_limit = MetadataTensor((174753, 12288), strides=(0, 1))
            self.assertEqual(self.block(at_limit, precision), 16384)
            self.assertIsNone(self.block(over_limit, precision))


if __name__ == "__main__":
    unittest.main()
