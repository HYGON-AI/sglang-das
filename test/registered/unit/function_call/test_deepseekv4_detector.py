"""Unit tests for DeepSeekV4Detector DSML streaming — no server, no model loading."""

import json
from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import Function, Tool, ToolChoice
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

DSML = "｜DSML｜"


def _wrapped(invoke: str) -> str:
    return f"<{DSML}tool_calls>\n{invoke}\n</{DSML}tool_calls>"


def _invoke(name: str, params: str = "") -> str:
    return f'<{DSML}invoke name="{name}">\n{params}\n</{DSML}invoke>'


def _param(name: str, is_string: str, value: str) -> str:
    return (
        f'<{DSML}parameter name="{name}" string="{is_string}">{value}</{DSML}parameter>'
    )


def _weather_call(city: str = "SF") -> str:
    return _wrapped(_invoke("get_weather", _param("city", "true", city)))


class TestDeepSeekV4Streaming(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            )
        ]

    def _feed(self, chunks):
        """Returns (normal_text, calls) accumulated over the chunks."""
        detector = DeepSeekV4Detector()
        normal, calls = "", []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal += result.normal_text
            calls.extend(result.calls)
        return normal, calls

    def test_preamble_in_same_delta_as_tool_call(self):
        """Prose sharing a delta with the tool call must not be dropped, and the
        streaming and one-shot paths must agree on it."""
        text = "Let me check.\n" + _weather_call()
        normal, calls = self._feed([text])

        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])
        self.assertEqual(
            normal, DeepSeekV4Detector().detect_and_parse(text, self.tools).normal_text
        )

    def test_preamble_before_bare_invoke_without_wrapper(self):
        """The bare `<｜DSML｜invoke …>` form has no tool_calls wrapper to walk
        back to, so the preamble is computed from the invoke itself."""
        text = "Checking.\n" + _invoke("get_weather", _param("city", "true", "SF"))
        normal, calls = self._feed([text])

        self.assertIn("Checking.", normal)
        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])

    def test_no_dsml_markers_leak_into_normal_text(self):
        text = "Prose.\n" + _weather_call()
        normal, _ = self._feed([text[i : i + 4] for i in range(0, len(text), 4)])

        self.assertNotIn(DSML, normal)

    def test_malformed_partial_json_falls_back_to_raw_value(self):
        """A partial non-string parameter must not escape as MalformedJSON."""
        detector = DeepSeekV4Detector()
        result = detector.parse_streaming_increment(
            f'<{DSML}tool_calls>\n<{DSML}invoke name="get_weather">\n'
            f'<{DSML}parameter name="city" string="false">{{"a"',
            self.tools,
        )

        self.assertEqual([c.name for c in result.calls if c.name], ["get_weather"])

    def test_non_streaming_parses_every_tool_calls_section(self):
        """A turn with two tool_calls sections must yield both calls."""
        result = DeepSeekV4Detector().detect_and_parse(
            f"{_weather_call('SF')}\n{_weather_call('NY')}", self.tools
        )

        self.assertEqual(len(result.calls), 2)

    def test_malformed_parameter_closers_are_recovered(self):
        """Known close-tag corruptions must not turn valid arguments into `{}`."""
        for closer in (
            f"</{DSML}parameterparameter>",
            f"</{DSML}parameter_param>",
            f'</{DSML}parameter string="true">',
        ):
            with self.subTest(closer=closer):
                malformed = _weather_call().replace(f"</{DSML}parameter>", closer)

                result = DeepSeekV4Detector().detect_and_parse(malformed, self.tools)

                self.assertEqual(len(result.calls), 1)
                self.assertEqual(
                    json.loads(result.calls[0].parameters), {"city": "SF"}
                )

    def test_malformed_parameter_closers_streaming_are_valid_json(self):
        for closer in (
            f"</{DSML}parameterparameter>",
            f"</{DSML}parameter_param>",
            f'</{DSML}parameter string="true">',
        ):
            with self.subTest(closer=closer):
                malformed = _weather_call().replace(f"</{DSML}parameter>", closer)

                _, calls = self._feed(
                    [malformed[i : i + 3] for i in range(0, len(malformed), 3)]
                )
                arguments = "".join(
                    call.parameters for call in calls if call.parameters
                )

                self.assertEqual(json.loads(arguments), {"city": "SF"})

    def test_wrapped_arguments_are_unwrapped_from_direct_json(self):
        text = _wrapped(
            _invoke("get_weather", '{"arguments":{"city":"SF"}}')
        )

        result = DeepSeekV4Detector().detect_and_parse(text, self.tools)

        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "SF"})

    def test_wrapped_arguments_are_unwrapped_from_xml(self):
        text = _wrapped(
            _invoke(
                "get_weather",
                _param("arguments", "true", '{"city":"SF"}'),
            )
        )

        result = DeepSeekV4Detector().detect_and_parse(text, self.tools)

        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "SF"})

    def test_scalar_is_wrapped_for_schema_declared_array(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    parameters={
                        "type": "object",
                        "properties": {
                            "queries": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["queries"],
                    },
                ),
            )
        ]
        text = _wrapped(_invoke("search", '{"queries":"SGLang"}'))

        result = DeepSeekV4Detector().detect_and_parse(text, tools)

        self.assertEqual(
            json.loads(result.calls[0].parameters), {"queries": ["SGLang"]}
        )

    def test_wrapped_scalar_maps_to_sole_array_property(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    parameters={
                        "type": "object",
                        "properties": {"queries": {"type": "array"}},
                        "required": ["queries"],
                    },
                ),
            )
        ]
        text = _wrapped(_invoke("search", '{"arguments":"SGLang"}'))

        result = DeepSeekV4Detector().detect_and_parse(text, tools)

        self.assertEqual(
            json.loads(result.calls[0].parameters), {"queries": ["SGLang"]}
        )

    def test_json_encoded_array_string_is_decoded(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    parameters={
                        "type": "object",
                        "properties": {"queries": {"type": "array"}},
                    },
                ),
            )
        ]
        text = _wrapped(
            _invoke("search", _param("queries", "true", '["SGLang", "ROCm"]'))
        )

        result = DeepSeekV4Detector().detect_and_parse(text, tools)

        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"queries": ["SGLang", "ROCm"]},
        )

    def test_array_repair_is_identical_in_streaming(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    parameters={
                        "type": "object",
                        "properties": {"queries": {"type": "array"}},
                    },
                ),
            )
        ]
        text = _wrapped(_invoke("search", '{"queries":"SGLang"}'))
        detector = DeepSeekV4Detector()
        calls = []
        for start in range(0, len(text), 4):
            calls.extend(
                detector.parse_streaming_increment(text[start : start + 4], tools).calls
            )
        arguments = "".join(call.parameters for call in calls if call.parameters)

        self.assertEqual(json.loads(arguments), {"queries": ["SGLang"]})

    def test_object_is_not_wrapped_as_array_without_item_proof(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    parameters={
                        "type": "object",
                        "properties": {"queries": {"type": "array"}},
                    },
                ),
            )
        ]
        text = _wrapped(_invoke("search", '{"queries":{"query":"SGLang"}}'))

        result = DeepSeekV4Detector().detect_and_parse(text, tools)

        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"queries": {"query": "SGLang"}},
        )

    def test_parse_error_neither_swallows_nor_duplicates(self):
        """An unexpected parse error must not empty the turn, and the dropped
        buffer must not come back on the next delta."""
        detector = DeepSeekV4Detector()

        with patch.object(
            DeepSeekV4Detector,
            "_parse_parameters_from_xml",
            side_effect=RuntimeError("boom"),
        ):
            first = detector.parse_streaming_increment(_weather_call(), self.tools)
            self.assertEqual(detector._buffer, "")
            second = detector.parse_streaming_increment(" tail", self.tools)

        self.assertIn("get_weather", first.normal_text)
        self.assertNotIn("get_weather", second.normal_text)
        # No half-formed call: the failure can land between a tool's name and its
        # arguments, so an argument-less named call must not reach the client.
        self.assertEqual(first.calls, [])

    def test_required_is_parsed_natively_without_any_grammar(self):
        """PD must not compile a grammar for DeepSeek-V4 required calls."""
        parser = FunctionCallParser(self.tools, "deepseekv4")

        self.assertTrue(
            parser.detector.supports_structural_tag_for_tool_choice("auto")
        )
        self.assertFalse(
            parser.detector.supports_structural_tag_for_tool_choice("required")
        )
        self.assertTrue(parser.detector.parses_required_natively())

        self.assertIsNone(
            parser.get_structure_constraint("required", parallel_tool_calls=False)
        )

    def test_named_parallel_choice_is_parsed_natively_without_any_grammar(self):
        """Native DSML may contain repeated invokes of the selected function."""
        parser = FunctionCallParser(self.tools, "deepseekv4")
        choice = ToolChoice(function={"name": "get_weather"})

        self.assertIsNone(
            parser.get_structure_constraint(choice, parallel_tool_calls=True)
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
