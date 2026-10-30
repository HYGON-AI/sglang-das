import ast
import json
import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.ebnf_composer import EBNFComposer

logger = logging.getLogger(__name__)


def get_argument_type(func_name: str, arg_key: str, defined_tools: list):
    name2tool = {tool.function.name: tool for tool in defined_tools}
    if func_name not in name2tool:
        return None
    tool = name2tool[func_name]
    properties = (tool.function.parameters or {}).get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    if arg_key not in properties:
        return None
    return infer_type_from_json_schema(properties[arg_key])


def get_composite_arg_type(prop):
    parent_type = prop.get("type", None)
    if parent_type:
        return parent_type

    # GLM NOTE: Support for anyOf/oneOf/not and recursive schema is omitted for now.
    # This implementation assumes a non-nested JSON schema combination using
    # `allOf` structure and type definitions are not deeply buried.
    sub_schemas = prop.get("allOf")
    if not isinstance(sub_schemas, list):
        return None

    for schema in sub_schemas:
        if not isinstance(schema, dict):
            continue

        t = schema.get("type")
        if isinstance(t, list):
            actual_type = next(
                (item for item in t if isinstance(item, str) and item), None
            )
            if actual_type:
                return actual_type
        elif isinstance(t, str) and t:
            return t

    return None


def infer_type_from_json_schema(schema: Dict[str, Any]) -> Optional[str]:
    """
    Infer the primary type of a parameter from JSON Schema.

    Supports complex JSON Schema structures including:
    - Direct type field (including type arrays)
    - anyOf/oneOf: parameter can be any of multiple types
    - enum: parameter must be one of enum values
    - allOf: parameter must satisfy all type definitions
    - properties: inferred as object type
    - items: inferred as array type

    Args:
        schema: JSON Schema definition

    Returns:
        Inferred type ('string', 'number', 'object', 'array', etc.) or None
    """
    if not isinstance(schema, dict):
        return None

    # Priority 1: Direct type field (including type arrays)
    if "type" in schema:
        type_value = schema["type"]
        if isinstance(type_value, str):
            return type_value
        elif isinstance(type_value, list) and type_value:
            # Handle type arrays: return first non-null type
            non_null_types = [t for t in type_value if t != "null"]
            if non_null_types:
                return non_null_types[0]
            return "string"  # If only null, default to string

    # Priority 2: Handle anyOf/oneOf
    if "anyOf" in schema or "oneOf" in schema:
        schemas = schema.get("anyOf") or schema.get("oneOf")
        types = []

        if isinstance(schemas, list):
            for sub_schema in schemas:
                inferred_type = infer_type_from_json_schema(sub_schema)
                if inferred_type:
                    types.append(inferred_type)

            if types:
                # If all types are the same, return unified type
                if len(set(types)) == 1:
                    return types[0]
                # When types differ, prioritize string (safest)
                if "string" in types:
                    return "string"
                # Otherwise return first type
                return types[0]

    # Priority 3: Handle enum (infer type from enum values)
    if "enum" in schema and isinstance(schema["enum"], list):
        if not schema["enum"]:
            return "string"

        # Infer type from enum values
        enum_types = set()
        for value in schema["enum"]:
            if value is None:
                enum_types.add("null")
            elif isinstance(value, bool):
                enum_types.add("boolean")
            elif isinstance(value, int):
                enum_types.add("integer")
            elif isinstance(value, float):
                enum_types.add("number")
            elif isinstance(value, str):
                enum_types.add("string")
            elif isinstance(value, list):
                enum_types.add("array")
            elif isinstance(value, dict):
                enum_types.add("object")

        # If type is uniform, return that type
        if len(enum_types) == 1:
            return enum_types.pop()
        # Mixed types, prioritize string
        return "string"

    # Priority 4: Handle allOf (must satisfy all types)
    if "allOf" in schema and isinstance(schema["allOf"], list):
        schemas = schema["allOf"]
        for sub_schema in schemas:
            inferred_type = infer_type_from_json_schema(sub_schema)
            if inferred_type and inferred_type != "string":
                return inferred_type
        return "string"

    # Priority 5: Infer object type
    if "properties" in schema:
        return "object"

    # Priority 6: Infer array type
    if "items" in schema:
        return "array"

    return None


def parse_arguments(json_value):
    try:
        parsed_value = json.loads(json_value)
        return parsed_value, True
    except:
        # If that fails, try wrapping it to unescape JSON characters
        try:
            # Wrap the value as a JSON string field
            wrapped = json.loads('{"tmp": "' + json_value + '"}')
            # parse the unescaped value
            parsed_value = json.loads(wrapped["tmp"])
            return parsed_value, True
        except:
            # Final fallback to ast.literal_eval
            try:
                parsed_value = ast.literal_eval(json_value)
                return parsed_value, True
            except:
                return json_value, False


class Glm5MoeDetector(BaseFormatDetector):
    """
    Detector for GLM-4.7 and GLM-5 models.
    Assumes function call format:
      <tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value><arg_key>date</arg_key><arg_value>2024-06-27</arg_value></tool_call><tool_call>get_weather<arg_key>city</arg_key><arg_value>上海</arg_value><arg_key>date</arg_key><arg_value>2024-06-27</arg_value></tool_call>
    """

    def __init__(self):
        super().__init__()
        from sglang.srt.constrained.glm.escape import get_global_escaped_special_tokens

        sp = get_global_escaped_special_tokens()
        self.bot_token = sp.get("<tool_call>")
        self.eot_token = sp.get("</tool_call>")
        key_open = sp.get("<arg_key>")
        key_close = sp.get("</arg_key>")
        val_open = sp.get("<arg_value>")
        val_close = sp.get("</arg_value>")
        bot_re = re.escape(self.bot_token)
        eot_re = re.escape(self.eot_token)
        key_open_re = re.escape(key_open)
        key_close_re = re.escape(key_close)
        val_open_re = re.escape(val_open)
        val_close_re = re.escape(val_close)
        self.func_call_regex = rf"{bot_re}.*?{eot_re}"
        self.func_detail_regex = re.compile(
            rf"{bot_re}(.*?)({key_open_re}.*?)?{eot_re}", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            rf"{key_open_re}(.*?){key_close_re}(?:\\n|\s)*{val_open_re}(.*?){val_close_re}",
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a glm-4.5 / glm-4.6 format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx] if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])
        match_result_list = re.findall(self.func_call_regex, text, re.DOTALL)
        calls = []
        try:
            for match_result in match_result_list:
                # Get function name
                func_detail = self.func_detail_regex.search(match_result)
                func_name = func_detail.group(1)
                arguments = {}
                func_args = func_detail.group(2)
                if func_args:
                    pairs = self.func_arg_regex.findall(func_args)
                    for arg_key, arg_value in pairs:
                        arg_type = get_argument_type(func_name, arg_key, tools)
                        if arg_type != "string":
                            arg_value, is_good_json = parse_arguments(arg_value)
                        arguments[arg_key] = arg_value
                # construct match_result for parse_base_json
                match_result = {"name": func_name, "parameters": arguments}
                calls.extend(self.parse_base_json(match_result, tools))
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for GLM-4.5 and GLM-4.6 format.
        """
        self._buffer += new_text
        current_text = self._buffer

        start = current_text.find(self.bot_token)
        if start == -1:
            self._buffer = ""
            if self.current_tool_id > 0:
                current_text = ""
            return StreamingParseResult(normal_text=current_text)
        # find ensures we find the first self.eot_token so there will be at most one tool_call in current_text[:end+len(self.eot_token)
        end = current_text.find(self.eot_token)
        if end != -1:
            # Initialize state if this is the first tool call
            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.prev_tool_call_arr = []
                self.streamed_args_for_tool = [""]
            # Ensure we have enough entries in our tracking arrays
            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")
            result = self.detect_and_parse(
                current_text[: end + len(self.eot_token)], tools=tools
            )
            if result.calls:
                arguments = result.calls[0].parameters or "{}"
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": result.calls[0].name,
                    "arguments": json.loads(arguments),
                }
                self.streamed_args_for_tool[self.current_tool_id] = result.calls[
                    0
                ].parameters
                result.calls[0].tool_index = self.current_tool_id
                self.current_tool_id += 1
            self._buffer = current_text[end + len(self.eot_token) :]
            return result
        normal_text = current_text[:start]
        self._buffer = current_text[start:]
        return StreamingParseResult(normal_text=normal_text)

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()

    def build_ebnf(self, tools: List[Tool]):
        from sglang.srt.constrained.glm.escape import get_global_escaped_special_tokens

        sp = get_global_escaped_special_tokens()
        key_open = sp.get("<arg_key>")
        key_close = sp.get("</arg_key>")
        val_open = sp.get("<arg_value>")
        val_close = sp.get("</arg_value>")
        seen_names, filtered_and_unique_tools = set(), []
        valid_name_pattern = re.compile(r"[^\"\s]+")
        for tool in tools:
            tool_name = tool.function.name
            if valid_name_pattern.fullmatch(tool_name) and tool_name not in seen_names:
                filtered_and_unique_tools.append(tool)
                seen_names.add(tool_name)

        return EBNFComposer.build_ebnf(
            filtered_and_unique_tools,
            individual_call_start_token=self.bot_token,
            individual_call_end_token=self.eot_token,
            tool_call_separator="\\n",
            function_format="xml",
            call_rule_fmt='"{name}" ( {arguments_rule} )',
            key_value_rule_fmt=(
                f'"{key_open}{{key}}{key_close}{val_open}" {{valrule}} "{val_close}"'
            ),
            key_value_separator="",
        )


class Glm5MoeToolParsingState(Enum):
    WAITING_BOT = 0
    PARSING_FUNCTION_NAME = 1
    PARSING_ARG_KEY = 2
    PARSED_ARG_KEY = 3
    PARSING_ARG_VAL = 4
    PARSED_ARG_VAL = 5


class Glm5MoeStreamDetector(Glm5MoeDetector):
    def __init__(self):
        super().__init__()
        from sglang.srt.constrained.glm.escape import get_global_escaped_special_tokens

        sp = get_global_escaped_special_tokens()
        self.begin_of_arg_key_token = sp.get("<arg_key>")
        self.end_of_arg_key_token = sp.get("</arg_key>")
        self.begin_of_arg_value_token = sp.get("<arg_value>")
        self.end_of_arg_value_token = sp.get("</arg_value>")
        self.tool_call_separator = "\n"

        self._state = Glm5MoeToolParsingState.WAITING_BOT

        self.current_function_name = ""
        self.current_arg_key = ""
        self.current_arg_type = None

    def _get_next_tool_id(self):
        self.current_tool_id += 1
        return self.current_tool_id

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for GLM-4.5 format.
        """

        normal_text, calls = "", []
        self._buffer += new_text

        # recursively parsing
        while len(self._buffer) > 0:
            # Case 1: waiting for bot_token
            if self._state == Glm5MoeToolParsingState.WAITING_BOT:
                start = self._buffer.find(self.bot_token)
                if start == -1:
                    normal_text += self._buffer
                    self._buffer = ""
                    break
                else:
                    normal_text = self._buffer[:start]
                    self._buffer = self._buffer[start + len(self.bot_token) :]
                    self._state = Glm5MoeToolParsingState.PARSING_FUNCTION_NAME
            # Case 2: parsing function name
            elif self._state == Glm5MoeToolParsingState.PARSING_FUNCTION_NAME:
                start = self._buffer.find(self.begin_of_arg_key_token)
                if start == -1:
                    start = self._buffer.find(self.eot_token)
                    if start == -1:
                        break
                    else:
                        function_name = self._buffer[:start]
                        self._buffer = self._buffer[start + len(self.eot_token) :]
                        calls.append(
                            ToolCallItem(
                                tool_index=self._get_next_tool_id(),
                                name=function_name,
                                parameters="{}",
                            )
                        )
                        self._state = Glm5MoeToolParsingState.WAITING_BOT
                else:
                    function_name = self._buffer[:start]
                    self._buffer = self._buffer[
                        start + len(self.begin_of_arg_key_token) :
                    ]
                    self.current_function_name = function_name
                    calls.append(
                        ToolCallItem(
                            tool_index=self._get_next_tool_id(),
                            name=function_name,
                            parameters='{"',
                        )
                    )
                    self._state = Glm5MoeToolParsingState.PARSING_ARG_KEY
                    self.current_arg_key = ""
            # Case 2: parsing arg key
            elif self._state == Glm5MoeToolParsingState.PARSING_ARG_KEY:
                start = self._buffer.find(self.end_of_arg_key_token)
                if start == -1:
                    partial_arg_key = self._buffer
                    self._buffer = ""
                    self.current_arg_key += partial_arg_key
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            parameters=json.dumps(partial_arg_key, ensure_ascii=False)[
                                1:-1
                            ],
                        )
                    )
                    break
                else:
                    partial_arg_key = self._buffer[:start]
                    self._buffer = self._buffer[
                        start + len(self.end_of_arg_key_token) :
                    ]
                    self.current_arg_key += partial_arg_key
                    self.current_arg_type = (
                        get_argument_type(
                            self.current_function_name, self.current_arg_key, tools
                        )
                        or "string"
                    )
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            parameters=(
                                json.dumps(partial_arg_key, ensure_ascii=False)[1:]
                                + ":"
                                + ('"' if self.current_arg_type == "string" else "")
                            ),
                        )
                    )
                    self._state = Glm5MoeToolParsingState.PARSED_ARG_KEY
            elif self._state == Glm5MoeToolParsingState.PARSED_ARG_KEY:
                start = self._buffer.find(self.begin_of_arg_value_token)
                if start == -1:
                    self._buffer = ""
                    break
                else:
                    self._buffer = self._buffer[
                        start + len(self.begin_of_arg_value_token) :
                    ]
                    self._state = Glm5MoeToolParsingState.PARSING_ARG_VAL
            # Case 3: parsing arg value
            elif self._state == Glm5MoeToolParsingState.PARSING_ARG_VAL:
                start = self._buffer.find(self.end_of_arg_value_token)
                if start == -1:
                    partial_arg_value = self._buffer
                    self._buffer = ""
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            parameters=(
                                json.dumps(partial_arg_value, ensure_ascii=False)[1:-1]
                                if self.current_arg_type == "string"
                                else partial_arg_value
                            ),
                        )
                    )
                    break
                else:
                    partial_arg_value = self._buffer[:start]
                    self._buffer = self._buffer[
                        start + len(self.end_of_arg_value_token) :
                    ]
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            parameters=(
                                json.dumps(partial_arg_value, ensure_ascii=False)[1:]
                                if self.current_arg_type == "string"
                                else partial_arg_value
                            ),
                        )
                    )
                    self._state = Glm5MoeToolParsingState.PARSED_ARG_VAL
            elif self._state == Glm5MoeToolParsingState.PARSED_ARG_VAL:
                start = self._buffer.find(self.begin_of_arg_key_token)
                if start == -1:
                    start = self._buffer.find(self.eot_token)
                    if start == -1:
                        self._buffer = ""
                        break
                    else:
                        self._buffer = self._buffer[start + len(self.eot_token) :]
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                parameters="}",
                            )
                        )
                        self._state = Glm5MoeToolParsingState.WAITING_BOT
                else:
                    self._buffer = self._buffer[
                        start + len(self.begin_of_arg_key_token) :
                    ]
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            parameters=',"',
                        )
                    )
                    self._state = Glm5MoeToolParsingState.PARSING_ARG_KEY
                    self.current_arg_key = ""
            else:  # should not reach here
                break

        return StreamingParseResult(normal_text=normal_text, calls=calls)
