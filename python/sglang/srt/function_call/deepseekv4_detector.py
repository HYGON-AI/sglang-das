import logging

from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector

logger = logging.getLogger(__name__)


class DeepSeekV4Detector(DeepSeekV32Detector):
    """
    Detector for DeepSeek V4 model function call format.

    The DeepSeek V4 format uses XML-like DSML tags to delimit function calls.
    Supports two parameter formats:

    Format 1 - XML Parameter Tags:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="function_name">
        <｜DSML｜parameter name="param_name" string="true">value</｜DSML｜parameter>
        ...
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Format 2 - Direct JSON:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="function_name">
        {
            "param_name": "value"
        }
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Examples:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        { "city": "San Francisco" }
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜DSML｜tool_calls>` and `</｜DSML｜tool_calls>`
    - Individual Tool Call: Wrapped between `<｜DSML｜invoke name="...">` and `</｜DSML｜invoke>`
    - Parameters: Either XML tags or direct JSON format
    - Supports multiple tool calls

    Reference: DeepSeek V4 format specification
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜DSML｜tool_calls>"
        self.eot_token = "</｜DSML｜tool_calls>"
        self.function_calls_regex = r"<｜DSML｜tool_calls>(.*?)</｜DSML｜tool_calls>"

    def supports_structural_tag_for_tool_choice(self, tool_choice) -> bool:
        """Avoid native required/named tags that can stall xgrammar under PD.

        Automatic strict tool choice can still use the native DeepSeek-V4 tag.
        Required and named choices are parsed from the model's native DSML
        output instead of compiling any grammar on the prefill worker.
        """
        return tool_choice == "auto"

    def parses_required_natively(self) -> bool:
        """Parse required/named calls as native DSML without an xgrammar rule.

        Both DeepSeek-V4 structural tags and JSON-schema constraints can block
        xgrammar compilation on the rank-0 prefill scheduler in PD mode.  The
        model already emits DSML tool calls, and this detector handles repeated
        invokes plus the model's malformed parameter closers, so constraining
        required/named requests is unnecessary and unsafe here.
        """
        return True

    def get_structural_tag_name(self) -> str:
        return "deepseek_v4"
