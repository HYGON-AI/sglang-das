from dataclasses import dataclass
from typing import Literal, Optional, Protocol, runtime_checkable

from .ebnf_utils import any_string_exclude
from .tool_schema import build_tool_call_rules


@runtime_checkable
class FunctionProtocol(Protocol):
    name: str
    parameters: Optional[object]


@dataclass
class SpecialTokenConfig:
    begin_of_thinking: str = "<think>"
    end_of_thinking: str = "</think>"
    begin_of_tool_call: str = "<tool_call>"
    end_of_tool_call: str = "</tool_call>"
    begin_of_key: str = "<arg_key>"
    end_of_key: str = "</arg_key>"
    begin_of_value: str = "<arg_value>"
    end_of_value: str = "</arg_value>"
    assistant_token: str = "<|assistant|>"

    def all_special_tokens(self) -> list[str]:
        return vars(self).values()


def get_special_token_config(tokenizer) -> SpecialTokenConfig:
    from .escape import get_global_escaped_special_tokens

    sp = get_global_escaped_special_tokens()
    return SpecialTokenConfig(
        begin_of_thinking=sp.get("<think>"),
        end_of_thinking=sp.get("</think>"),
        begin_of_tool_call=sp.get("<tool_call>"),
        end_of_tool_call=sp.get("</tool_call>"),
        begin_of_key=sp.get("<arg_key>"),
        end_of_key=sp.get("</arg_key>"),
        begin_of_value=sp.get("<arg_value>"),
        end_of_value=sp.get("</arg_value>"),
        assistant_token=sp.get("<|assistant|>"),
    )


def generation_constraint(
    enable_thinking: bool,
    functions: list[FunctionProtocol] | None,
    special_tokens: SpecialTokenConfig,
    chat_template_version: Literal["glm45", "glm47"],
    accommodate_chat_template: bool,
    allow_multiple_assistant_turns: bool,
    root_name: str = "root",
) -> str:
    """
    Args:
        accommodate_chat_template: bool
            Whether to skip the content that is already included in the chat template
            Expected glm45 chat template generation prompt:
            ```jinja
            {%- if add_generation_prompt -%}
                <|assistant|>{{- '\n<think></think>' if (enable_thinking is defined and not enable_thinking) else '' -}}
            {%- endif -%}
            ```

            Expected glm47 chat template generation prompt:
            ```jinja
            {%- if add_generation_prompt -%}
                <|assistant|>{{- '</think>' if (enable_thinking is defined and not enable_thinking) else '<think>' -}}
            {%- endif -%}
            ```
    """
    ebnf_lines = [
        f'{root_name} ::= assistant_turn ( "{special_tokens.assistant_token}" assistant_turn )*'
        if allow_multiple_assistant_turns
        else f"{root_name} ::= assistant_turn",
        "assistant_turn ::= thinking_block text_block tool_call_blocks",
    ]

    thinking_exclusions = [
        special_tokens.begin_of_tool_call,
        special_tokens.end_of_tool_call,
        special_tokens.begin_of_key,
        special_tokens.end_of_key,
        special_tokens.begin_of_value,
        special_tokens.end_of_value,
        special_tokens.end_of_thinking,  # eliminates ambiguity
    ]

    # Extra spaces
    if chat_template_version == "glm45":
        extra_seperator = '"\\n"'
    elif chat_template_version == "glm47":
        extra_seperator = ""
    else:
        raise NotImplementedError(
            f"Unsupported chat_template_version: {chat_template_version}"
        )

    # Thinking block
    if chat_template_version == "glm45":
        if enable_thinking:
            ebnf_lines.append(
                rf'thinking_block ::= "\n{special_tokens.begin_of_thinking}" thinking_block_content "{special_tokens.end_of_thinking}"'
            )

            ebnf_lines.extend(
                any_string_exclude("thinking_block_content", thinking_exclusions)
            )
        else:
            if accommodate_chat_template:
                #
                ebnf_lines.append('thinking_block ::= ""')
            else:
                ebnf_lines.append(
                    rf'thinking_block ::= "\n{special_tokens.begin_of_thinking}" "{special_tokens.end_of_thinking}"'
                )
    elif chat_template_version == "glm47":
        if enable_thinking:
            if accommodate_chat_template:
                ebnf_lines.append(
                    rf'thinking_block ::= thinking_block_content "{special_tokens.end_of_thinking}"'
                )
            else:
                ebnf_lines.append(
                    rf'thinking_block ::= "{special_tokens.begin_of_thinking}" thinking_block_content "{special_tokens.end_of_thinking}"'
                )
            ebnf_lines.extend(
                any_string_exclude("thinking_block_content", thinking_exclusions)
            )
        else:
            if accommodate_chat_template:
                ebnf_lines.append('thinking_block ::= ""')
            else:
                ebnf_lines.append(
                    rf'thinking_block ::= "{special_tokens.end_of_thinking}"'
                )
    else:
        raise NotImplementedError(
            f"Unsupported chat_template_version: {chat_template_version}"
        )

    # Text block
    ebnf_lines.extend(
        any_string_exclude(
            "text_without_special_tokens", special_tokens.all_special_tokens()
        )
    )

    ebnf_lines.append(
        f"text_block ::= ( {extra_seperator} text_without_special_tokens )?"
    )

    # Tool call blocks
    if functions:
        ebnf_lines.extend(
            build_tool_call_rules(
                non_terminal_name="tool_call_blocks",
                functions=functions,
                special_tokens=special_tokens,
                chat_template_version=chat_template_version,
            )
        )
    else:
        ebnf_lines.append('tool_call_blocks ::= ""')

    # deduplicate identical lines; raise on conflicting definitions
    # TODO: why? is tool call uniquely identified by tool call hash? how is too call hash produced
    non_terminals = {}  # lhs -> full line
    deduped_lines = []
    for line in ebnf_lines:
        assert "\n" not in line, "Each EBNF rule should be in a single line."
        lhs = line.split("::=")[0].strip()
        if lhs in non_terminals:
            if non_terminals[lhs] == line:
                continue  # skip identical duplicate
            raise ValueError(f"Duplicate non-terminal found: {lhs}")
        non_terminals[lhs] = line
        deduped_lines.append(line)

    return "\n".join(deduped_lines)


if __name__ == "__main__":
    special_tokens = SpecialTokenConfig()
    ebnf_grammar = generation_constraint(
        enable_thinking=True,
        functions=None,
        special_tokens=special_tokens,
        root_name="root",
        chat_template_version="glm47",
        accommodate_chat_template=False,
        allow_multiple_assistant_turns=False,
    )
    print(ebnf_grammar)
