"""
EBNF utilities for generating grammar rules that exclude specific strings.

This module provides functions to generate EBNF rules that match any text
except those containing specified substrings. Uses Aho-Corasick algorithm
for efficient multi-pattern matching.
"""

import hashlib
from collections import deque
from typing import Dict, List, Set


class _TrieNode:
    """Trie node with Aho-Corasick failure link."""

    def __init__(self, node_id: int):
        self.id = node_id
        self.children: Dict[str, _TrieNode] = {}
        self.is_end = False  # True if this node represents end of a pattern
        self.fail: _TrieNode = None  # Failure link for Aho-Corasick


def _build_trie_with_failure_links(
    patterns: List[str],
) -> tuple["_TrieNode", List["_TrieNode"]]:
    """Build Trie and compute Aho-Corasick failure links."""
    root = _TrieNode(0)
    all_nodes = [root]
    next_id = 1

    # Insert all patterns into Trie
    for pattern in patterns:
        node = root
        for char in pattern:
            if char not in node.children:
                new_node = _TrieNode(next_id)
                next_id += 1
                all_nodes.append(new_node)
                node.children[char] = new_node
            node = node.children[char]
        node.is_end = True

    # Build failure links using BFS
    root.fail = root
    queue = deque()

    # Initialize failure links for depth-1 nodes
    for child in root.children.values():
        child.fail = root
        queue.append(child)

    # BFS to build failure links for deeper nodes
    while queue:
        node = queue.popleft()
        for char, child in node.children.items():
            queue.append(child)
            # Find failure link for child
            fail_node = node.fail
            while fail_node != root and char not in fail_node.children:
                fail_node = fail_node.fail
            if char in fail_node.children and fail_node.children[char] != child:
                child.fail = fail_node.children[char]
            else:
                child.fail = root
            # Propagate is_end through failure links
            if child.fail.is_end:
                child.is_end = True

    return root, all_nodes


def _get_transition(node: "_TrieNode", char: str, root: "_TrieNode") -> "_TrieNode":
    """
    Get the next state after reading 'char' from 'node'.
    Follows failure links as in Aho-Corasick algorithm.
    """
    current = node
    while True:
        if char in current.children:
            return current.children[char]
        if current == root:
            return root
        current = current.fail


def _escape_char_class(s: str) -> str:
    """Escape special characters for use in EBNF character class [...]."""
    result = []
    for c in s:
        if c in r"\]^-":
            result.append("\\" + c)
        elif c == "\n":
            result.append("\\n")
        elif c == "\t":
            result.append("\\t")
        elif c == "\r":
            result.append("\\r")
        elif ord(c) < 32 or ord(c) > 126:
            result.append(f"\\x{ord(c):02X}")
        else:
            result.append(c)
    return "".join(result)


def _escape_string(c: str) -> str:
    """Escape a character for use in EBNF string literal "..."."""
    if c == '"':
        return '\\"'
    elif c == "\\":
        return "\\\\"
    elif c == "\n":
        return "\\n"
    elif c == "\t":
        return "\\t"
    elif c == "\r":
        return "\\r"
    elif ord(c) < 32 or ord(c) > 126:
        return f"\\x{ord(c):02X}"
    return c


def any_string_exclude(rule_name: str, negative_strings: List[str]) -> List[str]:
    """
    Generate EBNF rules that match any string except those containing
    any of the negative_strings as substrings.

    Uses Aho-Corasick algorithm to build a state machine that tracks
    multiple patterns simultaneously.

    Args:
        rule_name: The name of the root rule
        negative_strings: List of strings to exclude as substrings

    Returns:
        List of EBNF rule strings, each element is one rule

    Example:
        >>> any_string_exclude("xml_text", ["</arg_value>"])
        ['xml_text ::= s_xxxxxxxx_0*',
         's_xxxxxxxx_0 ::= [^<] | "<" s_xxxxxxxx_1',
         's_xxxxxxxx_1 ::= [^/<] | "<" s_xxxxxxxx_1 | "/" s_xxxxxxxx_2',
         ...]
    """
    # Handle empty input
    if not negative_strings:
        return [f"{rule_name} ::= [^]*"]

    # Sort and deduplicate for deterministic hashing
    sorted_strings = sorted(set(s for s in negative_strings if s))
    if not sorted_strings:
        return [f"{rule_name} ::= [^]*"]

    # Compute hash prefix for rule names
    hash_input = "\x00".join(sorted_strings)
    hash_prefix = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]

    # Build Trie with Aho-Corasick failure links
    root, all_nodes = _build_trie_with_failure_links(sorted_strings)

    # Collect all pattern characters
    all_pattern_chars: Set[str] = set()
    for pattern in sorted_strings:
        all_pattern_chars.update(pattern)

    # Generate state name
    def state_name(node: "_TrieNode") -> str:
        return f"s_{hash_prefix}_{node.id}"

    # Generate EBNF rules
    # Use recursive structure: each state can transition to another state or terminate
    # root ::= s_0
    # s_N ::= [^excluded] s_target | "c" s_target | "" (for termination)
    rules = []
    rules.append(f"{rule_name} ::= {state_name(root)}")

    for node in all_nodes:
        if node.is_end:
            # End states should never be reached, no rule needed
            continue

        excluded_chars: List[str] = []
        # Group transitions by target state: {target_id: [chars]}
        transitions_by_target: Dict[int, List[str]] = {}

        for char in all_pattern_chars:
            target = _get_transition(node, char, root)
            if target.is_end:
                # This transition would complete a pattern, exclude this char
                excluded_chars.append(char)
            else:
                # Record transition to target (including root)
                if target.id not in transitions_by_target:
                    transitions_by_target[target.id] = []
                transitions_by_target[target.id].append(char)

        # Build alternatives
        alternatives = []

        # All chars that need explicit handling (either excluded or have explicit target)
        all_explicit_chars = set(excluded_chars)
        for chars in transitions_by_target.values():
            all_explicit_chars.update(chars)

        # Character class for chars not in any pattern -> transition to root state
        if all_explicit_chars:
            escaped = _escape_char_class("".join(sorted(all_explicit_chars)))
            alternatives.append(f"[^{escaped}] {state_name(root)}")
        else:
            # No characters to exclude, match any character -> stay at root
            alternatives.append(f"[^] {state_name(root)}")

        # Explicit transitions to specific states
        for target_id in sorted(transitions_by_target.keys()):
            chars = transitions_by_target[target_id]
            target_node = next(n for n in all_nodes if n.id == target_id)
            for char in sorted(chars):
                alternatives.append(
                    f'"{_escape_string(char)}" {state_name(target_node)}'
                )

        # Add empty alternative to allow termination at any non-end state
        # This is safe because we use recursive structure that maintains state
        alternatives.append('""')

        rules.append(f"{state_name(node)} ::= {' | '.join(alternatives)}")

    return rules


if __name__ == "__main__":
    # Demo and test
    print("=== Single string exclusion: </arg_value> ===")
    grammar_rules = any_string_exclude("xml_text", ["</arg_value>"])
    grammar = "\n".join(grammar_rules)
    print(grammar)
    print()

    print("=== Multiple string exclusion: abc, xyz ===")
    grammar_rules2 = any_string_exclude("text", ["abc", "xyz"])
    print("\n".join(grammar_rules2))
    print()

    # Test with xgrammar if available
    try:
        import xgrammar as xgr

        print("=== Testing with xgrammar ===")
        tokenizer_info = xgr.TokenizerInfo([])
        grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="xml_text")

        test_cases = [
            ("</arg_value>", False),
            ("</arg_value", True),
            ("hello world", True),
            ("<New York>", True),
            ("hello</arg_value>world", False),
            ("</a</arg_value>", False),
            ("<<<</arg_value>", False),
            ("", True),
        ]

        for test_string, expected in test_cases:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            accepted = True
            for c in test_string:
                if not matcher.accept_string(c):
                    accepted = False
                    break

            status = "✓" if accepted == expected else "✗"
            print(f"{status} {test_string!r}: accepted={accepted}, expected={expected}")

    except ImportError:
        print("xgrammar not installed, skipping integration test")
