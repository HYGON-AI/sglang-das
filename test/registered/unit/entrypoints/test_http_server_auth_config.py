"""Tests for the real multi-tokenizer HTTP auth configuration provider."""

import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints import http_server
from sglang.srt.utils.auth import AuthConfigNotReady
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

class TestMultiTokenizerAuthConfig(unittest.TestCase):
    def setUp(self):
        self.previous_state = http_server.get_global_state()
        self.addCleanup(http_server.set_global_state, self.previous_state)

        self.had_mode_marker = hasattr(http_server.app, "is_single_tokenizer_mode")
        self.previous_mode = getattr(
            http_server.app, "is_single_tokenizer_mode", None
        )
        self.addCleanup(self._restore_mode_marker)

    def _restore_mode_marker(self):
        if self.had_mode_marker:
            http_server.app.is_single_tokenizer_mode = self.previous_mode
        elif hasattr(http_server.app, "is_single_tokenizer_mode"):
            delattr(http_server.app, "is_single_tokenizer_mode")

    def _set_state(self, api_key=None, admin_api_key=None):
        server_args = SimpleNamespace(
            api_key=api_key, admin_api_key=admin_api_key
        )
        tokenizer_manager = SimpleNamespace(server_args=server_args)
        http_server.set_global_state(
            SimpleNamespace(tokenizer_manager=tokenizer_manager)
        )

    def test_missing_mode_marker_does_not_bypass_auth(self):
        if hasattr(http_server.app, "is_single_tokenizer_mode"):
            delattr(http_server.app, "is_single_tokenizer_mode")
        self._set_state(api_key="user", admin_api_key="admin")

        self.assertEqual(
            http_server._get_multi_tokenizer_auth_config(), ("user", "admin")
        )

    def test_uninitialized_state_raises_not_ready(self):
        http_server.app.is_single_tokenizer_mode = False
        http_server.set_global_state(None)

        with self.assertRaises(AuthConfigNotReady):
            http_server._get_multi_tokenizer_auth_config()

    def test_configured_keys_are_returned(self):
        http_server.app.is_single_tokenizer_mode = False
        self._set_state(api_key="user", admin_api_key="admin")

        self.assertEqual(
            http_server._get_multi_tokenizer_auth_config(), ("user", "admin")
        )

    def test_single_tokenizer_mode_bypasses_dynamic_layer(self):
        http_server.app.is_single_tokenizer_mode = True
        http_server.set_global_state(None)

        self.assertIsNone(http_server._get_multi_tokenizer_auth_config())


if __name__ == "__main__":
    unittest.main()
