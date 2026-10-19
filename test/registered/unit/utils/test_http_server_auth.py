"""
Unit tests for HTTP server admin auth.

Usage:
    python3 -m pytest test/registered/unit/utils/test_http_server_auth.py -v
"""

import unittest
from contextlib import asynccontextmanager

from sglang.srt.utils.auth import (
    AuthConfigNotReady,
    AuthLevel,
    add_api_key_middleware,
    auth_level,
    decide_request_auth,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class TestHttpServerAdminAuth(unittest.TestCase):
    def _decide(
        self,
        *,
        method: str,
        path: str,
        authorization_header: str | None,
        api_key: str | None,
        admin_api_key: str | None,
        auth_level: AuthLevel,
    ):
        return decide_request_auth(
            method=method,
            path=path,
            authorization_header=authorization_header,
            api_key=api_key,
            admin_api_key=admin_api_key,
            auth_level=auth_level,
        )

    def test_no_keys_configured(self):
        # No keys configured -> NORMAL + ADMIN_OPTIONAL are open (legacy),
        # but ADMIN_FORCE must be rejected (403) explicitly.
        self.assertTrue(
            self._decide(
                method="GET",
                path="/v1/models",
                authorization_header=None,
                api_key=None,
                admin_api_key=None,
                auth_level=AuthLevel.NORMAL,
            ).allowed
        )
        self.assertTrue(
            self._decide(
                method="POST",
                path="/admin_optional_demo",
                authorization_header=None,
                api_key=None,
                admin_api_key=None,
                auth_level=AuthLevel.ADMIN_OPTIONAL,
            ).allowed
        )

        d = self._decide(
            method="POST",
            path="/admin_force_demo",
            authorization_header=None,
            api_key=None,
            admin_api_key=None,
            auth_level=AuthLevel.ADMIN_FORCE,
        )
        self.assertFalse(d.allowed)
        self.assertEqual(d.error_status_code, 403)

    def test_api_key_only(self):
        # api_key configured -> NORMAL requires api_key (legacy).
        self.assertFalse(
            self._decide(
                method="GET",
                path="/v1/models",
                authorization_header=None,
                api_key="user",
                admin_api_key=None,
                auth_level=AuthLevel.NORMAL,
            ).allowed
        )
        self.assertTrue(
            self._decide(
                method="GET",
                path="/v1/models",
                authorization_header="Bearer user",
                api_key="user",
                admin_api_key=None,
                auth_level=AuthLevel.NORMAL,
            ).allowed
        )

        # ADMIN_OPTIONAL requires api_key when only api_key is configured.
        self.assertFalse(
            self._decide(
                method="POST",
                path="/admin_optional_demo",
                authorization_header="Bearer wrong",
                api_key="user",
                admin_api_key=None,
                auth_level=AuthLevel.ADMIN_OPTIONAL,
            ).allowed
        )
        self.assertTrue(
            self._decide(
                method="POST",
                path="/admin_optional_demo",
                authorization_header="Bearer user",
                api_key="user",
                admin_api_key=None,
                auth_level=AuthLevel.ADMIN_OPTIONAL,
            ).allowed
        )

        # ADMIN_FORCE must be rejected even if api_key is configured (403).
        d = self._decide(
            method="POST",
            path="/admin_force_demo",
            authorization_header="Bearer user",
            api_key="user",
            admin_api_key=None,
            auth_level=AuthLevel.ADMIN_FORCE,
        )
        self.assertFalse(d.allowed)
        self.assertEqual(d.error_status_code, 403)

    def test_admin_api_key_only(self):
        # admin_api_key only:
        # - normal endpoints open
        # - optional/force endpoints require admin_api_key
        self.assertTrue(
            self._decide(
                method="GET",
                path="/v1/models",
                authorization_header="Bearer user",
                api_key=None,
                admin_api_key="admin",
                auth_level=AuthLevel.NORMAL,
            ).allowed
        )
        self.assertTrue(
            self._decide(
                method="GET",
                path="/v1/models",
                authorization_header=None,
                api_key=None,
                admin_api_key="admin",
                auth_level=AuthLevel.NORMAL,
            ).allowed
        )

        # Optional endpoints require admin_api_key when admin_api_key is configured.
        self.assertTrue(
            self._decide(
                method="POST",
                path="/admin_optional_demo",
                authorization_header="Bearer admin",
                api_key=None,
                admin_api_key="admin",
                auth_level=AuthLevel.ADMIN_OPTIONAL,
            ).allowed
        )
        self.assertFalse(
            self._decide(
                method="POST",
                path="/admin_optional_demo",
                authorization_header="Bearer user",
                api_key=None,
                admin_api_key="admin",
                auth_level=AuthLevel.ADMIN_OPTIONAL,
            ).allowed
        )

        d = self._decide(
            method="POST",
            path="/admin_force_demo",
            authorization_header="Bearer admin",
            api_key=None,
            admin_api_key="admin",
            auth_level=AuthLevel.ADMIN_FORCE,
        )
        self.assertTrue(d.allowed)

    def test_with_both_api_keys(self):
        # both api_key and admin_api_key configured:
        # - normal endpoints require api_key
        # - optional endpoints require admin_api_key (api_key is NOT accepted)
        # - force endpoints require admin_api_key
        self.assertTrue(
            self._decide(
                method="GET",
                path="/v1/models",
                authorization_header="Bearer user",
                api_key="user",
                admin_api_key="admin",
                auth_level=AuthLevel.NORMAL,
            ).allowed
        )
        self.assertFalse(
            self._decide(
                method="GET",
                path="/v1/models",
                authorization_header="Bearer admin",
                api_key="user",
                admin_api_key="admin",
                auth_level=AuthLevel.NORMAL,
            ).allowed
        )
        # Optional endpoints must require admin_api_key when both keys are configured.
        self.assertFalse(
            self._decide(
                method="POST",
                path="/admin_optional_demo",
                authorization_header="Bearer user",
                api_key="user",
                admin_api_key="admin",
                auth_level=AuthLevel.ADMIN_OPTIONAL,
            ).allowed
        )
        self.assertTrue(
            self._decide(
                method="POST",
                path="/admin_optional_demo",
                authorization_header="Bearer admin",
                api_key="user",
                admin_api_key="admin",
                auth_level=AuthLevel.ADMIN_OPTIONAL,
            ).allowed
        )
        self.assertFalse(
            self._decide(
                method="POST",
                path="/admin_force_demo",
                authorization_header="Bearer user",
                api_key="user",
                admin_api_key="admin",
                auth_level=AuthLevel.ADMIN_FORCE,
            ).allowed
        )
        self.assertTrue(
            self._decide(
                method="POST",
                path="/admin_force_demo",
                authorization_header="Bearer admin",
                api_key="user",
                admin_api_key="admin",
                auth_level=AuthLevel.ADMIN_FORCE,
            ).allowed
        )

    def test_options_is_always_allowed(self):
        # CORS preflight should never be blocked.
        self.assertTrue(
            self._decide(
                method="OPTIONS",
                path="/v1/models",
                authorization_header=None,
                api_key="user",
                admin_api_key="admin",
                auth_level=AuthLevel.ADMIN_FORCE,
            ).allowed
        )

    def test_health_and_metrics_are_always_allowed(self):
        # Health/metrics endpoints are always public by design, regardless of auth level / keys.
        combos = [
            dict(api_key=None, admin_api_key=None),
            dict(api_key="user", admin_api_key=None),
            dict(api_key=None, admin_api_key="admin"),
            dict(api_key="user", admin_api_key="admin"),
        ]
        paths_allowed = [
            "/health",
            "/health_generate",
            "/metrics",
            "/metrics/",
            "/metrics/prometheus",
        ]
        for keys in combos:
            for path in paths_allowed:
                self.assertTrue(
                    self._decide(
                        method="GET",
                        path=path,
                        authorization_header=None,
                        api_key=keys["api_key"],
                        admin_api_key=keys["admin_api_key"],
                        auth_level=AuthLevel.ADMIN_FORCE,
                    ).allowed,
                    msg=f"expected allowed for {path=} with {keys=}",
                )


class TestDynamicAuthMiddleware(unittest.TestCase):
    def _make_app(self, keys, *, single_tokenizer=False):
        from fastapi import FastAPI

        state = {"ready": False}

        def provider():
            if single_tokenizer:
                return None
            if not state["ready"]:
                raise AuthConfigNotReady()
            return keys

        @asynccontextmanager
        async def lifespan(app):
            # The middleware stack is already built when lifespan runs.
            state["ready"] = True
            yield
            state["ready"] = False

        app = FastAPI(lifespan=lifespan)

        @app.get("/v1/models")
        @app.get("/health")
        @app.get("/metrics")
        async def public():
            return {"ok": True}

        @app.get("/admin")
        @auth_level(AuthLevel.ADMIN_OPTIONAL)
        async def admin():
            return {"ok": True}

        @app.get("/admin_force")
        @auth_level(AuthLevel.ADMIN_FORCE)
        async def admin_force():
            return {"ok": True}

        add_api_key_middleware(
            app, api_key=None, admin_api_key=None, auth_config_provider=provider
        )
        if single_tokenizer:
            add_api_key_middleware(app, api_key=keys[0], admin_api_key=keys[1])
        return app

    def test_keys_resolved_after_lifespan_startup(self):
        from fastapi.testclient import TestClient

        for single in (False, True):
            for api_key, admin_key in (
                (None, None),
                ("user", None),
                (None, "admin"),
                ("user", "admin"),
            ):
                with self.subTest(single=single, api_key=api_key, admin_key=admin_key):
                    with TestClient(
                        self._make_app((api_key, admin_key), single_tokenizer=single)
                    ) as client:
                        for path, expected_key in (
                            ("/v1/models", api_key),
                            ("/admin", admin_key or api_key),
                        ):
                            for token in (None, "wrong", "user", "admin"):
                                headers = (
                                    {}
                                    if token is None
                                    else {"Authorization": f"Bearer {token}"}
                                )
                                expected = (
                                    200
                                    if not expected_key or token == expected_key
                                    else 401
                                )
                                self.assertEqual(
                                    client.get(path, headers=headers).status_code,
                                    expected,
                                )
                        self.assertEqual(client.get("/health").status_code, 200)
                        self.assertEqual(client.get("/metrics").status_code, 200)
                        for token in (None, "wrong", "user", "admin"):
                            headers = (
                                {}
                                if token is None
                                else {"Authorization": f"Bearer {token}"}
                            )
                            expected = (
                                403
                                if not admin_key
                                else (200 if token == admin_key else 401)
                            )
                            self.assertEqual(
                                client.get("/admin_force", headers=headers).status_code,
                                expected,
                            )

    def test_uninitialized_config_returns_503(self):
        from fastapi.testclient import TestClient

        # Without the context manager, TestClient does not run lifespan.
        client = TestClient(self._make_app(("user", None)))
        self.addCleanup(client.close)
        response = client.get("/v1/models")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(), {"error": "Authentication configuration is not ready"}
        )

    def test_no_keys_without_force_route_skips_route_matching(self):
        from unittest.mock import patch

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.get("/v1/models")
        async def models():
            return {"ok": True}

        add_api_key_middleware(
            app,
            api_key=None,
            admin_api_key=None,
            auth_config_provider=lambda: (None, None),
        )
        with patch(
            "sglang.srt.utils.auth._get_auth_level_from_app_and_scope"
        ) as match_auth_level:
            with TestClient(app) as client:
                self.assertEqual(client.get("/v1/models").status_code, 200)
            match_auth_level.assert_not_called()

    def test_no_keys_with_force_route_preserves_force_auth(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.get("/v1/models")
        async def models():
            return {"ok": True}

        @app.get("/admin_force")
        @auth_level(AuthLevel.ADMIN_FORCE)
        async def admin_force():
            return {"ok": True}

        add_api_key_middleware(
            app,
            api_key=None,
            admin_api_key=None,
            auth_config_provider=lambda: (None, None),
        )
        with TestClient(app) as client:
            self.assertEqual(client.get("/v1/models").status_code, 200)
            self.assertEqual(client.get("/admin_force").status_code, 403)

    def test_not_ready_precedes_no_key_fast_path(self):
        from unittest.mock import patch

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        state = {"ready": False}

        def provider():
            if not state["ready"]:
                raise AuthConfigNotReady()
            return None, None

        app = FastAPI()

        @app.get("/v1/models")
        async def models():
            return {"ok": True}

        add_api_key_middleware(
            app, api_key=None, admin_api_key=None, auth_config_provider=provider
        )
        with patch(
            "sglang.srt.utils.auth._get_auth_level_from_app_and_scope"
        ) as match_auth_level:
            with TestClient(app) as client:
                self.assertEqual(client.get("/v1/models").status_code, 503)
                state["ready"] = True
                self.assertEqual(client.get("/v1/models").status_code, 200)
            match_auth_level.assert_not_called()

    def test_dynamic_key_change_takes_effect_immediately(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        state = {"keys": (None, None)}
        app = FastAPI()

        @app.get("/v1/models")
        async def models():
            return {"ok": True}

        add_api_key_middleware(
            app,
            api_key=None,
            admin_api_key=None,
            auth_config_provider=lambda: state["keys"],
        )
        with TestClient(app) as client:
            self.assertEqual(client.get("/v1/models").status_code, 200)
            state["keys"] = ("user", None)
            self.assertEqual(client.get("/v1/models").status_code, 401)
            self.assertEqual(
                client.get(
                    "/v1/models", headers={"Authorization": "Bearer user"}
                ).status_code,
                200,
            )

    def test_force_route_added_before_stack_build_is_detected(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        add_api_key_middleware(
            app,
            api_key=None,
            admin_api_key=None,
            auth_config_provider=lambda: (None, None),
        )

        @app.get("/admin_force")
        @auth_level(AuthLevel.ADMIN_FORCE)
        async def admin_force():
            return {"ok": True}

        with TestClient(app) as client:
            self.assertEqual(client.get("/admin_force").status_code, 403)

    def test_non_http_scope_does_not_call_provider(self):
        from unittest.mock import Mock

        from fastapi import FastAPI, WebSocket
        from fastapi.testclient import TestClient

        provider = Mock(side_effect=AssertionError("provider must not be called"))
        app = FastAPI()

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            await websocket.close()

        add_api_key_middleware(
            app, api_key=None, admin_api_key=None, auth_config_provider=provider
        )
        with TestClient(app) as client:
            with client.websocket_connect("/ws"):
                pass
        provider.assert_not_called()

    def test_unexpected_provider_error_is_not_bypassed(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        def broken_provider():
            raise ValueError("bad auth configuration")

        app = FastAPI()
        add_api_key_middleware(
            app, api_key=None, admin_api_key=None, auth_config_provider=broken_provider
        )
        with TestClient(app) as client:
            with self.assertRaisesRegex(ValueError, "bad auth configuration"):
                client.get("/v1/models")


if __name__ == "__main__":
    unittest.main()
