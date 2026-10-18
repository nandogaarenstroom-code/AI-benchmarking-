import http.client
import json
import socket
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import app


class StubManager:
    def __init__(self):
        self.payload = None

    def start(self, payload):
        self.payload = payload
        return True

    def resume(self):
        return False

    def cancel(self):
        return False

    def load_result(self, result_id):
        return None

    def read_artifact(self, filename):
        return None


class ApiSecurityTests(unittest.TestCase):
    def setUp(self):
        self.server = app.BoundedThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.manager = StubManager()
        self.manager_patch = patch.object(app, "manager", self.manager)
        self.manager_patch.start()

    def tearDown(self):
        self.manager_patch.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, path, body, headers):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        connection.request("POST", path, body=body, headers=headers)
        response = connection.getresponse()
        result = response.status, response.read()
        connection.close()
        return result

    def test_json_response_ignores_client_disconnect_during_write(self):
        for failure_point in ("end_headers", "write"):
            for error in (BrokenPipeError(), ConnectionAbortedError(), ConnectionResetError()):
                with self.subTest(failure_point=failure_point, error=type(error).__name__):
                    handler = object.__new__(app.Handler)
                    handler.send_response = Mock()
                    handler.send_header = Mock()
                    handler.end_headers = Mock()
                    handler.wfile = Mock()
                    if failure_point == "end_headers":
                        handler.end_headers.side_effect = error
                    else:
                        handler.wfile.write.side_effect = error

                    handler._json({"models": []})

    def test_hardware_endpoint_returns_detection_and_advice(self):
        hardware = {"cpu": "Test CPU", "ram": "16 GiB"}
        advice = {"generation_options": {"max_tokens": 4096}}
        with patch.object(app, "hardware_info", return_value=hardware), patch.object(
                app, "recommend_settings", return_value=advice) as recommend:
            connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
            connection.request("GET", "/api/hardware")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read()),
                             {"hardware": hardware, "recommendations": advice})
            connection.close()
            recommend.assert_called_once_with(hardware)

    def test_nested_html_artifact_is_sandboxed(self):
        self.manager.read_artifact = Mock(return_value=b"<!doctype html><title>Voxel</title>")
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        connection.request("GET", "/results/model-abc/benchmark_20261018_120000_abcdef1234.html")
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "text/html; charset=utf-8")
        self.assertIn("sandbox allow-scripts", response.getheader("Content-Security-Policy"))
        self.assertNotIn("allow-same-origin", response.getheader("Content-Security-Policy"))
        response.read()
        connection.close()
        self.manager.read_artifact.assert_called_once_with(
            "model-abc/benchmark_20261018_120000_abcdef1234.html")

    def test_dashboard_contains_advanced_controls_and_live_rates(self):
        from html.parser import HTMLParser

        class Elements(HTMLParser):
            def __init__(self):
                super().__init__()
                self.elements = {}

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if "id" in attrs:
                    self.elements[attrs["id"]] = attrs

        parser = Elements()
        parser.feed((app.ROOT / "frontend" / "index.html").read_text())
        self.assertEqual(parser.elements["advanced-tab"]["role"], "tab")
        self.assertEqual(parser.elements["advanced-settings"]["role"], "tabpanel")
        self.assertIn("hidden", parser.elements["advanced-settings"])
        self.assertEqual(parser.elements["max-tokens"]["max"], "32768")
        self.assertIn("required", parser.elements["max-tokens"])
        self.assertIn("required", parser.elements["suite-seed"])
        for control in ("llama-gpu-layers", "llama-context", "runtime-context",
                        "runtime-gpu", "runtime-threads", "live-speeds",
                        "hardware-summary", "apply-recommendations", "suite-seed",
                        "launch-threads", "launch-extra-args", "llama-port"):
            self.assertIn(control, parser.elements)

    def test_cross_origin_and_missing_token_posts_are_rejected(self):
        body = json.dumps({"profile": "QUICK"})
        headers = {
            "Origin": "http://attacker.invalid",
            "Content-Type": "application/json",
            "X-CSRF-Token": app.CSRF_TOKEN,
        }
        status, _ = self.request("/api/benchmark", body, headers)
        self.assertEqual(status, 403)

        headers.update({"Origin": f"http://127.0.0.1:{self.server.server_port}"})
        headers.pop("X-CSRF-Token")
        status, _ = self.request("/api/benchmark", body, headers)
        self.assertEqual(status, 403)
        self.assertIsNone(self.manager.payload)

    def test_every_state_changing_route_requires_origin_and_csrf(self):
        routes = (
            "/api/benchmark", "/api/resume", "/api/cancel",
            "/api/llama-server/start", "/api/llama-server/stop",
            "/api/llama-server/select",
        )
        for path in routes:
            with self.subTest(path=path):
                status, _ = self.request(path, "{}", {
                    "Origin": f"http://127.0.0.1:{self.server.server_port}",
                    "Content-Type": "application/json",
                })
                self.assertEqual(status, 403)
                status, _ = self.request(path, "{}", {
                    "Origin": "http://attacker.invalid",
                    "Content-Type": "application/json",
                    "X-CSRF-Token": app.CSRF_TOKEN,
                })
                self.assertEqual(status, 403)

    def test_request_target_and_body_limits_cover_get_and_post(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        connection.request("GET", "/api/models?" + "x" * app.MAX_REQUEST_TARGET_SIZE,
                           headers={"Host": f"127.0.0.1:{self.server.server_port}"})
        response = connection.getresponse()
        response.read()
        connection.close()
        self.assertEqual(response.status, 414)

        headers = {
            "Origin": f"http://127.0.0.1:{self.server.server_port}",
            "Content-Type": "application/json",
            "X-CSRF-Token": app.CSRF_TOKEN,
        }
        too_large = "{}" + " " * (app.MAX_REQUEST_SIZE - 1)
        self.assertEqual(len(too_large), app.MAX_REQUEST_SIZE + 1)
        with socket.create_connection(("127.0.0.1", self.server.server_port), timeout=3) as client:
            request = (
                f"POST /api/benchmark HTTP/1.0\r\n"
                f"Host: 127.0.0.1:{self.server.server_port}\r\n"
                f"Content-Length: {len(too_large)}\r\n\r\n"
            )
            client.sendall(request.encode())
            self.assertIn(b"413", client.recv(4096))
        with socket.create_connection(("127.0.0.1", self.server.server_port), timeout=3) as client:
            request = (
                f"POST /api/benchmark HTTP/1.0\r\n"
                f"Host: 127.0.0.1:{self.server.server_port}\r\n"
                f"Content-Length: {'9' * 5000}\r\n\r\n"
            )
            client.sendall(request.encode())
            self.assertIn(b"413", client.recv(4096))
        boundary = "{}" + " " * (app.MAX_REQUEST_SIZE - 2)
        self.assertEqual(len(boundary), app.MAX_REQUEST_SIZE)
        status, body = self.request("/api/benchmark", boundary, headers)
        self.assertEqual(status, 400)
        self.assertIn(b"explicitly", body)

    def test_download_and_history_routes_reject_traversal_and_internal_state(self):
        for path in (
                "/api/results/../current.json",
                "/api/results/%2e%2e%2fcurrent.json",
                "/results/../current.json",
                "/results/.checkpoint.key"):
            with self.subTest(path=path):
                connection = http.client.HTTPConnection(
                    "127.0.0.1", self.server.server_port, timeout=3)
                connection.request(
                    "GET", path, headers={"Host": f"127.0.0.1:{self.server.server_port}"})
                response = connection.getresponse()
                response.read()
                connection.close()
                self.assertEqual(response.status, 404)

    def test_frontend_uses_text_nodes_for_untrusted_display(self):
        html = (Path(app.ROOT) / "frontend" / "index.html").read_text()
        self.assertNotIn("innerHTML", html)
        self.assertIn("content.textContent=", html)
        self.assertIn("cell.textContent=", html)

    def test_unsupported_content_type_is_rejected(self):
        status, _ = self.request("/api/benchmark", '{"profile":"QUICK"}', {
            "Origin": f"http://127.0.0.1:{self.server.server_port}",
            "Content-Type": "text/plain",
            "X-CSRF-Token": app.CSRF_TOKEN,
        })
        self.assertEqual(status, 415)
        self.assertIsNone(self.manager.payload)

    def test_malformed_json_and_invalid_profiles_are_rejected(self):
        headers = {
            "Origin": f"http://127.0.0.1:{self.server.server_port}",
            "Content-Type": "application/json",
            "X-CSRF-Token": app.CSRF_TOKEN,
        }
        status, _ = self.request("/api/benchmark", "{", headers)
        self.assertEqual(status, 400)
        status, _ = self.request("/api/benchmark", json.dumps({"profile": "UNKNOWN"}), headers)
        self.assertEqual(status, 400)
        self.assertIsNone(self.manager.payload)

    def test_cancel_endpoint_uses_protected_post_route(self):
        status, body = self.request("/api/cancel", "{}", {
            "Origin": f"http://127.0.0.1:{self.server.server_port}",
            "Content-Type": "application/json",
            "X-CSRF-Token": app.CSRF_TOKEN,
        })
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"cancelled": False})

    def test_valid_request_drops_caller_supplied_tests(self):
        status, _ = self.request("/api/benchmark", json.dumps({
            "profile": "QUICK", "tests": [{"evaluation_method": "code", "tests": ["raise Exception()"]}],
            "completed_tests": [{"id": "forged"}],
            "model": {"id": "demo-local", "backend": "Demo", "endpoint": "local://demo"},
        }), {
            "Origin": f"http://127.0.0.1:{self.server.server_port}",
            "Content-Type": "application/json",
            "X-CSRF-Token": app.CSRF_TOKEN,
        })
        self.assertEqual(status, 200)
        self.assertNotIn("tests", self.manager.payload)
        self.assertNotIn("completed_tests", self.manager.payload)

    def test_benchmark_api_requires_explicit_model_selection(self):
        status, body = self.request("/api/benchmark", '{"profile":"QUICK"}', {
            "Origin": f"http://127.0.0.1:{self.server.server_port}",
            "Content-Type": "application/json",
            "X-CSRF-Token": app.CSRF_TOKEN,
        })
        self.assertEqual(status, 400)
        self.assertIn("explicitly", json.loads(body)["error"])
        self.assertIsNone(self.manager.payload)

    def test_api_omits_missing_output_limit_and_preserves_explicit_override(self):
        headers = {
            "Origin": f"http://127.0.0.1:{self.server.server_port}",
            "Content-Type": "application/json",
            "X-CSRF-Token": app.CSRF_TOKEN,
        }
        for supplied, expected in (({}, None), ({"max_tokens": 512}, 512)):
            with self.subTest(supplied=supplied), patch.object(
                    app, "hardware_info", return_value={}):
                status, _ = self.request("/api/benchmark", json.dumps({
                    "model": {"id": "demo-local", "backend": "Demo", "endpoint": "local://demo"},
                    "generation_options": supplied,
                }), headers)
                self.assertEqual(status, 200)
                self.assertEqual(self.manager.payload["generation_options"].get("max_tokens"), expected)
                self.assertNotIn("num_gpu", self.manager.payload["generation_options"])

    def test_api_model_response_exposes_request_capability_matrix(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        connection.request("GET", "/api/models")
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        self.assertIn("parameter_capabilities", payload)
        self.assertEqual(set(payload["parameter_capabilities"]),
                         {"ollama", "lm_studio", "openai_compatible", "llama_cpp"})

    def test_api_invalid_option_returns_field_specific_error(self):
        headers = {
            "Origin": f"http://127.0.0.1:{self.server.server_port}",
            "Content-Type": "application/json",
            "X-CSRF-Token": app.CSRF_TOKEN,
        }
        status, body = self.request("/api/benchmark", json.dumps({
            "model": {"id": "demo-local", "backend": "Demo", "endpoint": "local://demo"},
            "generation_options": {"top_p": 9},
        }), headers)
        response = json.loads(body)
        self.assertEqual(status, 400)
        self.assertEqual(response["field"], "top_p")
        self.assertIn("top_p", response["error"])

    def test_llama_server_start_requires_csrf_and_passes_only_launcher_options(self):
        body = json.dumps({
            "source": "path", "executable_path": "C:\\models\\llama-server.exe",
            "package_id": "vendor.custom-fork",
            "model_path": "C:\\models\\test.gguf", "context_size": 65536,
            "gpu_layers": 24, "port": 9000, "launch_options": {"threads": 4},
            "tests": [{"id": "caller-controlled"}],
        })
        headers = {
            "Origin": f"http://127.0.0.1:{self.server.server_port}",
            "Content-Type": "application/json",
            "X-CSRF-Token": app.CSRF_TOKEN,
        }
        with patch.object(app.llama_server_manager, "start", return_value={
                "state": "running", "message": "ready"} ) as start:
            status, body_bytes = self.request("/api/llama-server/start", body, headers)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body_bytes), {"state": "running", "message": "ready"})
        start.assert_called_once_with(
            "path", "C:\\models\\llama-server.exe", "vendor.custom-fork",
            "C:\\models\\test.gguf", 65536, 24,
            launch_options={"threads": 4}, port=9000)

        headers.pop("X-CSRF-Token")
        with patch.object(app.llama_server_manager, "start") as start:
            status, _ = self.request("/api/llama-server/start", body, headers)
        self.assertEqual(status, 403)
        start.assert_not_called()

    def test_llama_server_status_and_stop_routes(self):
        with patch.object(app.llama_server_manager, "capabilities", return_value={
                "flags": {"--threads": {"takes_value": True}}, "supported": ["--threads"]}) as caps:
            connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
            connection.request("GET", "/api/llama-server/capabilities",
                               headers={"Host": f"127.0.0.1:{self.server.server_port}"})
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertIn("--threads", json.loads(response.read())["supported"])
            connection.close()
            caps.assert_called_once_with()

        with patch.object(app.llama_server_manager, "status", return_value={
                "state": "running", "message": "ready"}):
            connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
            connection.request("GET", "/api/llama-server/status",
                               headers={"Host": f"127.0.0.1:{self.server.server_port}"})
            response = connection.getresponse()
            self.assertEqual(json.loads(response.read()), {"state": "running", "message": "ready"})
            connection.close()
        with patch.object(app.llama_server_manager, "stop", return_value={
                "state": "stopped", "message": "stopped"}) as stop:
            status, body = self.request("/api/llama-server/stop", "{}", {
                "Origin": f"http://127.0.0.1:{self.server.server_port}",
                "Content-Type": "application/json",
                "X-CSRF-Token": app.CSRF_TOKEN,
            })
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["state"], "stopped")
        stop.assert_called_once_with()

    def test_frontend_receives_per_process_csrf_token(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        connection.request("GET", "/", headers={"Host": f"127.0.0.1:{self.server.server_port}"})
        response = connection.getresponse()
        body = response.read().decode()
        connection.close()
        self.assertEqual(response.status, 200)
        self.assertIn(app.CSRF_TOKEN, body)
        self.assertNotIn("__CSRF_TOKEN__", body)


if __name__ == "__main__":
    unittest.main()
