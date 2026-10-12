import http.client
import json
import threading
import unittest
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

    def test_benchmark_accepts_validated_category_scope_and_count(self):
        status, _ = self.request("/api/benchmark", json.dumps({
            "profile": "QUICK", "categories": ["Coding", "Knowledge"], "test_count": 7,
            "model": {"id": "demo-local", "backend": "Demo", "endpoint": "local://demo"},
        }), {
            "Origin": f"http://127.0.0.1:{self.server.server_port}",
            "Content-Type": "application/json",
            "X-CSRF-Token": app.CSRF_TOKEN,
        })
        self.assertEqual(status, 200)
        self.assertEqual(self.manager.payload["categories"], ["Coding", "Knowledge"])
        self.assertEqual(self.manager.payload["test_count"], 7)

        status, _ = self.request("/api/benchmark", json.dumps({
            "profile": "QUICK", "categories": ["Unknown"], "test_count": 0,
            "model": {"id": "demo-local", "backend": "Demo", "endpoint": "local://demo"},
        }), {
            "Origin": f"http://127.0.0.1:{self.server.server_port}",
            "Content-Type": "application/json",
            "X-CSRF-Token": app.CSRF_TOKEN,
        })
        self.assertEqual(status, 400)

    def test_real_model_requires_successful_chat_compatibility_check(self):
        model = {"id": "test-model", "backend": "LM Studio",
                 "endpoint": "http://localhost:1234/v1", "demo": False}
        with patch.object(app, "resolve_model", return_value=model), \
                patch.object(app, "ask_model", return_value={"text": "READY"}) as ask:
            status, body = self.request("/api/model-compatibility", json.dumps({"model": model}), {
                "Origin": f"http://127.0.0.1:{self.server.server_port}",
                "Content-Type": "application/json",
                "X-CSRF-Token": app.CSRF_TOKEN,
            })
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["compatible"])
        ask.assert_called_once_with(model, "Reply with exactly READY.", options={"max_tokens": 8})

        with patch.object(app, "resolve_model", return_value=model), \
                patch.object(app, "ask_model", side_effect=app.AdapterError("CONNECTION_ERROR", "offline")):
            status, body = self.request("/api/model-compatibility", json.dumps({"model": model}), {
                "Origin": f"http://127.0.0.1:{self.server.server_port}",
                "Content-Type": "application/json",
                "X-CSRF-Token": app.CSRF_TOKEN,
            })
        self.assertEqual(status, 502)
        self.assertFalse(json.loads(body)["compatible"])

    def test_preview_returns_generated_prompt_for_supported_category(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        connection.request("GET", "/api/benchmark-preview?category=Coding",
                           headers={"Host": f"127.0.0.1:{self.server.server_port}"})
        response = connection.getresponse()
        body = json.loads(response.read())
        connection.close()
        self.assertEqual(response.status, 200)
        self.assertEqual(body["previews"][0]["category"], "Coding")
        self.assertIn("Output Python code only.", body["previews"][0]["prompt"])

    def test_llama_server_start_requires_csrf_and_passes_only_launcher_options(self):
        body = json.dumps({
            "source": "path", "executable_path": "C:\\models\\llama-server.exe",
            "package_id": "vendor.custom-fork",
            "model_path": "C:\\models\\test.gguf", "context_size": 65536,
            "gpu_layers": 24, "tests": [{"id": "caller-controlled"}],
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
            "C:\\models\\test.gguf", 65536, 24)

        headers.pop("X-CSRF-Token")
        with patch.object(app.llama_server_manager, "start") as start:
            status, _ = self.request("/api/llama-server/start", body, headers)
        self.assertEqual(status, 403)
        start.assert_not_called()

    def test_llama_server_status_and_stop_routes(self):
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
