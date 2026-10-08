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
