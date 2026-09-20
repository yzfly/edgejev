"""起一个官方协议兼容的 `/v1/systemone`，官方 SDK 改个 base_url 就能切过来。

    edgejev serve --model ./jev-int8 --port 8009
    export TYPESAFE_BASE_URL=http://127.0.0.1:8009
    export TYPESAFE_API_KEY=local

与 githubnext/localjev 那类「提示上游 LLM 自报概率」的桥不同：这里的概率是
模型打分头直接算出来的，没有生成、没有 JSON 解析、没有失败重试。
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ALIASES = {"jev-latest", "jev-preview", "jev-1.13.0"}


def make_handler(agent, api_key=None):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.rstrip("/") in ("/ready", "/health"):
                return self._send(200, {"ok": True, "model": agent.model_name,
                                        "backend": agent.cfg.get("backend"),
                                        "precision": agent.cfg.get("precision"),
                                        "provider": agent.provider_note})
            self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self._send(404, {"error": {"message": "not found"}})
            if api_key:
                auth = self.headers.get("Authorization", "")
                if auth != "Bearer %s" % api_key:
                    return self._send(401, {"error": {"message": "bad api key"}})
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                state, questions = req.get("state"), req.get("questions")
                if state is None or not questions:
                    raise ValueError("需要 state 和 questions")
                out = agent.system_one(state, questions)
            except Exception as e:
                return self._send(400, {"error": {"message": str(e)[:400],
                                                  "type": "invalid_request_error"}})
            self._send(200, out)

    return H


def run(model_dir, host="127.0.0.1", port=8009, api_key=None, threads=None, provider=None):
    from .agent import Agent

    ag = Agent(model_dir, threads=threads, provider=provider)
    api_key = api_key or os.environ.get("EDGEJEV_API_KEY")
    srv = ThreadingHTTPServer((host, port), make_handler(ag, api_key))
    print("EdgeJev 已启动 http://%s:%d/v1/systemone" % (host, port))
    print("  模型 %s | 后端 %s | %s | %s"
          % (ag.model_name, ag.cfg.get("backend"), ag.cfg.get("precision"), ag.provider_note))
    print("  export TYPESAFE_BASE_URL=http://%s:%d" % (host, port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
