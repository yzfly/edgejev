"""edgejev 命令行： build / serve / eval / bench / info"""
import argparse
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="edgejev",
                                description="在自己的设备上跑 Jev 式的类型化决策")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="把上游 checkpoint 转成 EdgeJev 目录")
    b.add_argument("--backend", default="laya", help="laya | kev | nanojev | playjev")
    b.add_argument("--model", default=None, help="HF 模型 id 或本地路径，默认按后端选")
    b.add_argument("--subfolder", default=None)
    b.add_argument("--out", required=True)
    b.add_argument("--precision", default=None,
                   help="不给就用后端的默认值（laya=int8，kev/nanojev/playjev=fp32）。"
                        "可选 int8 | int8-pc | mixed | fp32")
    b.add_argument("--keep-fp32", action="store_true")

    s = sub.add_parser("serve", help="起一个官方协议兼容的 /v1/systemone")
    s.add_argument("--model", required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8009)
    s.add_argument("--api-key", default=None)
    s.add_argument("--threads", type=int, default=None)
    s.add_argument("--provider", default=None, help="auto | cpu | coreml")

    e = sub.add_parser("eval", help="在公开数据集上复现评测指标")
    e.add_argument("--model", required=True)
    e.add_argument("--task", default="all", help="ag_news | emotion | all")
    e.add_argument("--n", type=int, default=400)
    e.add_argument("--labels", default="auto", help="auto | bare | desc")
    e.add_argument("--batch", type=int, default=16)

    k = sub.add_parser("bench", help="测延迟")
    k.add_argument("--model", required=True)
    k.add_argument("--questions", type=int, default=3)
    k.add_argument("--iters", type=int, default=30)

    sub.add_parser("info", help="打印运行时环境")

    a = p.parse_args(argv)
    if a.cmd == "build":
        from . import build
        build.run(backend=a.backend, model=a.model, subfolder=a.subfolder,
                  out=a.out, precision=a.precision, keep_fp32=a.keep_fp32)
    elif a.cmd == "serve":
        from . import serve
        serve.run(a.model, a.host, a.port, a.api_key, a.threads, a.provider)
    elif a.cmd == "eval":
        from . import evaluate
        evaluate.run(a.model, a.task, a.n, a.labels, a.batch)
    elif a.cmd == "bench":
        from . import evaluate
        evaluate.bench(a.model, a.questions, a.iters)
    elif a.cmd == "info":
        from . import providers, backends
        print(providers.describe())
        print("已注册后端:", ", ".join(backends.names()))


if __name__ == "__main__":
    sys.exit(main())
