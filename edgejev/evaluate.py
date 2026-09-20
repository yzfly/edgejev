"""在公开数据集上复现评测指标，以及测延迟。

    edgejev eval  --model ./jev-int8 --task all --n 400
    edgejev bench --model ./jev-int8 --questions 3

口径对齐 classifier.dev 公布的 benchmark（AG News / dair-ai emotion，各 400 条），
所以数字可以直接和托管的真 Jev 对照。
"""
import json
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request

TASKS = {
    "ag_news": dict(
        ds="fancyzhx/ag_news", cfg="default", split="test",
        instructions="Which category does this text belong to?",
        labels=["World", "Sports", "Business", "Sci/Tech"],
        desc={"World": "international news, politics, war, society",
              "Sports": "sports results, athletes, teams, matches",
              "Business": "companies, markets, economy, finance",
              "Sci/Tech": "science, technology, computing, research"},
        best="desc"),
    "emotion": dict(
        ds="dair-ai/emotion", cfg="split", split="test",
        instructions="Which emotion does the writer express?",
        labels=["sadness", "joy", "love", "anger", "fear", "surprise"],
        desc={"sadness": "feeling sad, down, depressed, hurt",
              "joy": "feeling happy, glad, pleased, content",
              "love": "feeling love, affection, caring, tenderness",
              "anger": "feeling angry, irritated, annoyed, furious",
              "fear": "feeling afraid, scared, anxious, worried",
              "surprise": "feeling surprised, amazed, shocked, stunned"},
        best="bare"),
}


def _get(url, retries=6):
    """datasets-server 偶发 502/503，退避重试。"""
    for a in range(retries):
        try:
            return json.load(urllib.request.urlopen(url, timeout=60))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if a == retries - 1 or (code is not None and code not in (429, 500, 502, 503, 504)):
                raise
            time.sleep(min(20, 2 ** a))


def fetch(task, n):
    t = TASKS[task]
    out = []
    while len(out) < n:
        q = urllib.parse.urlencode({"dataset": t["ds"], "config": t["cfg"],
                                    "split": t["split"], "offset": len(out),
                                    "length": min(100, n - len(out))})
        rows = _get("https://datasets-server.huggingface.co/rows?" + q)["rows"]
        if not rows:
            break
        out += [(r["row"]["text"], t["labels"][int(r["row"]["label"])]) for r in rows]
    return out


def _criteria(t, style):
    if style == "bare":
        return {l: None for l in t["labels"]}
    return {l: t["desc"][l] for l in t["labels"]}


def run(model_dir, task="all", n=400, labels="auto", batch=16):
    from .agent import Agent

    ag = Agent(model_dir)
    print("模型 %s | 后端 %s | %s | %s"
          % (ag.model_name, ag.cfg.get("backend"), ag.cfg.get("precision"), ag.provider_note))
    tasks = list(TASKS) if task == "all" else [task]
    results = {}
    for name in tasks:
        t = TASKS[name]
        style = t["best"] if labels == "auto" else labels
        data = fetch(name, n)
        qdef = {"label": {"type": "choice", "instructions": t["instructions"],
                          "criteria": _criteria(t, style)}}
        ok, lat = 0, []
        for text, gold in data:
            s = time.perf_counter()
            r = ag.system_one(text, qdef)
            lat.append((time.perf_counter() - s) * 1000)
            ok += r["answers"]["label"]["choice"] == gold
        acc = ok / len(data)
        results[name] = {"acc": acc, "n": len(data), "labels": style,
                         "ms_median": statistics.median(lat)}
        print("  %-9s (%s标签, n=%d): 准确率 %.1f%%  延迟中位 %.1f ms"
              % (name, "带描述" if style == "desc" else "光", len(data), acc * 100,
                 statistics.median(lat)))
    return results


def bench(model_dir, questions=3, iters=30):
    from .agent import Agent

    ag = Agent(model_dir)
    print("模型 %s | %s | %s" % (ag.model_name, ag.cfg.get("precision"), ag.provider_note))
    pool = {
        "dept": {"type": "choice", "instructions": "which team should handle this",
                 "criteria": {"billing": "payments", "technical": "bugs", "sales": "pricing"}},
        "urgent": {"type": "noul", "instructions": "is this urgent"},
        "anger": {"type": "score", "instructions": "how frustrated",
                  "criteria": ["calm", "annoyed", "furious"]},
        "refund": {"type": "noul", "instructions": "was a refund requested"},
    }
    qs = dict(list(pool.items())[:questions])
    state = "The customer was charged twice and is asking for a refund."
    for _ in range(5):
        ag.system_one(state, qs)
    ts = []
    for _ in range(iters):
        s = time.perf_counter()
        ag.system_one(state, qs)
        ts.append((time.perf_counter() - s) * 1000)
    ts.sort()
    print("  %d 题一次请求，n=%d：最快 %.1f / 中位 %.1f / p95 %.1f ms（每题 %.1f ms）"
          % (questions, iters, ts[0], statistics.median(ts),
             ts[min(len(ts) - 1, int(len(ts) * 0.95))], statistics.median(ts) / questions))
