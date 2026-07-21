"""
L7 Day 4: Agent 监控与评估
===========================
Day 1-3 搭建了结构化通信、状态机、持久化记忆。
但怎么知道这个 Multi-Agent 系统"好不好"？

本 demo 实现：
1. Agent 决策追踪 — 每一步都打日志（结合 L5 的 Trace ID 思路）
2. 指标聚合 — 成功率、平均耗时、重复率、瓶颈检测
3. 可视化报表 — 直接输出指标摘要
"""

import json
import os
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict
from openai import OpenAI

client = OpenAI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================
# 一、追踪系统
# =============================================

class TraceLogger:
    """
    结构化日志记录器。每个 Agent 共享同一个 trace，
    按 trace_id 聚合能看到完整的决策链路。
    """

    def __init__(self):
        self.trace_id = uuid.uuid4().hex[:12]
        self.events: list[dict] = []
        self.round = 0

    def log(self, agent: str, action: str, status: str,
            detail: str = "", duration: float = 0.0, **extra):
        self.round += 1
        event = {
            "round": self.round,
            "trace_id": self.trace_id,
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "status": status,
            "duration_s": round(duration, 2),
            "detail": detail[:120],
            **extra
        }
        self.events.append(event)
        return event

    def get_agent_events(self, agent: str) -> list[dict]:
        return [e for e in self.events if e["agent"] == agent]

    def summary(self) -> dict:
        """生成聚合指标"""
        if not self.events:
            return {"error": "no events"}

        total = len(self.events)
        by_agent = defaultdict(list)
        for e in self.events:
            by_agent[e["agent"]].append(e)

        agent_metrics = {}
        for agent, evts in by_agent.items():
            durations = [e["duration_s"] for e in evts if e["duration_s"] > 0]
            success = [e for e in evts if e["status"] == "ok"]
            failed = [e for e in evts if e["status"] in ("fail", "retry")]
            retries = len([e for e in evts if e["action"] == "revise"])

            agent_metrics[agent] = {
                "calls": len(evts),
                "success": len(success),
                "failed": len(failed),
                "retries": retries,
                "avg_duration_s": round(sum(durations) / len(durations), 2) if durations else 0,
                "total_duration_s": round(sum(durations), 2)
            }

        # 瓶颈检测：平均耗时最长的 Agent
        bottleneck = max(agent_metrics.items(),
                         key=lambda x: x[1]["avg_duration_s"])

        return {
            "trace_id": self.trace_id,
            "total_events": total,
            "total_rounds": self.round,
            "agent_metrics": dict(agent_metrics),
            "bottleneck": {
                "agent": bottleneck[0],
                "avg_duration_s": bottleneck[1]["avg_duration_s"]
            }
        }


# =============================================
# 二、消息通信（复用前几天的）
# =============================================

@dataclass
class Message:
    sender: str
    receiver: str
    msg_type: str
    payload: dict
    trace_id: str = ""
    turn: int = 1
    timestamp: float = 0.0


class MessageBus:
    def __init__(self, logger: TraceLogger):
        self.logger = logger
        self.history: list[Message] = []

    def send(self, msg: Message):
        msg.timestamp = time.time()
        msg.turn = len(self.history) + 1
        self.history.append(msg)
        self.logger.log(
            agent=msg.sender, action=f"send:{msg.msg_type}",
            status="ok", detail=f"→ {msg.receiver}"
        )
        print(f"  [{msg.turn}] {msg.sender} → {msg.receiver}  [{msg.msg_type}]")


# =============================================
# 三、可追踪的 Agent
# =============================================

class TracedAgent:
    """带执行追踪的 Agent"""

    def __init__(self, name: str, system_prompt: str, bus: MessageBus,
                 logger: TraceLogger, model: str = "deepseek-chat",
                 temperature: float = 0.3):
        self.name = name
        self.system_prompt = system_prompt
        self.bus = bus
        self.logger = logger
        self.model = model
        self.temperature = temperature

    def think(self, user_msg: str) -> str:
        start = time.time()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                temperature=self.temperature,
                max_tokens=1024
            )
            reply = response.choices[0].message.content
            duration = time.time() - start

            self.logger.log(
                agent=self.name, action="llm_call", status="ok",
                detail=f"tokens={response.usage.total_tokens if response.usage else '?'}",
                duration=duration
            )
            return reply
        except Exception as e:
            duration = time.time() - start
            self.logger.log(
                agent=self.name, action="llm_call", status="fail",
                detail=str(e)[:120], duration=duration
            )
            raise


class TracedResearcher(TracedAgent):
    SYSTEM_PROMPT = """你是资深研究员。输出 JSON 格式：
{"key_points": ["..."], "confidence": 0-1}"""

    def execute(self, topic: str) -> str:
        result = self.think(f"研究主题：{topic}")
        self.logger.log(
            agent=self.name, action="research", status="ok",
            detail=f"topic={topic[:40]}..."
        )
        return result


class TracedWriter(TracedAgent):
    SYSTEM_PROMPT = """你是科普写作者。基于研究资料创作文章，输出 JSON 格式：
{"title": "...", "content": "...", "word_count": 0}"""

    def execute(self, topic: str, research: str, round_n: int = 0) -> str:
        prompt = f"主题：{topic}\n研究资料：{research}"
        if round_n > 0:
            prompt += f"\n这是第 {round_n} 次修改。"
        result = self.think(prompt)

        action = "write" if round_n == 0 else "revise"
        self.logger.log(
            agent=self.name, action=action, status="ok",
            detail=f"round={round_n}, topic={topic[:40]}..."
        )
        return result


class TracedReviewer(TracedAgent):
    SYSTEM_PROMPT = """你是严格的内容审核员。输出 JSON 格式：
{"issues": [...], "rating": 1-5, "verdict": "通过/需要修改"}
评分低于 4 时必须输出"需要修改"。"""

    def execute(self, article: str, research: str) -> str:
        result = self.think(f"审核文章：\n{article}\n\n参考：{research}")
        return result


# =============================================
# 四、对比实验：追踪两条不同难度的流水线
# =============================================

def run_pipeline(topic: str, logger: TraceLogger, max_retries: int = 1) -> dict:
    """运行一条完整流水线，返回审核结果和指标"""
    bus = MessageBus(logger)

    researcher = TracedResearcher("researcher", TracedResearcher.SYSTEM_PROMPT,
                                  bus, logger, temperature=0.1)
    writer = TracedWriter("writer", TracedWriter.SYSTEM_PROMPT,
                          bus, logger, temperature=0.4)
    reviewer = TracedReviewer("reviewer", TracedReviewer.SYSTEM_PROMPT,
                              bus, logger, temperature=0.1)

    start = time.time()

    # 研究员
    raw_research = researcher.execute(topic)
    try:
        research_json = json.loads(
            raw_research.strip().removeprefix("```json").removesuffix("```").strip()
        )
    except:
        research_json = {"key_points": [raw_research[:100]], "confidence": 0.5}
    bus.send(Message("researcher", "writer", "research_result", research_json))
    research_result = raw_research

    # 写作 + 审核循环
    final_rating = 0
    passed = False
    for attempt in range(max_retries + 1):
        raw_article = writer.execute(topic, research_result, attempt)
        try:
            article_json = json.loads(
                raw_article.strip().removeprefix("```json").removesuffix("```").strip()
            )
        except:
            article_json = {"title": topic, "content": raw_article[:200], "word_count": len(raw_article)}

        bus.send(Message("writer", "reviewer", "article_draft", article_json))

        raw_review = reviewer.execute(raw_article, research_result)
        try:
            review_json = json.loads(
                raw_review.strip().removeprefix("```json").removesuffix("```").strip()
            )
        except:
            review_json = {"issues": [], "rating": 3, "verdict": "需要修改"}

        final_rating = review_json.get("rating", 0)
        verdict = review_json.get("verdict", "需要修改")

        logger.log(
            agent="reviewer", action="review", status="ok",
            detail=f"rating={final_rating}/5, verdict={verdict}",
            rating=final_rating
        )

        if verdict == "通过" or final_rating >= 4:
            passed = True
            logger.log(agent="pipeline", action="completed", status="ok",
                       detail=f"passed after {attempt + 1} attempt(s)")
            break

        logger.log(agent="pipeline", action="retry", status="retry",
                   detail=f"rating={final_rating}, round {attempt + 1}")

    total_duration = time.time() - start
    logger.log(agent="pipeline", action="finished", status="ok",
               duration=total_duration,
               detail=f"topic={topic[:30]}, rating={final_rating}, passed={passed}")

    return {
        "topic": topic,
        "rating": final_rating,
        "passed": passed,
        "attempts": attempt + 1,
        "duration": round(total_duration, 1)
    }


def run_comparison():
    """运行多条不同难度的流水线，对比性能"""
    print("=" * 60)
    print("L7 Day 4: Agent 监控与评估")
    print("=" * 60)

    topics = [
        "多层感知机的反向传播原理",           # 简单
        "Actor-Critic 架构如何平衡探索与利用",  # 中等
        "分布式共识算法中的 Paxos 协议"         # 复杂
    ]

    results = []
    all_loggers = []

    for i, topic in enumerate(topics):
        print(f"\n--- 流水线 {i + 1}: {topic} ---")
        logger = TraceLogger()
        result = run_pipeline(topic, logger, max_retries=1)
        results.append(result)
        all_loggers.append(logger)

        summary = logger.summary()
        print(f"\n结果: {'✓ 通过' if result['passed'] else '✗ 未通过'} "
              f"| 评分: {result['rating']}/5 "
              f"| 尝试: {result['attempts']} 次 "
              f"| 耗时: {result['duration']}s")

    # === 汇总报表 ===
    print(f"\n{'='*60}")
    print("汇总评估报告")
    print(f"{'='*60}\n")

    # 总体指标
    all_events = sum(len(l.events) for l in all_loggers)
    avg_duration = sum(r["duration"] for r in results) / len(results)
    pass_rate = sum(1 for r in results if r["passed"]) / len(results) * 100

    print(f"总流水线数: {len(results)}")
    print(f"通过率: {pass_rate:.0f}%")
    print(f"总追踪事件: {all_events}")
    print(f"平均耗时: {avg_duration:.1f}s")
    print()

    # 各主题详情
    print(f"{'主题':<35} {'评分':>4} {'状态':>6} {'尝试':>4} {'耗时':>6}")
    print("-" * 60)
    for r in results:
        status = "✓" if r["passed"] else "✗"
        print(f"{r['topic']:<35} {r['rating']:>4}/5 {status:>6} "
              f"{r['attempts']:>3}次 {r['duration']:>5.1f}s")

    # Agent 级聚合指标
    print(f"\nAgent 级性能分解")
    print(f"{'Agent':<15} {'调用':>4} {'成功':>4} {'失败':>4} "
          f"{'平均耗时':>8} {'总耗时':>8}")
    print("-" * 55)

    # 合并所有 logger 的指标
    combined = defaultdict(lambda: {"calls": 0, "success": 0, "failed": 0,
                                     "durations": []})
    for logger in all_loggers:
        summary = logger.summary()
        for agent, metrics in summary.get("agent_metrics", {}).items():
            if agent == "pipeline":
                continue
            m = combined[agent]
            m["calls"] += metrics["calls"]
            m["success"] += metrics["success"]
            m["failed"] += metrics["failed"]
            m["durations"].append(metrics["avg_duration_s"])

    for agent, m in sorted(combined.items()):
        avg_d = round(sum(m["durations"]) / len(m["durations"]), 2) if m["durations"] else 0
        total_d = round(sum(m["durations"]), 2)
        print(f"{agent:<15} {m['calls']:>4} {m['success']:>4} "
              f"{m['failed']:>4} {avg_d:>7.2f}s {total_d:>7.2f}s")

    # 瓶颈分析
    bottlenecks = []
    for logger in all_loggers:
        s = logger.summary()
        bottlenecks.append(s.get("bottleneck", {}))

    if bottlenecks:
        slowest = max(bottlenecks, key=lambda b: b.get("avg_duration_s", 0))
        print(f"\n瓶颈 Agent: {slowest.get('agent', '?')} "
              f"(平均 {slowest.get('avg_duration_s', 0)}s/次)")
        print("  提示：如果这个 Agent 耗时明显高于其他，考虑：")
        print("  - 换更快的模型（如 deepseek-chat → deepseek-turbo）")
        print("  - 减少 max_tokens")
        print("  - 给这个 Agent 分配更简单的子任务")

    # 追踪事件示例
    print(f"\n追踪事件示例（来自第一条流水线）")
    print(f"{'轮次':>4} {'Agent':<12} {'动作':<18} {'状态':<6} {'耗时':>6} {'详情'}")
    print("-" * 80)
    for e in all_loggers[0].events[:8]:
        print(f"{e['round']:>4} {e['agent']:<12} {e['action']:<18} "
              f"{e['status']:<6} {e['duration_s']:>5.1f}s {e['detail'][:50]}")


# =============================================
# 五、运行
# =============================================

if __name__ == "__main__":
    run_comparison()
