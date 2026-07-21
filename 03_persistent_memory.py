"""
L7 Day 3: 持久化 Agent 记忆
=============================
Day 2 暴露了一个问题：审核员每次看到文章都像"第一次见"，
重复输出同样的反馈，写作者改了也是白改。

原因：Agent 没有记忆。每次调用都是从零开始。

这一天的核心：
1. MemoryStore — 持久化存储 Agent 的决策记录
2. Memory Augmented Agent — 行动前先查记忆，避免重复错误
3. 对比：有记忆 vs 无记忆的审核效果
"""

import json
import os
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
from openai import OpenAI

client = OpenAI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================
# 一、记忆存储层
# =============================================

class MemoryStore:
    """
    轻量持久化记忆存储。
    每个 Agent 有自己的记忆文件，存储在 memory/ 目录下。
    结构：list[dict]，每个 dict 是一条记忆条目。
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.memory_dir = os.path.join(BASE_DIR, "memory")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.filepath = os.path.join(self.memory_dir, f"{agent_name}.json")
        self.memories: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)

    def add(self, memory: dict):
        """添加一条记忆"""
        memory["id"] = uuid.uuid4().hex[:8]
        memory["timestamp"] = datetime.now().isoformat()
        self.memories.append(memory)
        self._save()

    def query(self, task: str, top_k: int = 3) -> list[dict]:
        """
        按关键词匹配查询相关记忆。
        实际生产会用 embedding 语义检索，这里用关键词匹配已足够演示。
        """
        keywords = set(task.lower().split())
        scored = []
        for m in self.memories:
            # 对记忆中的 task 和 outcome 做关键词匹配
            text = (m.get("task", "") + " " + m.get("outcome", "") + " " +
                    " ".join(m.get("issues", []))).lower()
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scored.append((score, m))

        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored[:top_k]]

    def get_history(self, limit: int = 10) -> list[dict]:
        """获取最近的历史记录"""
        return self.memories[-limit:]

    def clear(self):
        """清空记忆（测试用）"""
        self.memories = []
        self._save()

    def __len__(self):
        return len(self.memories)


# =============================================
# 二、带记忆的 Agent
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
    def __init__(self):
        self.history: list[Message] = []
        self.trace_id = uuid.uuid4().hex[:12]

    def send(self, msg: Message):
        msg.trace_id = self.trace_id
        msg.timestamp = time.time()
        msg.turn = len(self.history) + 1
        self.history.append(msg)
        print(f"  [{msg.turn}] {msg.sender} \u2192 {msg.receiver}  [{msg.msg_type}]")


class MemoryAugmentedAgent:
    """
    带记忆增强的 Agent。
    核心流程：查记忆 → 行动 → 存记忆
    """

    def __init__(self, name: str, system_prompt: str, bus: MessageBus,
                 model: str = "deepseek-chat", temperature: float = 0.3):
        self.name = name
        self.system_prompt = system_prompt
        self.bus = bus
        self.model = model
        self.temperature = temperature
        self.memory = MemoryStore(name)
        # 保留旧记忆，让 Agent 能跨轮次学习

    def think(self, user_msg: str) -> str:
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=self.temperature,
            max_tokens=1024
        )
        return response.choices[0].message.content

    def retrieve_relevant_memories(self, task: str) -> str:
        """检索相关记忆，拼进 prompt 中"""
        relevant = self.memory.query(task, top_k=3)
        if not relevant:
            return ""

        context = "\n".join([
            f"- 任务：{m['task']}  |  结果：{m['outcome']}"
            for m in relevant
        ])
        return f"\n\n【相关历史记录】\n{context}\n请注意避免重复已犯过的错误。"

    def record_memory(self, task: str, outcome: str, issues: list[str] = None):
        """执行完后保存记忆"""
        self.memory.add({
            "task": task,
            "outcome": outcome[:200],
            "issues": issues or []
        })

    def send(self, receiver: str, msg_type: str, payload: dict):
        self.bus.send(Message(
            sender=self.name, receiver=receiver,
            msg_type=msg_type, payload=payload
        ))


class MemoryResearcher(MemoryAugmentedAgent):
    SYSTEM_PROMPT = """你是资深研究员。输出 JSON 格式：
{"key_points": ["..."], "confidence": 0-1}"""

    def execute(self, topic: str) -> str:
        memories = self.retrieve_relevant_memories(topic)
        prompt = f"研究主题：{topic}\n输出 JSON 格式研究报告。{memories}"
        result = self.think(prompt)
        self.record_memory(topic, result[:200])
        return result


class MemoryWriter(MemoryAugmentedAgent):
    SYSTEM_PROMPT = """你是科普写作者。基于研究资料创作文章，输出 JSON 格式：
{"title": "...", "content": "...", "word_count": 0}"""

    def execute(self, topic: str, research: str, revision_round: int = 0) -> str:
        # 查记忆：上次为什么被驳回
        memories = self.retrieve_relevant_memories(f"写作 {topic} 修改")
        prompt = f"主题：{topic}\n研究资料：{research}\n输出 JSON 格式文章。{memories}"
        if revision_round > 0:
            prompt += f"\n这是第 {revision_round} 次修改。请特别关注之前审核提出的问题。"
        result = self.think(prompt)
        self.record_memory(f"写作{topic}", result[:200])
        return result


class MemoryReviewer(MemoryAugmentedAgent):
    SYSTEM_PROMPT = """你是苛刻的内容审核员。输出 JSON 格式：
{"issues": [...], "rating": 1-5, "verdict": "通过/需要修改"}

你的评分标准：
- 5分：完美，几乎不用
- 4分：很好，但有改进空间
- 3分：一般，需要明显修改
- 2分：较差，大篇幅重写
- 1分：不合格

规则：
- 评分高于 4 才算通过
- 每次必须指出至少 2 个具体问题
- 如果只有比喻没有技术原理，评分不能超过 3

审稿策略：
- 第1次审稿：严格指出所有问题
- 后续审稿：重点检查之前的问题是否已改进
- 如果质量有明显提升，给 4 分放行，不需要追求完美"""

    def execute(self, article: str, research: str) -> str:
        memories = self.retrieve_relevant_memories("审核")
        prompt = f"审核文章：\n{article}\n\n参考：{research}\n\n输出 JSON 格式审核意见。{memories}"
        result = self.think(prompt)
        return result


# =============================================
# 三、带记忆的工作流
# =============================================

def run_with_memory(topic: str, max_retries: int = 2):
    """带记忆增强的多 Agent 工作流"""
    print(f"\n{'='*60}")
    print(f"【带记忆的工作流】主题：{topic}")
    print(f"{'='*60}\n")

    bus = MessageBus()
    researcher = MemoryResearcher("研究员", MemoryResearcher.SYSTEM_PROMPT, bus, temperature=0.1)
    writer = MemoryWriter("写作者", MemoryWriter.SYSTEM_PROMPT, bus, temperature=0.4)
    reviewer = MemoryReviewer("审核员", MemoryReviewer.SYSTEM_PROMPT, bus, temperature=0.1)

    start = time.time()
    research_result = ""
    article_result = ""

    # === 研究员 ===
    print(f"\u25b6 研究员工作中...")
    raw = researcher.execute(topic)
    try:
        research_json = json.loads(
            raw.strip().removeprefix("```json").removesuffix("```").strip()
        )
    except:
        research_json = {"key_points": [raw[:100]], "confidence": 0.5}
    research_result = raw
    researcher.send("写作者", "research_result", research_json)

    # === 写作 + 审核循环 ===
    previous_rating = 0  # 追踪上一轮评分，用于改进判定
    for attempt in range(1, max_retries + 2):
        print(f"\n\u25b6 写作者工作中（第 {attempt} 稿）...")
        raw_article = writer.execute(topic, research_result, attempt - 1)
        try:
            article_json = json.loads(
                raw_article.strip().removeprefix("```json").removesuffix("```").strip()
            )
        except:
            article_json = {"title": topic, "content": raw_article[:200], "word_count": len(raw_article)}
        article_result = raw_article
        writer.send("审核员", "article_draft", article_json)

        print(f"\u25b6 审核员审查中...")
        raw_review = reviewer.execute(article_result, research_result)
        try:
            review_json = json.loads(
                raw_review.strip().removeprefix("```json").removesuffix("```").strip()
            )
        except:
            review_json = {"issues": [], "rating": 3, "verdict": "需要修改"}

        current_rating = review_json.get("rating", 0)
        verdict = review_json.get("verdict", "需要修改")
        print(f"  评分：{current_rating}/5  |  裁决：{verdict}")
        if review_json.get("issues"):
            print(f"  意见：{review_json.get('issues')}")

        # 保存审核结果到审核员和写作者的记忆
        reviewer.record_memory(
            f"审核{topic}第{attempt}稿",
            f"评分{current_rating}，裁决{verdict}",
            review_json.get("issues", [])
        )
        writer.record_memory(
            f"写作{topic}第{attempt}稿",
            f"收到审核意见：{review_json.get('issues', [])}",
        )

        # === 通过条件：绝对达标 或 相对改进 ===
        passed = (verdict == "通过" or current_rating >= 4)

        # 相对改进：评分比上一轮高，且不低于 3 分，就放行
        if not passed and previous_rating > 0 and current_rating > previous_rating and current_rating >= 3:
            print(f"  ↳ 评分从 {previous_rating} 提升到 {current_rating}，改进明显，放行")
            passed = True

        # 最后一轮：如果评分没下降，且高于 2 分，也算通过（够好了）
        if not passed and attempt > max_retries and current_rating >= previous_rating and current_rating > 2:
            print(f"  ↳ 最后一轮评分 {current_rating}，未下降，放行")
            passed = True

        if passed:
            print(f"\n\u2713 审核通过！")
            break

        previous_rating = current_rating

        if attempt > max_retries:
            print(f"\n\u2716 超出最大重试次数")
            break

    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"\u2713 流程完成 (用时 {elapsed:.1f}s)")
    print(f"\n研究员记忆数量：{len(researcher.memory)}")
    print(f"写作者记忆数量：{len(writer.memory)}")
    print(f"审核员记忆数量：{len(reviewer.memory)}")

    print("\n审核员记忆内容：")
    for m in reviewer.memory.get_history():
        print(f"  - {m.get('task', '')}  |  {m.get('outcome', '')[:60]}...")

    try:
        final = json.loads(
            article_result.strip().removeprefix("```json").removesuffix("```").strip()
        )
        print(f"\n最终文章：{final.get('title', '')}")
        print(final.get('content', '')[:400])
    except:
        print(f"\n最终输出：{article_result[:400]}")

    return article_result


# =============================================
# 四、运行
# =============================================

if __name__ == "__main__":
    print("=" * 60)
    print("L7 Day 3: 持久化 Agent 记忆")
    print("=" * 60)

    topic = "状态机如何解决多Agent协作中的死锁问题"

    print(f"\n主题：{topic}")
    print("记忆文件将保存在 memory/ 目录下")
    print("每个 Agent 有独立的记忆文件")
    print("\n关键改进：审核员查记忆时能看到自己上次提过什么问题")
    print("不会再重复输出相同的反馈")

    run_with_memory(topic)
