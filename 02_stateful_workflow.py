"""
L7 Day 2: 状态驱动 Agent 工作流
==================================
Day 1 的 Agent 协作是线性流水线：研究员→写作者→审核员，一条路走到底。
但如果审核不通过怎么办？写作者改完怎么回来？

状态机的思路：每个 Agent 不是"被调用一次就结束"，而是有完整的生命周期。
状态机（State Machine）管理每一步该做什么，允许循环、分支、重试。

核心概念：
- 状态（State）：Agent 当前在做什么（IDLE / WORKING / REVIEW / REVISE / DONE）
- 转移（Transition）：什么条件下从一个状态跳到另一个
- 守卫（Guard）：转移前必须满足的条件（如最大重试次数）
"""

import json
import time
import uuid
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional
from openai import OpenAI

client = OpenAI()

# =============================================
# 一、Agent 生命周期状态
# =============================================

class AgentState(Enum):
    IDLE = "idle"            # 等待分配任务
    WORKING = "working"      # 正在执行任务
    REVIEW = "review"        # 等待被审核
    REVISE = "revise"        # 需要修改
    COMPLETED = "completed"  # 任务完成
    FAILED = "failed"        # 超出重试次数


class WorkflowState(Enum):
    """整个工作流的状态"""
    INIT = "init"
    RESEARCHING = "researching"
    WRITING = "writing"
    REVIEWING = "reviewing"
    FINALIZED = "finalized"
    ABORTED = "aborted"


# =============================================
# 二、工作流引擎（状态机）
# =============================================

class WorkflowEngine:
    """
    状态机核心。
    管理整个 Multi-Agent 工作流的生命周期和状态转移。
    """

    def __init__(self, max_retries: int = 2):
        self.state = WorkflowState.INIT
        self.max_retries = max_retries
        self.retry_count = 0
        self.transition_log: list[dict] = []
        self.workflow_id = uuid.uuid4().hex[:8]

    def transition(self, to_state: WorkflowState, reason: str = "") -> bool:
        """尝试转移到下一个状态"""
        from_state = self.state

        # 守卫：检查是否允许转移
        if not self._can_transition(from_state, to_state):
            print(f"  [引擎] 不允许转移: {from_state.value} → {to_state.value}")
            return False

        # 守卫：超过最大重试次数则中止
        if to_state == WorkflowState.REVIEWING and self.retry_count >= self.max_retries:
            print(f"  [引擎] 超出最大重试次数 ({self.max_retries})，中止流程")
            self.state = WorkflowState.ABORTED
            self._log_transition(from_state, WorkflowState.ABORTED, "超出最大重试次数")
            return True

        self.state = to_state
        self._log_transition(from_state, to_state, reason)
        print(f"  [引擎] {from_state.value} → {to_state.value}  {reason}")
        return True

    def _can_transition(self, from_state: WorkflowState, to_state: WorkflowState) -> bool:
        """状态转移规则表"""
        allowed = {
            WorkflowState.INIT:        {WorkflowState.RESEARCHING},
            WorkflowState.RESEARCHING: {WorkflowState.WRITING},
            WorkflowState.WRITING:     {WorkflowState.REVIEWING},
            WorkflowState.REVIEWING:   {WorkflowState.FINALIZED, WorkflowState.WRITING},
            WorkflowState.FINALIZED:   set(),
            WorkflowState.ABORTED:     set(),
        }
        return to_state in allowed.get(from_state, set())

    def _log_transition(self, _from, _to, reason):
        self.transition_log.append({
            "from": _from.value, "to": _to.value,
            "reason": reason, "timestamp": time.time()
        })

    def needs_revision(self) -> bool:
        """审核不通过，打回修改"""
        self.retry_count += 1
        return self.transition(WorkflowState.WRITING, f"第 {self.retry_count} 次修改")

    def summary(self) -> str:
        """生成状态转移摘要"""
        lines = ["状态转移记录："]
        for t in self.transition_log:
            lines.append(f"  {t['from']} → {t['to']}: {t['reason']}")
        lines.append(f"最终状态: {self.state.value}")
        lines.append(f"重试次数: {self.retry_count}")
        return "\n".join(lines)


# =============================================
# 三、消息总线（复用 Day 1）
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

    def to_llm_context(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class MessageBus:
    def __init__(self):
        self.history: list[Message] = []
        self.trace_id = uuid.uuid4().hex[:12]

    def send(self, msg: Message):
        msg.trace_id = self.trace_id
        msg.timestamp = time.time()
        msg.turn = len(self.history) + 1
        self.history.append(msg)
        print(f"  [{msg.turn}] {msg.sender} → {msg.receiver}  [{msg.msg_type}]")


# =============================================
# 四、带状态的 Agent
# =============================================

class StatefulAgent:
    """支持状态转移的 Agent"""

    def __init__(self, name: str, system_prompt: str, bus: MessageBus,
                 model: str = "deepseek-chat", temperature: float = 0.3):
        self.name = name
        self.system_prompt = system_prompt
        self.bus = bus
        self.model = model
        self.temperature = temperature
        self.state = AgentState.IDLE
        self.memory: list[dict] = []

    def think(self, user_msg: str) -> str:
        self.memory.append({"role": "user", "content": user_msg})
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                *self.memory[-10:]
            ],
            temperature=self.temperature,
            max_tokens=1024
        )
        reply = response.choices[0].message.content
        self.memory.append({"role": "assistant", "content": reply})
        return reply

    def send(self, receiver: str, msg_type: str, payload: dict):
        msg = Message(
            sender=self.name, receiver=receiver,
            msg_type=msg_type, payload=payload
        )
        self.bus.send(msg)


class ResearcherAgent(StatefulAgent):
    SYSTEM_PROMPT = """你是资深研究员，输出 JSON 格式研究结果。
包含 key_points (list[str], 3-5个要点)、confidence (float, 0-1)。

规则：
1. 只输出事实性信息
2. 每条 2 句话以内
3. 不确定就降低 confidence"""

    def execute(self, topic: str) -> str:
        self.state = AgentState.WORKING
        result = self.think(
            f"研究主题：{topic}\n输出 JSON 格式研究报告。\n"
            f"之前的研究记录：{[m.payload for m in self.bus.history if m.msg_type == 'research_result']}"
        )
        self.state = AgentState.COMPLETED
        return result


class WriterAgent(StatefulAgent):
    SYSTEM_PROMPT = """你是科普写作者。基于研究资料写 300 字以内的科普文章。
输出 JSON 格式：title (str)、content (str)、word_count (int)。

规则：
1. 用比喻让概念容易理解
2. 选 2-3 个核心要点展开，不要全部覆盖
3. 如果这是修改版，标注每次改动的内容"""

    def execute(self, topic: str, research: str, revision_round: int = 0) -> str:
        self.state = AgentState.WORKING
        prompt = f"主题：{topic}\n研究资料：{research}\n输出 JSON 格式文章。"
        if revision_round > 0:
            prompt += f"\n这是第 {revision_round} 次修改。请标注修改的部分。"
        result = self.think(prompt)
        self.state = AgentState.REVIEW
        return result


class ReviewerAgent(StatefulAgent):
    SYSTEM_PROMPT = """你是严格的内容审核员。输出 JSON 格式：
issues (list[str]、rating (int 1-5)、verdict (str: 通过/需要修改)。

如果评分 < 4，verdict 必须写"需要修改"，并给出具体的修改方向。"""

    def execute(self, article: str, research: str) -> str:
        self.state = AgentState.WORKING
        result = self.think(
            f"审核以下文章：\n{article}\n\n参考研究资料：{research}\n\n输出 JSON 格式的审核意见。"
        )
        self.state = AgentState.COMPLETED
        return result


# =============================================
# 五、编排器（带状态机决策）
# =============================================

def run_stateful_workflow(topic: str):
    """用状态机驱动的完整工作流"""
    print(f"\n{'='*60}")
    print(f"【状态驱动工作流】主题：{topic}")
    print(f"{'='*60}\n")

    bus = MessageBus()
    engine = WorkflowEngine(max_retries=2)

    researcher = ResearcherAgent("研究员", ResearcherAgent.SYSTEM_PROMPT, bus, temperature=0.1)
    writer = WriterAgent("写作者", WriterAgent.SYSTEM_PROMPT, bus, temperature=0.4)
    reviewer = ReviewerAgent("审核员", ReviewerAgent.SYSTEM_PROMPT, bus, temperature=0.1)

    start = time.time()
    research_result = ""
    article_result = ""

    # === 状态机循环 ===
    while engine.state not in (WorkflowState.FINALIZED, WorkflowState.ABORTED):

        if engine.state == WorkflowState.INIT:
            engine.transition(WorkflowState.RESEARCHING, "启动研究")

        elif engine.state == WorkflowState.RESEARCHING:
            print(f"\n▶ 研究员工作中...")
            raw = researcher.execute(topic)
            try:
                research_json = json.loads(
                    raw.strip().removeprefix("```json").removesuffix("```").strip()
                )
            except:
                research_json = {"key_points": [raw[:100]], "confidence": 0.5}
            research_result = raw

            researcher.send("写作者", "research_result", research_json)
            engine.transition(WorkflowState.WRITING, "研究完成")

        elif engine.state == WorkflowState.WRITING:
            print(f"\n▶ 写作者工作中 (第 {engine.retry_count + 1} 稿)...")
            raw = writer.execute(topic, research_result, engine.retry_count)
            try:
                article_json = json.loads(
                    raw.strip().removeprefix("```json").removesuffix("```").strip()
                )
            except:
                article_json = {"title": topic, "content": raw[:200], "word_count": len(raw)}
            article_result = raw

            writer.send("审核员", "article_draft", article_json)
            engine.transition(WorkflowState.REVIEWING, "写作完成，提交审核")

        elif engine.state == WorkflowState.REVIEWING:
            print(f"\n▶ 审核员审查中...")
            raw = reviewer.execute(article_result, research_result)
            try:
                review_json = json.loads(
                    raw.strip().removeprefix("```json").removesuffix("```").strip()
                )
            except:
                review_json = {"issues": [], "rating": 3, "verdict": "需要修改"}

            print(f"  评分：{review_json.get('rating', 'N/A')}/5")
            print(f"  裁决：{review_json.get('verdict', 'N/A')}")
            if review_json.get("issues"):
                print(f"  意见：{review_json.get('issues')}")

            reviewer.send("协调者", "review", review_json)

            if review_json.get("verdict") == "通过" or review_json.get("rating", 0) >= 4:
                engine.transition(WorkflowState.FINALIZED, "审核通过")
            else:
                if not engine.needs_revision():
                    # 打回修改
                    pass

    elapsed = time.time() - start

    # === 输出结果 ===
    print(f"\n{'='*60}")
    print(f"✓ 流程结束 (用时 {elapsed:.1f}s)")
    print(engine.summary())

    try:
        final_article = json.loads(
            article_result.strip().removeprefix("```json").removesuffix("```").strip()
        )
        print(f"\n最终文章：{final_article.get('title', '')}")
        print(final_article.get('content', '')[:400])
    except:
        print(f"\n最终输出：{article_result[:400]}")

    return engine.state.value


# =============================================
# 六、对比：线性流水线 vs 状态机
# =============================================

def run_linear_pipeline(topic: str):
    """线性流水线（Day 1 方式，没有状态机）"""
    print(f"\n{'='*60}")
    print(f"【线性流水线（无状态机对比）】主题：{topic}")
    print(f"{'='*60}\n")

    bus = MessageBus()
    researcher = ResearcherAgent("研究员", ResearcherAgent.SYSTEM_PROMPT, bus, temperature=0.1)
    writer = WriterAgent("写作者", WriterAgent.SYSTEM_PROMPT, bus, temperature=0.4)
    reviewer = ReviewerAgent("审核员", ReviewerAgent.SYSTEM_PROMPT, bus, temperature=0.1)

    start = time.time()

    # 一路走到底，不回头看
    raw_research = researcher.execute(topic)
    try:
        research_json = json.loads(raw_research.strip().removeprefix("```json").removesuffix("```").strip())
    except:
        research_json = {"key_points": [raw_research[:100]], "confidence": 0.5}
    researcher.send("写作者", "research_result", research_json)

    raw_article = writer.execute(topic, raw_research)
    try:
        article_json = json.loads(raw_article.strip().removeprefix("```json").removesuffix("```").strip())
    except:
        article_json = {"title": topic, "content": raw_article[:200], "word_count": len(raw_article)}
    writer.send("审核员", "article_draft", article_json)

    raw_review = reviewer.execute(raw_article, raw_research)
    try:
        review_json = json.loads(raw_review.strip().removeprefix("```json").removesuffix("```").strip())
    except:
        review_json = {"issues": [], "rating": 3, "verdict": "需要修改"}

    elapsed = time.time() - start
    print(f"\n✓ 线性流水线完成 (用时 {elapsed:.1f}s)")
    print(f"  评分：{review_json.get('rating', 'N/A')}/5")
    print(f"  裁决：{review_json.get('verdict', 'N/A')}")

    # 线性流水线的致命问题：审核不通过不会重试
    if review_json.get("verdict") == "需要修改":
        print(f"  !!! 审核不通过，但线性流水线不会重试 !!!")
        print(f"  !!! 有问题的文章直接输出了 !!!")

    return review_json


# =============================================
# 七、运行
# =============================================

if __name__ == "__main__":
    topic = "状态机在 Multi-Agent 系统中的应用"

    print("=" * 60)
    print("L7 Day 2: 状态驱动 Agent 工作流")
    print("=" * 60)

    # 先跑无状态机的看看问题在哪
    linear_review = run_linear_pipeline(topic)

    # 再跑有状态机的
    final_state = run_stateful_workflow(topic)

    print(f"\n{'='*60}")
    print("核心差异：")
    print("- 线性流水线：审核不通过就直接输出问题了，没有补救机制")
    print(f"- 状态机工作流：审核不通过可打回修改（最多{2}次），最终状态={final_state}")
    print("- 状态转移都被记录在 transition_log 中，可回溯每个决策点")
