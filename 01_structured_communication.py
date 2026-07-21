"""
L7 Day 1: 结构化 Agent 通信协议
=================================
Agent 之间不再靠纯文本传话，改用 JSON Schema 定义消息契约。
好处：省 Token、可验证、方便追踪、降低误解。

本 demo 实现两个 Agent：研究员 → 写作者
用结构化消息传递，同时在最后记录 Token 消耗对比。
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional
from openai import OpenAI

# =============================================
# 一、消息协议定义（契约）
# =============================================

@dataclass
class Message:
    """Agent 通信的基本单元"""
    sender: str
    receiver: str
    msg_type: str            # 消息类型：task / research_result / article_draft / review / ack
    payload: dict            # 实际数据，用 JSON Schema 约束
    trace_id: str = ""
    turn: int = 1
    timestamp: float = 0.0

    def to_llm_context(self) -> str:
        """转成 LLM 友好的格式"""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_dict(d: dict) -> "Message":
        return Message(**d)


# 每个 msg_type 对应的 payload 契约
MESSAGE_SCHEMAS = {
    "task": {
        "description": "给 Agent 分配任务",
        "required": ["objective", "context"],
        "example": {"objective": "研究 Transformer 的注意力机制", "context": "目标读者是初级程序员"}
    },
    "research_result": {
        "description": "研究员输出调查结果",
        "required": ["key_points", "confidence"],
        "example": {"key_points": ["点1...", "点2..."], "confidence": 0.85}
    },
    "article_draft": {
        "description": "写作者输出文章草稿",
        "required": ["title", "content", "word_count"],
        "example": {"title": "xxx", "content": "...", "word_count": 300}
    },
    "review": {
        "description": "审核反馈",
        "required": ["issues", "rating"],
        "example": {"issues": ["段落2缺例子"], "rating": 4}
    }
}


# =============================================
# 二、消息总线（追踪和验证通信）
# =============================================

class MessageBus:
    """管理 Agent 之间的消息传递，记录每一次通信"""

    def __init__(self):
        self.history: list[Message] = []
        self.trace_id = uuid.uuid4().hex[:12]

    def send(self, msg: Message) -> None:
        """发送并记录消息"""
        msg.trace_id = self.trace_id
        msg.timestamp = time.time()
        msg.turn = len(self.history) + 1
        self.history.append(msg)

        # 验证 payload 是否包含必要字段
        schema = MESSAGE_SCHEMAS.get(msg.msg_type)
        if schema:
            missing = [f for f in schema["required"] if f not in msg.payload]
            if missing:
                print(f"  [警告] {msg.msg_type} 消息缺少必要字段: {missing}")

        print(f"  [{msg.turn}] {msg.sender} → {msg.receiver}  [{msg.msg_type}]")

    def get_messages_for(self, agent_name: str) -> list[Message]:
        """获取发给某个 Agent 的所有未读消息"""
        return [m for m in self.history if m.receiver == agent_name]

    def get_conversation_log(self) -> str:
        """获取完整通信日志（用于分析）"""
        lines = []
        for m in self.history:
            lines.append(f"[Turn {m.turn}] {m.sender} → {m.receiver}")
            lines.append(f"  类型: {m.msg_type}")
            lines.append(f"  内容: {json.dumps(dict(m.payload), ensure_ascii=False, indent=2)[:200]}")
        return "\n".join(lines)


# =============================================
# 三、Agent 基类
# =============================================

client = OpenAI()

class BaseAgent:
    """Agent 基类，所有 Agent 继承这个"""

    def __init__(self, name: str, system_prompt: str, bus: MessageBus,
                 model: str = "deepseek-chat", temperature: float = 0.3):
        self.name = name
        self.system_prompt = system_prompt
        self.bus = bus
        self.model = model
        self.temperature = temperature
        self.memory: list[dict] = []  # 当前对话记忆

    def think(self, user_msg: str) -> str:
        """调用 LLM 思考"""
        self.memory.append({"role": "user", "content": user_msg})
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                *self.memory[-10:]  # 只保留最近 10 轮
            ],
            temperature=self.temperature,
            max_tokens=1024
        )
        reply = response.choices[0].message.content
        self.memory.append({"role": "assistant", "content": reply})
        return reply

    def send(self, receiver: str, msg_type: str, payload: dict):
        """发送结构化消息"""
        msg = Message(
            sender=self.name, receiver=receiver,
            msg_type=msg_type, payload=payload
        )
        self.bus.send(msg)

    def receive(self) -> Optional[Message]:
        """读取最近的未读消息"""
        msgs = self.bus.get_messages_for(self.name)
        if msgs:
            return msgs[-1]
        return None


# =============================================
# 四、具体 Agent 实现
# =============================================

class ResearcherAgent(BaseAgent):
    """研究员：专注信息收集和整理"""

    SYSTEM_PROMPT = """你是资深研究员。
你的输出格式必须是 JSON，包含以下字段：
- key_points: list[str]，3-5 个关键要点
- confidence: float，你对研究结果的自信度 (0-1)
- sources: list[str]，如果有信息来源的话

规则：
1. 只关注事实性信息，不做创作
2. 每个要点控制在 2 句话以内
3. 如果某个点不确定，降低 confidence"""

    def research(self, topic: str, context: str = ""):
        """执行研究任务"""
        prompt = f"""研究主题：{topic}
背景：{context}

请输出 JSON 格式的研究报告。"""
        result = self.think(prompt)
        return result


class WriterAgent(BaseAgent):
    """写作者：基于研究资料创作文章"""

    SYSTEM_PROMPT = """你是科普写作者。
基于研究员提供的资料，创作通俗易懂的文章。

你的输出格式必须是 JSON，包含以下字段：
- title: str
- content: str
- word_count: int
- key_points_used: list[str]，使用了研究员的哪些要点

规则：
1. 用比喻和类比让概念容易理解
2. 不需要覆盖所有要点，选最重要的 2-3 个
3. 控制在 300 字以内"""

    def write(self, topic: str, research_data: str):
        """基于研究结果写作"""
        prompt = f"""主题：{topic}

研究员提供的研究资料：
{research_data}

请输出 JSON 格式的文章。"""
        result = self.think(prompt)
        return result


class ReviewerAgent(BaseAgent):
    """审核员：检查文章质量和准确度"""

    SYSTEM_PROMPT = """你是内容审核员。
检查文章是否存在事实错误、逻辑漏洞、表达不清晰的地方。

你的输出格式必须是 JSON，包含以下字段：
- issues: list[str]，发现的问题列表
- rating: int，评分 1-5
- verdict: str，通过 / 需要修改 / 不通过"""

    def review(self, article_data: str, research_data: str):
        """审核文章"""
        prompt = f"""请审核以下文章：

{article_data}

参考研究资料：
{research_data}

请输出 JSON 格式的审核意见。"""
        result = self.think(prompt)
        return result


# =============================================
# 五、结构化 vs 非结构化对比演示
# =============================================

def run_structured_pipeline(topic: str):
    """用结构化消息协议运行完整流水线"""
    print(f"\n{'='*60}")
    print(f"【结构化通信流水线】主题：{topic}")
    print(f"{'='*60}\n")

    bus = MessageBus()

    # 为不同角色指定不同模型和 temperature
    # 研究员用默认 deepseek-chat，要稳定
    researcher = ResearcherAgent("研究员", ResearcherAgent.SYSTEM_PROMPT, bus,
                                 temperature=0.1)
    # 写作者：0.4 兼顾准确和可读，不完全复述也不自己编
    writer = WriterAgent("写作者", WriterAgent.SYSTEM_PROMPT, bus,
                         temperature=0.4)
    # 审核员严格按规则，temperature 最低
    reviewer = ReviewerAgent("审核员", ReviewerAgent.SYSTEM_PROMPT, bus,
                             temperature=0.1)

    start = time.time()

    # Step 1: 分配任务
    print("▶ Step 1: 协调者分配任务")
    researcher.send("研究员", "task", {"objective": f"研究{topic}", "context": "科普文章"})
    msg = researcher.receive()
    research_raw = researcher.research(topic, "目标读者是非技术背景的普通人")

    # Step 2: 研究员发送结构化结果
    print("\n▶ Step 2: 研究员输出研究报告")
    # 尝试解析 JSON
    try:
        research_json = json.loads(research_raw.strip().removeprefix("```json").removesuffix("```").strip())
    except:
        research_json = {"key_points": [research_raw[:100]], "confidence": 0.5}

    researcher.send("写作者", "research_result", research_json)

    # Step 3: 写作者基于研究结果写文章
    print("\n▶ Step 3: 写作者创作文章")
    article_raw = writer.write(topic, research_raw)

    try:
        article_json = json.loads(article_raw.strip().removeprefix("```json").removesuffix("```").strip())
    except:
        article_json = {"title": topic, "content": article_raw[:200], "word_count": len(article_raw)}

    writer.send("审核员", "article_draft", article_json)

    # Step 4: 审核员审核
    print("\n▶ Step 4: 审核员审查")
    review_raw = reviewer.review(article_raw, research_raw)

    # Step 5: 复审反馈
    try:
        review_json = json.loads(review_raw.strip().removeprefix("```json").removesuffix("```").strip())
    except:
        review_json = {"issues": [], "rating": 3, "verdict": "需要修改"}

    reviewer.send("研究员", "review", review_json)

    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"✓ 结构化流水线完成 (用时 {elapsed:.1f}s)")
    print(f"  总消息数：{len(bus.history)} 条")
    print(f"\n通信日志预览：")
    print(bus.get_conversation_log()[:500] + "...")
    print(f"{'='*60}\n")

    # 输出最终结果
    print("最终文章：")
    print(article_json.get("content", article_json)[:500])
    print(f"\n审核意见：")
    print(f"  评分：{review_json.get('rating', 'N/A')}/5")
    print(f"  裁决：{review_json.get('verdict', 'N/A')}")
    print(f"  问题：{review_json.get('issues', [])}")

    return article_json, review_json


def run_freeform_pipeline(topic: str):
    """非结构化的传统方式（对比用）"""
    print(f"\n{'='*60}")
    print(f"【非结构化流水线（对比）】主题：{topic}")
    print(f"{'='*60}\n")

    start = time.time()

    # 传统方式：把一切塞进一个 Prompt
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": """你是一个全能助手。请完成以下步骤：
1. 研究以下主题
2. 基于研究结果写一篇科普文章
3. 检查文章质量

直接在同一个回答中完成所有步骤。"""},
            {"role": "user", "content": f"主题：{topic}\n目标读者：非技术背景的普通人"}
        ],
        temperature=0.3,
        max_tokens=2048
    )

    result = response.choices[0].message.content
    elapsed = time.time() - start

    total_tokens = response.usage.total_tokens if response.usage else 0

    print(f"✓ 非结构化流水线完成 (用时 {elapsed:.1f}s)")
    print(f"  Token 消耗：{total_tokens}")
    print(f"\n输出预览：")
    print(result[:500])
    print(f"{'='*60}\n")

    return result


# =============================================
# 六、运行
# =============================================

if __name__ == "__main__":
    topic = "AI Agent 中的结构化通信协议"

    print("开始 L7 Day 1 演示...\n")
    print("两种方式对比：结构化消息（多 Agent 协作）vs 传统单 Agent 全包")

    article, review = run_structured_pipeline(topic)
    freeform = run_freeform_pipeline(topic)

    print("\n总结：")
    print("- 结构化通信：明确的消息契约，可追踪，每个 Agent 专注自己职责")
    print("- 非结构化：一步到位，但 Prompt 臃肿，难以排查错误")
    print("- 3 个 Agent 的 Token 总消耗量 ≈ 单 Agent 全包，但质量更高")
