"""
LLM 统一接口 — 支持多种大语言模型后端。

支持的 backend:
    - deepseek: DeepSeek API (deepseek-chat / deepseek-v4-pro)
    - zhipu:    智谱 BigModel API (glm-4.5-air 等，OpenAI 兼容)
    - dashscope: 阿里云百炼 DashScope (qwen3.5-flash 等，OpenAI 兼容)
    - openai:   OpenAI API (GPT-4o-mini, GPT-4)
    - vllm:     本地 vLLM 部署 (Qwen2, Llama3)

所有后端使用 OpenAI 兼容的 /v1/chat/completions 端点，
切换 LLM 只需改一行配置。

使用示例:
    llm = LLMInterface(backend="dashscope", model="qwen3.5-flash")
    response = llm.chat([{"role": "user", "content": "分析这份ECG"}])

    # Mock 模式（无 API Key 时测试用，显式开启）
    llm = LLMInterface(backend="dashscope", mock=True)
    response = llm.chat([...])  # 返回预设占位回复，llm.mock_mode == True

检查报告 1.9 修复：
    - 旧版任何 API 异常都静默降级为 mock 占位文本（医疗场景最危险的
      "看起来正常、实则全是占位符"模式）→ 新版默认异常向上抛，由调用方决策；
      仅显式 mock=True（或环境变量 AGENT_MOCK=1）时返回占位回复，且
      mock_mode 置 True 供调用方显著提示。
    - API key 缺失/为空不再构造必然失败的客户端，而是显式不可用
      （is_available=False），把配置错误与运行时错误分开。
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
    """从候选 .env 文件加载环境变量（遍历全部候选并合并，不覆盖已存在值）。"""
    candidates = [
        Path(__file__).parent.parent.parent.parent / ".env",  # ecg-ai-agent/.env
        Path.cwd() / ".env",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                # 兼容 "export KEY=..." 前缀
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            logger.info(f"已加载环境变量文件: {p}")
        except Exception as e:
            logger.warning(f".env 读取失败: {e}")


_load_dotenv()


class LLMInterface:
    """统一的 LLM 接口，支持多种后端。

    参数:
        backend:     "deepseek" / "openai" / "vllm"
        model:       模型标识符
        api_key:     API 密钥（默认从环境变量读取）
        base_url:    自定义 API 基础 URL
        temperature: 生成温度 (0-1, 越低越确定性)
        max_tokens:  最大输出 token 数
        mock:        显式 mock 模式（占位回复，mock_mode=True）
        timeout:     请求超时（秒）
        max_retries: SDK 自动重试次数
    """

    def __init__(
        self,
        backend: str = "deepseek",
        model: str = "deepseek-v4-pro",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        mock: bool = False,
        timeout: float = 120.0,
        max_retries: int = 2,
    ):
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.mock_mode = bool(mock) or os.environ.get("AGENT_MOCK", "") == "1"

        if backend in ("deepseek", "openai", "zhipu", "dashscope"):
            self._init_openai_compatible(backend, model, api_key, base_url)
        elif backend == "vllm":
            self._init_vllm(model, base_url)
        else:
            raise ValueError(f"未知的 backend: {backend}")

    def _init_openai_compatible(self, backend, model, api_key, base_url):
        """初始化 OpenAI 兼容客户端（DeepSeek / OpenAI）。"""
        # 4A/4J 审查修复：💡-1 mock 模式尽早跳过第三方 client 构造与 import
        # （旧版 mock=True 且无 key 时仍先 import openai，未安装即 ImportError，
        # 抬高 mock 测试的环境依赖）
        if self.mock_mode:
            logger.warning(f"{backend} 进入显式 mock 模式，跳过 openai 客户端初始化")
            self._client = None
            return
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("openai 包未安装。安装命令: pip install openai")
            raise

        if backend == "deepseek":
            api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
            base_url = base_url or "https://api.deepseek.com/v1"
        elif backend == "zhipu":
            # 智谱 BigModel（OpenAI 兼容端点）：glm-4.5-air / glm-4.5 / glm-4-flash
            api_key = api_key or os.environ.get("ZHIPU_API_KEY", "")
            base_url = base_url or "https://open.bigmodel.cn/api/paas/v4"
        elif backend == "dashscope":
            # 阿里云百炼 DashScope（OpenAI 兼容模式）：qwen3.5-flash / qwen-max 等
            api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
            base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        elif backend == "openai":
            api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            base_url = base_url or "https://api.openai.com/v1"

        # 检查报告 1.9 修复：key 缺失/为空 → 显式不可用（不构造必然失败的客户端）
        if not api_key:
            if self.mock_mode:
                logger.warning(f"{backend.upper()} API key 缺失，进入显式 mock 模式")
                self._client = None
                return
            self._client = None
            logger.error(f"缺少 {backend.upper()}_API_KEY（.env 或环境变量）")
            return

        self._client = OpenAI(
            api_key=api_key, base_url=base_url,
            timeout=self.timeout, max_retries=self.max_retries,
        )

    def _init_vllm(self, model, base_url):
        """初始化本地 vLLM 客户端。"""
        # 4A/4J 审查修复：💡-1 vLLM 后端同样尊重 mock（旧版无视 mock_mode，
        # 直接构造真实客户端并 import openai）
        if self.mock_mode:
            logger.warning("vLLM 后端进入显式 mock 模式，跳过客户端初始化")
            self._client = None
            return
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key="not-needed",
                base_url=base_url or "http://localhost:8000/v1",
                timeout=self.timeout, max_retries=self.max_retries,
            )
        except ImportError:
            logger.error("vLLM 客户端需要 openai 包")
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        response_format: Optional[Dict] = None,
    ) -> str:
        """发送聊天补全请求。

        检查报告 1.9 修复：API 异常不再静默降级为占位文本——mock 模式外
        一律向上抛，由调用方决策（评测中断、演示显式报"模型未连接"）。

        返回:
            LLM 回复的文本内容。
        """
        if self.mock_mode or not self._client:
            if not self.mock_mode:
                raise RuntimeError(
                    "LLM 不可用（缺少 API key 且未显式启用 mock）；"
                    "设置 DEEPSEEK_API_KEY 或构造 LLMInterface(mock=True)")
            logger.warning("LLM mock 模式：返回占位回复（非真实模型输出）")
            return self._mock_response(messages)

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if response_format:
            kwargs["response_format"] = response_format

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        # 如果 LLM 返回了工具调用请求
        if choice.message.tool_calls:
            tool_calls = [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in choice.message.tool_calls
            ]
            return json.dumps({"tool_calls": tool_calls})

        return choice.message.content or ""

    def _mock_response(self, messages) -> str:
        """Mock 回复（显式 mock 模式下的占位输出）。

        4A/4J 审查修复：🟡-1 按提示词语义分派回复类型（旧版以 "ECG" 子串为
        开关，planner/reasoner/reflector/judge 的 prompt 全含 "ECG"，导致
        judge/reflector 收到计划 JSON、mock 测试口径失真）。
        分派依据各调用方 prompt 的唯一特征词：
          - 判官（RubricJudge）：含 correctness/grounding 评分维度 → 评分 JSON
          - 规划器（Planner）：含 "plan"/"计划" → 计划 JSON
          - 反射器（Reflector）：含 continue/revise/complete 指令 → "continue"
          - 推理器（Reasoner）及其他：通用占位文本
        """
        last_msg = messages[-1]["content"] if messages else ""

        # 判官（唯一含评分维度词）：返回三维评分 JSON（满足 RubricJudge 校验）
        if "correctness" in last_msg or "grounding" in last_msg or "rubric" in last_msg.lower():
            return json.dumps({
                "correctness": 4, "completeness": 4, "grounding": 4,
                "total": 4, "comments": "mock 占位评分（非真实模型输出）",
            })

        # 报告/推理器（含双层结构特征词）——复检D（🟡-3）：必须在 planner 分支
        # 之前；KB 注入文本可能含 "plan" 子串（treatment plan 等）被 planner
        # 抢先命中，mock 报告层恒为空（smoke_demo status="分析完成" 但报告空）
        if "risk_assessment" in last_msg or "双层" in last_msg or "findings" in last_msg:
            return json.dumps({
                "findings": [{"name": "Sinus Rhythm",
                              "evidence": "mock 占位（非真实模型输出）"}],
                "risk_assessment": {"level": "low", "rationale": "mock 占位"},
                "recommendations": {"urgency": "routine",
                                    "actions": ["mock 占位：结合临床复核"]},
            })

        # 反射器（含 continue/revise/complete 指令）——4A 定向复查（γ）：必须
        # 排在 planner 分支之前，否则反射器 prompt 中的"如果计划需要调整"的
        # "计划"被 planner 抢先命中，mock 反射器恒收到计划 JSON。
        # 复检2 修复（🟡-2）：判据收敛为 "continue"（反射器专属词——
        # planner.revise_plan 的 feedback 含 "revise" 会被宽泛判据误抢，
        # 实测 revise_plan 收到 "continue" 而非计划 JSON）
        if "continue" in last_msg.lower():
            return "continue"

        # 规划器（唯一含 "plan"/"计划"）
        if "plan" in last_msg.lower() or "计划" in last_msg:
            # 4A/4J 审查修复：🟡-2 移除 generate_report（已从默认计划/注册表/
            # _is_critical 统一移除）；💡-2 本清单为 mock 模式占位计划，
            # 动作名与默认回退计划一致（不强行引用 ToolRegistry 单一事实源）。
            return json.dumps({
                "plan": [
                    {"action": "extract_r_peaks", "reason": "测量心率和节律"},
                    {"action": "classify_arrhythmia", "reason": "诊断心律失常"},
                ]
            })

        return "【占位回复 · LLM 未连接】基于 ECG 结果，心律在正常范围内。未检测到急性异常。建议临床对照确认。"

    @property
    def is_available(self) -> bool:
        """检查 LLM 后端是否确实可用（非 mock 且客户端已构造）。"""
        return (not self.mock_mode) and hasattr(self, '_client') and self._client is not None
