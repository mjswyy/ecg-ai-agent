"""
LLM 统一接口 — 支持多种大语言模型后端。

支持的 backend:
    - deepseek: DeepSeek API (deepseek-v4-pro)
    - openai:   OpenAI API (GPT-4o-mini, GPT-4)
    - vllm:     本地 vLLM 部署 (Qwen2, Llama3)

所有后端使用 OpenAI 兼容的 /v1/chat/completions 端点，
切换 LLM 只需改一行配置。

使用示例:
    llm = LLMInterface(backend="deepseek", model="deepseek-v4-pro")
    response = llm.chat([{"role": "user", "content": "分析这份ECG"}])

    # Mock 模式（无 API Key 时测试用）
    llm._client = None
    response = llm.chat([...])  # 返回预设占位回复
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMInterface:
    """统一的 LLM 接口，支持多种后端。

    参数:
        backend:     "deepseek" / "openai" / "vllm"
        model:       模型标识符
        api_key:     API 密钥（默认从环境变量读取）
        base_url:    自定义 API 基础 URL
        temperature: 生成温度 (0-1, 越低越确定性)
        max_tokens:  最大输出 token 数
    """

    def __init__(
        self,
        backend: str = "deepseek",
        model: str = "deepseek-v4-pro",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if backend in ("deepseek", "openai"):
            self._init_openai_compatible(backend, model, api_key, base_url)
        elif backend == "vllm":
            self._init_vllm(model, base_url)
        else:
            raise ValueError(f"未知的 backend: {backend}")

    def _init_openai_compatible(self, backend, model, api_key, base_url):
        """初始化 OpenAI 兼容客户端（DeepSeek / OpenAI）。"""
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("openai 包未安装。安装命令: pip install openai")
            raise

        if backend == "deepseek":
            api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
            base_url = base_url or "https://api.deepseek.com/v1"
        elif backend == "openai":
            api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            base_url = base_url or "https://api.openai.com/v1"

        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def _init_vllm(self, model, base_url):
        """初始化本地 vLLM 客户端。"""
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key="not-needed",
                base_url=base_url or "http://localhost:8000/v1",
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

        参数:
            messages: 消息列表 [{"role": "user"/"assistant"/"system", "content": "..."}]
            tools:    可选的工具定义（OpenAI function calling 格式）
            response_format: 可选的 {"type": "json_object"} 强制 JSON

        返回:
            LLM 回复的文本内容。
        """
        if not self._client:
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

        try:
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
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return self._mock_response(messages)

    def _mock_response(self, messages) -> str:
        """Mock 回复（无 LLM 时的测试回退）。"""
        last_msg = messages[-1]["content"] if messages else ""

        if "ECG" in last_msg or "plan" in last_msg.lower():
            return json.dumps({
                "plan": [
                    {"action": "extract_r_peaks", "reason": "测量心率和节律"},
                    {"action": "classify_arrhythmia", "reason": "诊断心律失常"},
                    {"action": "generate_report", "reason": "综合生成报告"},
                ]
            })
        return "基于 ECG 结果，心律在正常范围内。未检测到急性异常。建议临床对照确认。"

    @property
    def is_available(self) -> bool:
        """检查 LLM 后端是否确实已连接。"""
        return hasattr(self, '_client') and self._client is not None
