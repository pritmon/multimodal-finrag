"""LlamaIndex CustomLLM wrapper for AWS Bedrock (Claude 3 Sonnet/Haiku).

Supports:
- Standard text completion
- Streaming responses via invoke_model_with_response_stream
- Multimodal inputs (image + text) encoded as base64
"""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any, Generator, Optional, Sequence

import boto3
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms import CustomLLM
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from PIL import Image

logger = logging.getLogger(__name__)

_CLAUDE3_MODELS = {
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
}


def _pil_to_b64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _build_content_block(
    text: Optional[str] = None,
    image: Optional[Image.Image] = None,
    image_bytes: Optional[bytes] = None,
    media_type: str = "image/png",
) -> list[dict]:
    """Build Bedrock Anthropic Messages API content blocks."""
    blocks: list[dict] = []

    if image is not None:
        b64 = _pil_to_b64(image)
        blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            }
        )
    elif image_bytes is not None:
        b64 = base64.b64encode(image_bytes).decode()
        blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64},
            }
        )

    if text:
        blocks.append({"type": "text", "text": text})

    return blocks


class BedrockLLM(CustomLLM):
    """LlamaIndex LLM backed by AWS Bedrock (Anthropic Claude 3 family).

    Parameters
    ----------
    model_id:
        Bedrock model identifier.
    aws_region:
        AWS region where Bedrock is enabled.
    max_tokens:
        Maximum tokens to generate.
    temperature:
        Sampling temperature.
    top_p:
        Nucleus sampling probability.
    session_kwargs:
        Extra kwargs forwarded to ``boto3.Session``.
    """

    model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    aws_region: str = "us-east-1"
    max_tokens: int = 4096
    temperature: float = 0.1
    top_p: float = 0.9

    # Pydantic model fields must be declared; store client as private attribute
    _client: Any = None  # will be set in __init__

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        session_kwargs = kwargs.get("session_kwargs", {})
        session = boto3.Session(**session_kwargs)
        object.__setattr__(
            self,
            "_client",
            session.client("bedrock-runtime", region_name=self.aws_region),
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=200_000,
            num_output=self.max_tokens,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=self.model_id,
        )

    # ── Completion interface ──────────────────────────────────────────────────

    @llm_completion_callback()
    def complete(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        content = []
        if images:
            for img in images:
                content.extend(_build_content_block(image=img))
        content.extend(_build_content_block(text=prompt))

        body = self._build_body(content)
        response = self._client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        text = result["content"][0]["text"]
        return CompletionResponse(text=text, raw=result)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        content = _build_content_block(text=prompt)
        body = self._build_body(content)

        response = self._client.invoke_model_with_response_stream(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        def _gen() -> Generator[CompletionResponse, None, None]:
            accumulated = ""
            for event in response["body"]:
                chunk = json.loads(event["chunk"]["bytes"])
                delta = chunk.get("delta", {})
                if delta.get("type") == "text_delta":
                    token = delta.get("text", "")
                    accumulated += token
                    yield CompletionResponse(text=accumulated, delta=token, raw=chunk)

        return _gen()

    # ── Chat interface ────────────────────────────────────────────────────────

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        anthropic_messages = _llama_messages_to_anthropic(messages)
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "messages": anthropic_messages,
        }
        # Extract system message if present
        system_msgs = [m for m in messages if m.role == MessageRole.SYSTEM]
        if system_msgs:
            body["system"] = system_msgs[0].content

        response = self._client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        text = result["content"][0]["text"]
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=text),
            raw=result,
        )

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        anthropic_messages = _llama_messages_to_anthropic(messages)
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": anthropic_messages,
        }

        response = self._client.invoke_model_with_response_stream(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        def _gen() -> Generator[ChatResponse, None, None]:
            accumulated = ""
            for event in response["body"]:
                chunk = json.loads(event["chunk"]["bytes"])
                delta = chunk.get("delta", {})
                if delta.get("type") == "text_delta":
                    token = delta.get("text", "")
                    accumulated += token
                    yield ChatResponse(
                        message=ChatMessage(role=MessageRole.ASSISTANT, content=accumulated),
                        delta=token,
                        raw=chunk,
                    )

        return _gen()

    # ── Multimodal helper ─────────────────────────────────────────────────────

    def complete_with_images(
        self,
        text: str,
        images: list[Image.Image],
        **kwargs: Any,
    ) -> str:
        """Convenience wrapper for vision + text completion."""
        resp = self.complete(text, images=images, **kwargs)
        return resp.text

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_body(self, content: list[dict]) -> dict:
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "messages": [{"role": "user", "content": content}],
        }


# ── Utilities ─────────────────────────────────────────────────────────────────

def _llama_messages_to_anthropic(messages: Sequence[ChatMessage]) -> list[dict]:
    """Convert LlamaIndex ChatMessage list to Anthropic Messages API format."""
    result = []
    for msg in messages:
        if msg.role == MessageRole.SYSTEM:
            continue  # handled separately
        role = "user" if msg.role == MessageRole.USER else "assistant"
        result.append({"role": role, "content": [{"type": "text", "text": msg.content}]})
    return result
