"""LlamaIndex CustomLLM wrapper for AWS Bedrock (Amazon Nova Lite / Claude).

HOW IT WORKS (simple analogy):
  This class is the "brain" of the RAG system — it wraps the AWS Bedrock API
  so that LlamaIndex can call it like any other language model.

  Think of it like a UiPath HTTP Request activity configured to call AWS Bedrock.
  The class handles:
  - Building the correct JSON payload format (Nova vs Claude have different formats)
  - Sending the request to AWS Bedrock via boto3
  - Parsing the response back into text

  Two API formats are supported:
  - Amazon Nova Lite: {"text": "..."} for text, {"image": {...}} for images
  - Anthropic Claude 3: {"type": "text", "text": "..."} for text

  IMPORTANT: is_chat_model = False is intentional. If set to True, LlamaIndex
  routes through its chat() method using Claude-format messages, which Nova
  rejects. Setting False forces LlamaIndex to use complete() instead.

Supports:
- Standard text completion (complete)
- Streaming responses (stream_complete, stream_chat)
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


def _extract_text(result: dict) -> str:
    """Parse the text response from either Claude or Nova response format.

    Claude format response:
      {"content": [{"text": "The answer is..."}]}

    Nova format response:
      {"output": {"message": {"content": [{"text": "The answer is..."}]}}}

    Returns empty string if neither format matches.
    """
    if "content" in result:      # Anthropic Claude format
        return result["content"][0]["text"]
    if "output" in result:       # Amazon Nova format
        return result["output"]["message"]["content"][0]["text"]
    return ""

# Set of Claude 3 model IDs — used to choose the correct request format
_CLAUDE3_MODELS = {
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
}


def _pil_to_b64(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL Image to a base64-encoded string.

    Base64 encodes binary image bytes as ASCII text, which can be safely
    embedded inside JSON payloads sent to the Bedrock API.
    """
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _build_content_block(
    text: Optional[str] = None,
    image: Optional[Image.Image] = None,
    image_bytes: Optional[bytes] = None,
    media_type: str = "image/png",
) -> list[dict]:
    """Build Bedrock Anthropic Messages API content blocks.

    Content blocks are the pieces of a message — each block is either:
    - A text block: {"type": "text", "text": "..."}
    - An image block: {"type": "image", "source": {"type": "base64", ...}}

    NOTE: This builds Claude-format blocks. _nova_content() converts them
    to Nova format if the model is Nova.

    Can accept either a PIL Image object or raw bytes — useful for chart images
    stored as bytes in ChartNode.
    """
    blocks: list[dict] = []

    # If a PIL Image was provided, encode it to base64 and add an image block
    if image is not None:
        b64 = _pil_to_b64(image)
        blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            }
        )
    # If raw bytes were provided instead of a PIL Image
    elif image_bytes is not None:
        b64 = base64.b64encode(image_bytes).decode()
        blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64},
            }
        )

    # If text was provided, add a text block
    if text:
        blocks.append({"type": "text", "text": text})

    return blocks


class BedrockLLM(CustomLLM):
    """LlamaIndex LLM backed by AWS Bedrock (Nova Lite / Claude 3).

    Extends LlamaIndex's CustomLLM — implements the required interface methods
    so it can be used as a drop-in LLM in any LlamaIndex pipeline.

    Parameters
    ----------
    model_id:
        Bedrock model identifier (e.g. "amazon.nova-lite-v1:0").
    aws_region:
        AWS region where Bedrock is enabled (must be us-east-1 for Nova).
    max_tokens:
        Maximum tokens to generate in the response.
    temperature:
        Sampling temperature. 0.0 = deterministic, 1.0 = very creative.
        0.1 is used here — mostly deterministic for factual financial answers.
    top_p:
        Nucleus sampling — only consider tokens making up the top P% of probability.
    session_kwargs:
        Extra kwargs forwarded to boto3.Session (credentials, region, etc.).
    """

    # Pydantic field declarations (required by LlamaIndex's CustomLLM base)
    model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    aws_region: str = "us-east-1"
    max_tokens: int = 4096
    temperature: float = 0.1   # low temperature = factual, consistent answers
    top_p: float = 0.9

    # Private attribute: the boto3 Bedrock client (set in __init__)
    _client: Any = None

    class Config:
        arbitrary_types_allowed = True  # allow storing boto3 client (non-Pydantic type)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Create the boto3 Bedrock client — this handles authentication and HTTP
        session_kwargs = kwargs.get("session_kwargs", {})
        session = boto3.Session(**session_kwargs)
        # Store the client in the private attribute (bypass Pydantic immutability)
        object.__setattr__(
            self,
            "_client",
            session.client("bedrock-runtime", region_name=self.aws_region),
        )

    @property
    def metadata(self) -> LLMMetadata:
        """Return LLM metadata for LlamaIndex's internal routing logic.

        IMPORTANT: is_chat_model=False is intentional.
        If True, LlamaIndex routes through chat() using Claude message format,
        which Amazon Nova rejects with a 400 error.
        Setting False forces LlamaIndex to always use complete() instead.
        """
        return LLMMetadata(
            context_window=200_000,          # Nova supports up to 200K token context
            num_output=self.max_tokens,
            is_chat_model=False,             # ← IMPORTANT: do NOT change to True
            is_function_calling_model=False,
            model_name=self.model_id,
        )

    # ── Completion Interface ──────────────────────────────────────────────────

    @llm_completion_callback()  # LlamaIndex decorator for logging/tracing
    def complete(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate a completion for a text prompt, optionally with images.

        This is the main method called by LlamaIndex during RAG query generation.

        If images are provided (chart images from the PDF), they are included
        in the Bedrock request as base64-encoded content blocks.
        The model can then "see" the charts while answering the question.
        """
        # Build the content block list: images first (if any), then the text prompt
        content = []
        if images:
            for img in images:
                content.extend(_build_content_block(image=img))
        content.extend(_build_content_block(text=prompt))

        # Build the full request body (format depends on model type: Nova vs Claude)
        body = self._build_body(content)
        response = self._client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())  # parse the JSON response
        text = _extract_text(result)                  # extract the generated text
        return CompletionResponse(text=text, raw=result)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Stream a completion response token by token.

        Uses Bedrock's streaming API — the response arrives incrementally,
        so the user sees text appearing progressively (like ChatGPT streaming).

        Yields CompletionResponse objects with accumulated text + the new delta
        (the new token added since the last yield).
        """
        content = _build_content_block(text=prompt)
        body = self._build_body(content)

        # invoke_model_with_response_stream returns an event stream
        response = self._client.invoke_model_with_response_stream(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        def _gen() -> Generator[CompletionResponse, None, None]:
            """Generator that yields one token at a time from the stream."""
            accumulated = ""  # build up the full response text incrementally
            for event in response["body"]:
                chunk = json.loads(event["chunk"]["bytes"])
                delta = chunk.get("delta", {})
                # "text_delta" events contain new tokens
                if delta.get("type") == "text_delta":
                    token = delta.get("text", "")
                    accumulated += token
                    # Yield both the full text so far and the new delta token
                    yield CompletionResponse(text=accumulated, delta=token, raw=chunk)

        return _gen()

    # ── Chat Interface ────────────────────────────────────────────────────────

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Handle a chat-style conversation (multi-turn messages).

        Converts LlamaIndex ChatMessage objects to the format expected by
        either Nova or Claude. System messages are handled separately in
        both formats (they're not part of the messages array).
        """
        # Convert LlamaIndex messages to Bedrock format
        anthropic_messages = _llama_messages_to_anthropic(messages, nova=self._is_nova())
        system_msgs = [m for m in messages if m.role == MessageRole.SYSTEM]

        if self._is_nova():
            # Nova format: system message goes in a "system" array with {"text": ...}
            body: dict = {
                "messages": anthropic_messages,
                "inferenceConfig": {
                    "max_new_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                },
            }
            if system_msgs:
                body["system"] = [{"text": system_msgs[0].content}]
        else:
            # Claude format: system message is a top-level string field
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "messages": anthropic_messages,
            }
            if system_msgs:
                body["system"] = system_msgs[0].content

        response = self._client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        text = _extract_text(result)
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=text),
            raw=result,
        )

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        """Stream a chat response token by token.

        Same as stream_complete but for multi-turn chat format.
        Yields ChatResponse objects as tokens arrive.
        """
        anthropic_messages = _llama_messages_to_anthropic(messages, nova=self._is_nova())
        if self._is_nova():
            body = {
                "messages": anthropic_messages,
                "inferenceConfig": {
                    "max_new_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            }
        else:
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

    # ── Multimodal Convenience Helper ─────────────────────────────────────────

    def complete_with_images(
        self,
        text: str,
        images: list[Image.Image],
        **kwargs: Any,
    ) -> str:
        """Convenience wrapper: send text + images and return just the text response.

        Used by the pipeline when relevant charts are found for a query.
        The model can "see" the chart images while generating the answer.
        Returns the answer string directly (not a CompletionResponse object).
        """
        resp = self.complete(text, images=images, **kwargs)
        return resp.text

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _is_nova(self) -> bool:
        """Check if this instance is configured to use an Amazon Nova model.

        Nova and Claude have different JSON request formats — this flag
        is used throughout to choose the correct format.
        """
        return "nova" in self.model_id.lower()

    def _nova_content(self, claude_content: list[dict]) -> list[dict]:
        """Convert Claude-format content blocks to Nova format.

        Claude format:  {"type": "text", "text": "..."}
        Nova format:    {"text": "..."}

        Claude image:   {"type": "image", "source": {"type": "base64", "data": "..."}}
        Nova image:     {"image": {"format": "png", "source": {"bytes": "..."}}}
        """
        nova_blocks = []
        for block in claude_content:
            if block.get("type") == "text":
                # Claude text block → Nova text block (simpler format)
                nova_blocks.append({"text": block["text"]})
            elif block.get("type") == "image":
                # Claude image block → Nova image block (different structure)
                src = block.get("source", {})
                nova_blocks.append({
                    "image": {
                        "format": src.get("media_type", "image/png").split("/")[-1],  # "image/png" → "png"
                        "source": {"bytes": src.get("data", "")},
                    }
                })
            # Skip any unknown block types silently
        return nova_blocks

    def _build_body(self, content: list[dict]) -> dict:
        """Build the full Bedrock request body from a list of content blocks.

        Chooses between Nova and Claude format based on the model ID.
        Nova uses "inferenceConfig" with "max_new_tokens".
        Claude uses top-level "max_tokens" with "anthropic_version".
        """
        if self._is_nova():
            return {
                "messages": [{"role": "user", "content": self._nova_content(content)}],
                "inferenceConfig": {
                    "max_new_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                },
            }
        # Claude format
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "messages": [{"role": "user", "content": content}],
        }


# ── Utility Functions ─────────────────────────────────────────────────────────

def _llama_messages_to_anthropic(messages: Sequence[ChatMessage], nova: bool = False) -> list[dict]:
    """Convert LlamaIndex ChatMessage list to Anthropic/Nova Messages API format.

    LlamaIndex uses its own ChatMessage type (role + content).
    Bedrock expects a specific JSON structure with role and content blocks.

    System messages are excluded here — they are handled separately in
    the chat() method (either as a "system" string in Claude or a
    "system" array in Nova).

    Nova requires strict user/assistant alternation — consecutive messages
    from the same role are merged into one to satisfy this constraint.

    Example input (LlamaIndex format):
      [ChatMessage(role=USER, content="What is revenue?")]

    Example output (Nova format):
      [{"role": "user", "content": [{"text": "What is revenue?"}]}]

    Example output (Claude format):
      [{"role": "user", "content": [{"type": "text", "text": "What is revenue?"}]}]
    """
    result = []
    for msg in messages:
        if msg.role == MessageRole.SYSTEM:
            continue  # system messages are handled separately in the body builder

        # Map LlamaIndex roles to Bedrock roles (only "user" and "assistant" allowed)
        role = "user" if msg.role == MessageRole.USER else "assistant"
        content = msg.content or ""

        # Build the content block in the appropriate format
        if nova:
            block = {"text": content}                          # Nova: simple text
        else:
            block = {"type": "text", "text": content}         # Claude: typed text

        # Merge consecutive messages from the same role
        # (Nova requires strict alternation: user, assistant, user, assistant...)
        if result and result[-1]["role"] == role:
            result[-1]["content"].append(block)   # append to existing message
        else:
            result.append({"role": role, "content": [block]})  # new message

    return result
