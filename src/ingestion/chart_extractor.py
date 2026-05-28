"""Chart and figure detection with CLIP + Bedrock Claude vision captioning.

Pipeline:
1. CLIP zero-shot classification to score whether an image is a chart/graph.
2. If score exceeds the threshold, send the image to Bedrock Claude 3 for a
   detailed financial-chart caption.
3. Return a ChartNode with image bytes and the generated caption.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass
from typing import Optional

import boto3
import open_clip
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Candidate labels for CLIP zero-shot classification
_CHART_LABELS = [
    "a financial chart or graph",
    "a table of numbers or data",
    "a bar chart",
    "a line graph",
    "a pie chart",
    "a scatter plot",
    "a candlestick chart",
    "a heat map",
    "an infographic",
    "a photograph",
    "a logo or icon",
    "decorative artwork",
    "a text paragraph",
]

_CHART_POSITIVE_LABELS = {
    "a financial chart or graph",
    "a table of numbers or data",
    "a bar chart",
    "a line graph",
    "a pie chart",
    "a scatter plot",
    "a candlestick chart",
    "a heat map",
    "an infographic",
}


@dataclass
class ChartNode:
    """Represents a detected chart/figure with its auto-generated caption."""

    image_bytes: bytes          # PNG bytes
    caption: str
    chart_type: str             # top CLIP label
    clip_score: float           # CLIP confidence for chart class
    page_number: int
    image_index: int
    source: str
    width: int
    height: int

    def to_dict(self) -> dict:
        return {
            "caption": self.caption,
            "chart_type": self.chart_type,
            "clip_score": self.clip_score,
            "page_number": self.page_number,
            "image_index": self.image_index,
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "image_b64": base64.b64encode(self.image_bytes).decode(),
        }


class ChartExtractor:
    """Detect charts and generate captions using CLIP + Bedrock Claude vision.

    Parameters
    ----------
    clip_model_name:
        Open CLIP model name (e.g. ``"ViT-B-32"``).
    clip_pretrained:
        Open CLIP pretrained weights tag (e.g. ``"openai"``).
    chart_threshold:
        Minimum combined CLIP probability for positive chart labels to classify
        an image as a chart.
    bedrock_model_id:
        Bedrock model to use for vision captioning.
    aws_region:
        AWS region for the Bedrock client.
    device:
        PyTorch device string; defaults to CUDA if available.
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        chart_threshold: float = 0.45,
        bedrock_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        aws_region: str = "us-east-1",
        device: Optional[str] = None,
    ) -> None:
        self.chart_threshold = chart_threshold
        self.bedrock_model_id = bedrock_model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading CLIP model %s/%s on %s", clip_model_name, clip_pretrained, self.device)
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained
        )
        self._model = self._model.to(self.device).eval()
        self._tokenizer = open_clip.get_tokenizer(clip_model_name)
        self._text_features = self._encode_labels()

        self._bedrock = boto3.client("bedrock-runtime", region_name=aws_region)

    # ── Public API ────────────────────────────────────────────────────────────

    def is_chart(self, image: Image.Image) -> tuple[bool, float, str]:
        """Return (is_chart, chart_probability, top_label) for an image."""
        probs, top_label = self._clip_classify(image)
        chart_prob = sum(
            p for label, p in zip(_CHART_LABELS, probs.tolist())
            if label in _CHART_POSITIVE_LABELS
        )
        return chart_prob >= self.chart_threshold, chart_prob, top_label

    def caption_chart(self, image: Image.Image, source_context: str = "") -> str:
        """Generate a detailed financial caption for a chart image via Bedrock Claude."""
        img_b64 = _image_to_base64(image)
        prompt_text = (
            "You are a financial analyst examining a chart or figure from a financial document. "
            "Provide a detailed, precise description of this chart including: "
            "(1) the chart type, (2) the title or subject matter if visible, "
            "(3) the axes labels and units, (4) key data trends or insights, "
            "(5) any notable data points, peaks, troughs, or anomalies. "
            "Be specific about numerical values where legible. "
        )
        if source_context:
            prompt_text += f"Context: this image comes from '{source_context}'. "
        prompt_text += "Description:"

        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 512,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": img_b64,
                                },
                            },
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ],
            }
        )

        try:
            response = self._bedrock.invoke_model(
                modelId=self.bedrock_model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            return result["content"][0]["text"].strip()
        except Exception as exc:
            logger.warning("Bedrock captioning failed: %s", exc)
            return "Chart/figure detected (captioning unavailable)."

    def extract_charts(
        self,
        images: list,  # list of EmbeddedImage from pdf_parser
        generate_captions: bool = True,
    ) -> list[ChartNode]:
        """Filter chart images and optionally generate captions for each."""
        chart_nodes: list[ChartNode] = []

        for emb_img in images:
            pil = emb_img.image
            detected, score, top_label = self.is_chart(pil)
            if not detected:
                logger.debug(
                    "Image p%d/i%d classified as non-chart (score=%.3f, top=%s)",
                    emb_img.page_number,
                    emb_img.image_index,
                    score,
                    top_label,
                )
                continue

            logger.info(
                "Chart detected p%d/i%d score=%.3f type=%s",
                emb_img.page_number,
                emb_img.image_index,
                score,
                top_label,
            )

            img_bytes = emb_img.to_bytes("PNG")
            caption = ""
            if generate_captions:
                caption = self.caption_chart(pil, source_context=emb_img.source)

            chart_nodes.append(
                ChartNode(
                    image_bytes=img_bytes,
                    caption=caption,
                    chart_type=top_label,
                    clip_score=score,
                    page_number=emb_img.page_number,
                    image_index=emb_img.image_index,
                    source=emb_img.source,
                    width=pil.width,
                    height=pil.height,
                )
            )

        return chart_nodes

    # ── Private helpers ───────────────────────────────────────────────────────

    def _encode_labels(self) -> torch.Tensor:
        """Pre-compute normalised CLIP text features for all candidate labels."""
        tokens = self._tokenizer(_CHART_LABELS).to(self.device)
        with torch.no_grad():
            text_feats = self._model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        return text_feats

    def _clip_classify(self, image: Image.Image) -> tuple[torch.Tensor, str]:
        """Return softmax probabilities and the top label name."""
        img_tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_feats = self._model.encode_image(img_tensor)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            logits = (100.0 * img_feats @ self._text_features.T).softmax(dim=-1)
        probs = logits.squeeze(0).cpu()
        top_idx = probs.argmax().item()
        return probs, _CHART_LABELS[top_idx]


# ── Utilities ─────────────────────────────────────────────────────────────────

def _image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
