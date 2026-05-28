"""POST /entities — extract financial named entities using the LoRA NER model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.schemas import EntityRequest, EntityResponse, EntityResult
from src.config import Settings, get_settings

if TYPE_CHECKING:
    from src.finetune.inference import NERInferenceEngine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/entities", tags=["ner"])


def _get_ner_engine(request: Request) -> "NERInferenceEngine":
    engine = getattr(request.app.state, "ner_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NER engine is not initialised; check LORA_MODEL_PATH configuration",
        )
    return engine


@router.post(
    "",
    response_model=EntityResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract financial named entities",
    description=(
        "Run the fine-tuned LoRA BERT NER model over input text and return all "
        "detected financial entities: ORG (organisations), MONEY (monetary values), "
        "DATE (dates / fiscal periods), PERCENT (percentage figures)."
    ),
)
async def extract_entities(
    body: EntityRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> EntityResponse:
    engine = _get_ner_engine(request)

    logger.info("NER extraction on %d chars", len(body.text))
    try:
        entities = engine.extract_entities(body.text)
    except Exception as exc:
        logger.exception("NER extraction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entity extraction failed: {exc}",
        )

    entity_results = [
        EntityResult(
            text=e.text,
            label=e.label,
            start=e.start,
            end=e.end,
            confidence=round(e.confidence, 4),
        )
        for e in entities
    ]

    return EntityResponse(
        entities=entity_results,
        entity_count=len(entity_results),
        text_length=len(body.text),
        model_path=str(settings.lora_model_path),
    )
