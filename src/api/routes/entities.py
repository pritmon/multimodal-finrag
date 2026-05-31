"""POST /entities — extract financial named entities using the LoRA NER model.

HOW IT WORKS (simple analogy):
  NER (Named Entity Recognition) is like a smart "Find and Highlight" feature
  that automatically finds and labels important terms in financial text.

  Given: "Goldman Sachs reported $2.1 billion in revenue for Q3 2023"
  Returns:
    - "Goldman Sachs" → ORG (organisation)
    - "$2.1 billion"  → MONEY (monetary value)
    - "Q3 2023"       → DATE (time period)

  Entity types recognised by this model:
    ORG     → company names (Goldman Sachs, Apple Inc, JPMorgan Chase)
    MONEY   → monetary values ($2.1 billion, 500 million, $4.2 trillion)
    DATE    → dates and periods (Q3 2023, FY2024, January 2024)
    PERCENT → percentages (15%, 2.5 percent, 150 basis points)

  This uses a BERT model fine-tuned with LoRA adapters on synthetic financial
  training data. LoRA (Low-Rank Adaptation) fine-tunes only a small percentage
  of the model's parameters — much more efficient than full fine-tuning.

  The NER engine is optional — if the LoRA model weights aren't present,
  this endpoint returns HTTP 503 (Service Unavailable).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.schemas import EntityRequest, EntityResponse, EntityResult
from src.config import Settings, get_settings

# TYPE_CHECKING: only imported for type hints (avoids circular import at runtime)
if TYPE_CHECKING:
    from src.finetune.inference import NERInferenceEngine

logger = logging.getLogger(__name__)

# All endpoints in this router live under /entities with tag "ner"
router = APIRouter(prefix="/entities", tags=["ner"])


def _get_ner_engine(request: Request) -> "NERInferenceEngine":
    """Dependency function: get the NER engine from app.state.

    Returns HTTP 503 if the engine isn't available (model file missing,
    or startup failed). This way the rest of the API still works even
    if NER isn't configured.
    """
    engine = getattr(request.app.state, "ner_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NER engine is not initialised; check LORA_MODEL_PATH configuration",
        )
    return engine


@router.post(
    "",                                     # path: POST /entities
    response_model=EntityResponse,          # validates + documents response shape
    status_code=status.HTTP_200_OK,
    summary="Extract financial named entities",
    description=(
        "Run the fine-tuned LoRA BERT NER model over input text and return all "
        "detected financial entities: ORG (organisations), MONEY (monetary values), "
        "DATE (dates / fiscal periods), PERCENT (percentage figures)."
    ),
)
async def extract_entities(
    body: EntityRequest,           # validated request body (text to process)
    request: Request,              # needed to access app.state.ner_engine
    settings: Settings = Depends(get_settings),
) -> EntityResponse:
    """Extract named financial entities from the provided text.

    The text is passed through the fine-tuned BERT NER model.
    Returns a list of detected entities with:
      - text: the exact string matched in the input
      - label: entity type (ORG, MONEY, DATE, PERCENT)
      - start/end: character offsets in the input string
      - confidence: model confidence score (0.0 to 1.0)

    Overlapping entities are deduplicated — the higher-confidence one wins.
    """
    # Get the NER engine from app.state (raises 503 if not available)
    engine = _get_ner_engine(request)

    logger.info("NER extraction on %d chars", len(body.text))
    try:
        # Run the LoRA NER model on the input text
        # Returns a list of Entity objects (text, label, start, end, confidence)
        entities = engine.extract_entities(body.text)
    except Exception as exc:
        logger.exception("NER extraction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entity extraction failed: {exc}",
        )

    # Convert Entity objects to API-friendly EntityResult schemas
    # Round confidence to 4 decimal places for clean JSON output
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
        entity_count=len(entity_results),  # total number of entities found
        text_length=len(body.text),        # length of the input text
        model_path=str(settings.lora_model_path),  # which model was used
    )
