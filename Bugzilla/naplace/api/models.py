from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class TextItem(BaseModel):
    """Single input text unit for prediction."""

    text: str = Field(..., description="Bug report summary or description")


class PredictionRequest(BaseModel):
    """Request body for prediction endpoints."""

    texts: List[TextItem] = Field(
        ...,
        description="List of texts to classify",
        # min_length=1,  # ❌ da togliere, rompe Pydantic v1
    )


class PredictionItem(BaseModel):
    """Single prediction result."""

    input_text: str = Field(..., description="Original input text")
    predicted_label: str = Field(..., description="Predicted component / class")
    probability: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score for the predicted label, if available",
    )


class PredictionResponse(BaseModel):
    """Response body from prediction endpoints."""

    model_name: str = Field(..., description="Name of the model used for prediction")
    predictions: List[PredictionItem] = Field(
        ..., description="List of predictions for each input text"
    )
