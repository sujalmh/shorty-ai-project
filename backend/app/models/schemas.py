from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class CellAttributes(BaseModel):
    type: Optional[str] = None  # numeric, string, date, formula
    tags: Optional[List[str]] = None
    source: Optional[str] = None  # manual or ai
    formula: Optional[str] = None  # expression if computed


class ImportDataRequest(BaseModel):
    rows: List[Dict[str, Any]] = []


class DataResponse(BaseModel):
    columns: List[Dict[str, Any]]
    rows: List[Dict[str, Any]]
    metadata: Dict[str, Any] = {}


class CellUpdateRequest(BaseModel):
    rowIndex: int
    field: str
    value: Any
    attributes: Optional[CellAttributes] = None


class AddColumnRequest(BaseModel):
    name: str
    expression: str


class HighlightRequest(BaseModel):
    condition: str  # e.g., "profit < 0"


class PlotRequest(BaseModel):
    x: str
    y: Optional[str] = None
    kind: Optional[str] = "bar"  # bar, line, pie


class ChatRequest(BaseModel):
    message: str
    # Chat mode: "auto" uses LLM if available else rules; "llm" forces LLM; "rules" forces fallback.
    mode: Optional[str] = "auto"


class ChatResponse(BaseModel):
    content: str  # assistant reply
    actions: List[Dict[str, Any]] = []  # structured steps for UI (no hidden chain-of-thought)
    suggestions: List[str] = []  # up to 2 LLM-generated query suggestions
