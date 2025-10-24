from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Response
import logging
import json

logger = logging.getLogger("ai.api")
from ..models.schemas import (
    ImportDataRequest,
    DataResponse,
    CellUpdateRequest,
    AddColumnRequest,
    HighlightRequest,
    PlotRequest,
    ChatRequest,
    ChatResponse,
)
from ..services.data_service import data_service
from ..services.ai_service import ai_service

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/data", response_model=DataResponse)
def get_data() -> DataResponse:
    return DataResponse(
        columns=data_service.get_columns(),
        rows=data_service.get_rows(),
        metadata=data_service.get_metadata_summary(),
    )


@router.post("/import", response_model=DataResponse)
def import_data(payload: ImportDataRequest) -> DataResponse:
    try:
        data_service.import_rows(payload.rows)
        return DataResponse(
            columns=data_service.get_columns(),
            rows=data_service.get_rows(),
            metadata=data_service.get_metadata_summary(),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/import-csv", response_model=DataResponse)
async def import_csv(file: UploadFile = File(...), encoding: str | None = Form(None)) -> DataResponse:
    try:
        logger.info("POST /import-csv filename=%s encoding=%s", getattr(file, "filename", None), encoding)
        content = await file.read()
        data_service.import_csv_bytes(content, encoding=encoding)
        return DataResponse(
            columns=data_service.get_columns(),
            rows=data_service.get_rows(),
            metadata=data_service.get_metadata_summary(),
        )
    except Exception as e:
        logger.exception("POST /import-csv failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/import-excel", response_model=DataResponse)
async def import_excel(file: UploadFile = File(...), sheet_name: str | None = Form(None)) -> DataResponse:
    try:
        logger.info("POST /import-excel filename=%s sheet_name=%s", getattr(file, "filename", None), sheet_name)
        content = await file.read()
        data_service.import_excel_bytes(content, sheet_name=sheet_name)
        return DataResponse(
            columns=data_service.get_columns(),
            rows=data_service.get_rows(),
            metadata=data_service.get_metadata_summary(),
        )
    except Exception as e:
        logger.exception("POST /import-excel failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/cell", response_model=DataResponse)
def update_cell(payload: CellUpdateRequest) -> DataResponse:
    try:
        attrs = payload.attributes.dict() if payload.attributes else None
        data_service.update_cell(payload.rowIndex, payload.field, payload.value, attrs)
        return DataResponse(
            columns=data_service.get_columns(),
            rows=data_service.get_rows(),
            metadata=data_service.get_metadata_summary(),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/rows/add", response_model=DataResponse)
def add_row() -> DataResponse:
    try:
        data_service.add_row()
        return DataResponse(
            columns=data_service.get_columns(),
            rows=data_service.get_rows(),
            metadata=data_service.get_metadata_summary(),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/columns/add-empty", response_model=DataResponse)
def add_empty_column(name: str = Form(...), fill: str | None = Form(None)) -> DataResponse:
    try:
        data_service.add_empty_column(name, fill)
        return DataResponse(
            columns=data_service.get_columns(),
            rows=data_service.get_rows(),
            metadata=data_service.get_metadata_summary(),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/add-column", response_model=DataResponse)
def add_column(payload: AddColumnRequest) -> DataResponse:
    try:
        data_service.add_column_expression(payload.name, payload.expression)
        return DataResponse(
            columns=data_service.get_columns(),
            rows=data_service.get_rows(),
            metadata=data_service.get_metadata_summary(),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/highlight")
def highlight(payload: HighlightRequest) -> dict:
    try:
        rows = data_service.highlight_rows(payload.condition)
        return {"rows": rows}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/plot")
def plot(payload: PlotRequest) -> Response:
    logger.info("POST /plot x=%s y=%s kind=%s", payload.x, payload.y, payload.kind)
    try:
        fig = data_service.plot(payload.x, payload.y, payload.kind or "bar")
        try:
            logger.info("POST /plot ok traces=%s", len((fig or {}).get("data", [])))
        except Exception:
            pass
        return Response(content=json.dumps(fig, default=str), media_type="application/json")
    except Exception as e:
        logger.exception("POST /plot failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    logger.info("POST /chat mode=%s message=%s", payload.mode, payload.message)
    try:
        resp = ai_service.chat(payload.message, payload.mode)
        try:
            logger.info("POST /chat resp actions=%s", [a.get("type") for a in resp.actions])
        except Exception:
            pass
        return resp
    except Exception as e:
        logger.exception("POST /chat failed: %s", e)
        # Fallback to error message while keeping schema
        return ChatResponse(content=f"Error: {str(e)}", actions=[{"type": "error"}])
