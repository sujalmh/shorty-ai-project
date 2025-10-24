from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Callable, cast

from pydantic import BaseModel, Field

from ..models.schemas import ChatResponse
from .data_service import data_service
from dotenv import load_dotenv
load_dotenv()

import logging
logger = logging.getLogger("ai.ai_service")

# Optional LangChain + OpenAI tooling
# Define a flexible name "lc_tool" usable whether LangChain is installed or not.
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
    from langchain_core.tools import tool as _lc_tool
    lc_tool = _lc_tool  # type: ignore[assignment]
except Exception:  # pragma: no cover - LangChain is optional
    ChatOpenAI = None  # type: ignore[assignment]
    SystemMessage = HumanMessage = AIMessage = ToolMessage = object  # type: ignore[assignment]
    def lc_tool(name_or_callable: Any = None, *args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        def _decorator(f: Any) -> Any:
            return f
        if callable(name_or_callable):
            return name_or_callable
        return _decorator


# ----------------------------- LangChain Tools -----------------------------
class AddColumnArgs(BaseModel):
    """Create a new computed column using a Pandas expression, e.g. revenue - cost"""
    name: str = Field(..., description="Name of the new column to create, e.g. profit")
    expression: str = Field(..., description="Pandas df.eval() compatible expression, e.g. revenue - cost")


@lc_tool("add_column", args_schema=AddColumnArgs)
def add_column_tool(name: str, expression: str) -> str:
    data_service.add_column_expression(name, expression)
    return json.dumps({"status": "ok", "action": "add_column", "name": name, "expression": expression})


class HighlightArgs(BaseModel):
    """Return row indices where the boolean condition is true, e.g. profit < 0"""
    condition: str = Field(..., description="Pandas boolean expression, e.g. profit < 0")


@lc_tool("highlight_rows", args_schema=HighlightArgs)
def highlight_rows_tool(condition: str) -> str:
    rows = data_service.highlight_rows(condition)
    return json.dumps({"status": "ok", "action": "highlight_rows", "condition": condition, "rows": rows})


class SummarizeArgs(BaseModel):
    """Summarize a numeric column by computing its total sum"""
    column: Optional[str] = Field(None, description="Column to sum, e.g. revenue. If omitted, choose a reasonable numeric column.")


@lc_tool("summarize_total", args_schema=SummarizeArgs)
def summarize_total_tool(column: Optional[str] = None) -> str:
    col = column
    if not col:
        # pick a reasonable numeric column
        for c in data_service.df.columns:
            sc = str(c).lower()
            if sc in {"revenue", "sales", "amount", "cost", "profit", "q1", "q2"}:
                col = str(c)
                break
        if not col:
            return json.dumps({"status": "error", "message": "No numeric column inferred"})
    total_val = float(data_service.df[col].fillna(0).sum())
    return json.dumps({"status": "ok", "action": "summary", "column": col, "total": total_val})


class UpdateCellArgs(BaseModel):
    """Update a single cell at (rowIndex, field) with a new value"""
    rowIndex: int = Field(..., description="Zero-based row index")
    field: str = Field(..., description="Column/field name")
    value: Any = Field(..., description="New cell value")


@lc_tool("update_cell", args_schema=UpdateCellArgs)
def update_cell_tool(rowIndex: int, field: str, value: Any) -> str:
    data_service.update_cell(rowIndex, field, value, {"source": "ai"})
    return json.dumps({"status": "ok", "action": "update_cell", "rowIndex": rowIndex, "field": field, "value": value})


class AddRowArgs(BaseModel):
    """Append a row to the dataframe with optional values"""
    values: Optional[Dict[str, Any]] = Field(None, description="Optional dict of field->value for the new row")


@lc_tool("add_row", args_schema=AddRowArgs)
def add_row_tool(values: Optional[Dict[str, Any]] = None) -> str:
    idx = data_service.add_row(values)
    return json.dumps({"status": "ok", "action": "add_row", "rowIndex": idx, "values": values or {}})


class AddEmptyColumnArgs(BaseModel):
    """Create a new empty column with optional default fill"""
    name: str = Field(..., description="Name of the new empty column")
    fill: Optional[Any] = Field(None, description="Default fill value for existing rows")


@lc_tool("add_empty_column", args_schema=AddEmptyColumnArgs)
def add_empty_column_tool(name: str, fill: Optional[Any] = None) -> str:
    data_service.add_empty_column(name, fill)
    return json.dumps({"status": "ok", "action": "add_empty_column", "name": name, "fill": fill})


class DeleteRowsArgs(BaseModel):
    """Delete one or more rows by zero-based indices"""
    rows: List[int] = Field(..., description="Zero-based row indices to delete")


@lc_tool("delete_rows", args_schema=DeleteRowsArgs)
def delete_rows_tool(rows: List[int]) -> str:
    deleted = data_service.delete_rows(rows or [])
    return json.dumps({"status": "ok", "action": "delete_rows", "rows": deleted})


class DeleteColumnArgs(BaseModel):
    """Delete a column by name"""
    name: str = Field(..., description="Exact column/field name to delete")


@lc_tool("delete_column", args_schema=DeleteColumnArgs)
def delete_column_tool(name: str) -> str:
    ok = data_service.delete_column(name)
    return json.dumps({"status": "ok" if ok else "noop", "action": "delete_column", "name": name, "deleted": ok})


class PlotArgs(BaseModel):
    """Generate a plot (bar, line, or pie) and return a Plotly figure spec"""
    x: str = Field(..., description="Column to use for x axis or 'names' for pie")
    y: Optional[str] = Field(None, description="Column to use for y values (required for bar/line/pie values)")
    kind: Optional[str] = Field("bar", description="One of bar, line, pie")


@lc_tool("plot", args_schema=PlotArgs)
def plot_tool(x: str, y: Optional[str] = None, kind: Optional[str] = "bar") -> str:
    fig = data_service.plot(x=x, y=y, kind=kind or "bar")
    return json.dumps({"status": "ok", "action": "plot", "kind": kind or "bar", "x": x, "y": y, "figure": fig})


@lc_tool("get_columns")
def get_columns_tool() -> str:
    """Return the current dataframe column schema with types"""
    return json.dumps({"columns": data_service.get_columns()})


@lc_tool("get_preview")
def get_preview_tool() -> str:
    """Return a short preview of the current data (first 5 rows)"""
    return json.dumps({"rows": data_service.get_rows()[:5]})


LANGCHAIN_TOOLS = [
    get_columns_tool,
    get_preview_tool,
    add_column_tool,
    update_cell_tool,
    add_row_tool,
    add_empty_column_tool,
    delete_rows_tool,
    delete_column_tool,
    highlight_rows_tool,
    summarize_total_tool,
    plot_tool,
]


SYSTEM_PROMPT = (
    "You are an AI assistant for an interactive spreadsheet.\n"
    "- You MUST decide and call one or more tools to fulfill EVERY user request. Do not answer without using tools.\n"
    "- Available tools: get_columns, get_preview, add_column, update_cell, add_row, add_empty_column, delete_rows, delete_column, summarize_total, highlight_rows, plot.\n"
    "- Prefer minimal, targeted tool usage. If the user asks for context, call get_columns or get_preview.\n"
    "- After using tools, return a concise confirmation of exactly what changed (e.g., column created, rows highlighted, totals, deleted rows/columns).\n"
    "- Never reveal chain-of-thought; only provide the outcome and brief rationale."
)


# ----------------------------- Service -----------------------------
class AIService:
    """
    AI assistant: uses LangChain + OpenAI if OPENAI_API_KEY is configured,
    otherwise falls back to a lightweight rule-based handler.
    """

    def __init__(self) -> None:
        self.use_llm = bool(os.getenv("OPENAI_API_KEY")) and ChatOpenAI is not None
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        # Simple in-memory conversational history (single-session demo)
        self.memory: List[Dict[str, str]] = []
        self.max_turns: int = 10  # keep last 10 user/assistant turns
        if self.use_llm:
            self._init_llm()

    def _init_llm(self) -> None:
        # temperature 0 for deterministic tool selection
        self.llm = cast(Any, ChatOpenAI)(model=self.model_name, temperature=0)  # type: ignore[misc]
        # bind tool calling
        self.llm_with_tools = self.llm.bind_tools(LANGCHAIN_TOOLS)
        # name -> callable mapping to run local side-effects
        self.tool_map = {t.name: t for t in LANGCHAIN_TOOLS}  # type: ignore[attr-defined]

    def _build_messages(self, user_text: str) -> List[Any]:
        """
        Build the LLM message list using short conversational memory.
        We store only user/assistant visible text (no chain-of-thought).
        """
        msgs: List[Any] = [SystemMessage(content=SYSTEM_PROMPT)]  # type: ignore
        # Use the last N turns (2 messages per turn)
        for m in self.memory[-(self.max_turns * 2):]:
            role = m.get("role")
            content = m.get("content", "")
            if not content:
                continue
            if role == "user":
                msgs.append(HumanMessage(content=content))  # type: ignore
            else:
                msgs.append(AIMessage(content=content))  # type: ignore
        msgs.append(HumanMessage(content=user_text))  # type: ignore
        return msgs

    def _remember(self, user_text: str, assistant_text: str) -> None:
        """Append a user/assistant turn and trim to window."""
        self.memory.append({"role": "user", "content": user_text})
        self.memory.append({"role": "assistant", "content": assistant_text})
        if len(self.memory) > self.max_turns * 2:
            self.memory = self.memory[-(self.max_turns * 2):]

    def _chat_llm(self, message: str) -> ChatResponse:
        # Invoke LLM with tools, execute tool calls, then ask LLM for the final message using ToolMessages.
        msgs: List[Any] = self._build_messages(message)
        actions: List[Dict[str, Any]] = []
 
        ai: Any = self.llm_with_tools.invoke(msgs)  # may contain tool_calls
        summary_parts: List[str] = []
        tool_messages: List[Any] = []
 
        tcalls = getattr(ai, "tool_calls", None)
        if tcalls:
            for tc in tcalls:
                name = tc.get("name")
                args = tc.get("args", {}) or {}
                tool_callable = self.tool_map.get(name)
                if not tool_callable:
                    continue
                try:
                    result = tool_callable.invoke(args) if hasattr(tool_callable, "invoke") else tool_callable.run(args)  # type: ignore
                except Exception as e:
                    result = json.dumps({"status": "error", "message": str(e)})
 
                # Parse tool output for UI actions or summaries
                try:
                    payload = json.loads(result)
                except Exception:
                    payload = {"raw": result}
 
                if isinstance(payload, dict):
                    if payload.get("action"):
                        a = payload["action"]
                        if a == "add_column":
                            actions.append({"type": "add_column", "name": payload.get("name"), "expression": payload.get("expression")})
                            summary_parts.append(f"Added column '{payload.get('name')}'.")
                        elif a == "highlight_rows":
                            rows = payload.get("rows", []) or []
                            actions.append({"type": "highlight_rows", "rows": rows, "condition": payload.get("condition")})
                            summary_parts.append(f"Highlighted {len(rows)} row(s) where {payload.get('condition')}.")
                        elif a == "summary":
                            actions.append({"type": "summary", "column": payload.get("column"), "total": payload.get("total")})
                            summary_parts.append(f"Total {payload.get('column')}: {payload.get('total')}.")
                        elif a == "plot":
                            actions.append({"type": "plot", "kind": payload.get("kind"), "x": payload.get("x"), "y": payload.get("y"), "figure": payload.get("figure")})
                            summary_parts.append(f"Generated {payload.get('kind')} chart of {payload.get('y')} by {payload.get('x')}.")
                        elif a == "update_cell":
                            actions.append({"type": "update_cell", "rowIndex": payload.get("rowIndex"), "field": payload.get("field"), "value": payload.get("value")})
                            summary_parts.append(f"Updated row {payload.get('rowIndex')} • {payload.get('field')} = {payload.get('value')}.")
                        elif a == "add_row":
                            actions.append({"type": "add_row", "rowIndex": payload.get("rowIndex"), "values": payload.get("values")})
                            summary_parts.append(f"Added row at index {payload.get('rowIndex')}.")
                        elif a == "add_empty_column":
                            actions.append({"type": "add_empty_column", "name": payload.get("name"), "fill": payload.get("fill")})
                            summary_parts.append(f"Added empty column '{payload.get('name')}'.")
                        elif a == "delete_rows":
                            rows = payload.get("rows", []) or []
                            actions.append({"type": "delete_rows", "rows": rows})
                            summary_parts.append(f"Deleted {len(rows)} row(s).")
                        elif a == "delete_column":
                            actions.append({"type": "delete_column", "name": payload.get("name"), "deleted": payload.get("deleted")})
                            summary_parts.append(f"Deleted column '{payload.get('name')}'." if payload.get("deleted") else f"Column '{payload.get('name')}' not found.")
                    else:
                        # Non-action tools: craft a concise summary fragment
                        if "columns" in payload:
                            try:
                                cols = ", ".join([str(c.get("field")) for c in payload["columns"] if isinstance(c, dict)])
                                summary_parts.append(f"Columns: {cols}.")
                            except Exception:
                                pass
                        if "rows" in payload:
                            try:
                                summary_parts.append(f"Preview rows: {len(payload['rows'])}.")
                            except Exception:
                                pass
 
                # Add ToolMessage so the model can produce a helpful final reply
                try:
                    tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc.get("id", "")))  # type: ignore
                except Exception:
                    pass
 
            # Ask LLM to produce the final concise response, informed by tool outputs
            try:
                ai_final: Any = self.llm.invoke([*msgs, ai, *tool_messages])
                content = (getattr(ai_final, "content", "") or "").strip()
            except Exception:
                content = ""
 
            if not content and not summary_parts and not actions:
                # No useful text/actions: try a second pass coercing tool use (no regex/rule fallback)
                try:
                    nudge = SystemMessage(("You MUST call one or more tools to satisfy the user's request. Choose the best tool(s) and call them now."))  # type: ignore
                    ai2: Any = self.llm_with_tools.invoke([*msgs, nudge])
                    tcalls2 = getattr(ai2, "tool_calls", None)
                    if tcalls2:
                        try:
                            ai_final2: Any = self.llm.invoke([*msgs, ai2])  # type: ignore
                            content2 = (getattr(ai_final2, "content", "") or "").strip()
                        except Exception:
                            content2 = ""
                        if content2:
                            content = content2
                except Exception:
                    pass

                if not content and not actions:
                    return ChatResponse(content="Unable to determine a tool-based action. Please rephrase the request.", actions=[])
            if not content:
                # Fall back to our summary if model didn't produce content
                content = " ".join(summary_parts) if summary_parts else "Completed requested action."
            elif summary_parts:
                # Combine LLM content with our structured summary
                content = f"{content} {' '.join(summary_parts)}".strip()

            # If no actions were produced, run a second pass coercing tool use
            if not actions:
                try:
                    nudge = SystemMessage(("You MUST call one or more tools to satisfy the user's request. Choose the best tool(s) and call them now."))  # type: ignore
                    ai2: Any = self.llm_with_tools.invoke([*msgs, nudge])
                    tcalls2 = getattr(ai2, "tool_calls", None)
                    if tcalls2:
                        for tc in tcalls2:
                            name2 = tc.get("name")
                            args2 = tc.get("args", {}) or {}
                            tool_callable2 = self.tool_map.get(name2)
                            if not tool_callable2:
                                continue
                            try:
                                result2 = tool_callable2.invoke(args2) if hasattr(tool_callable2, "invoke") else tool_callable2.run(args2)  # type: ignore
                            except Exception as e:
                                result2 = json.dumps({"status": "error", "message": str(e)})
                            try:
                                payload2 = json.loads(result2)
                            except Exception:
                                payload2 = {"raw": result2}
                            if isinstance(payload2, dict) and payload2.get("action"):
                                a2 = payload2["action"]
                                if a2 == "add_column":
                                    actions.append({"type": "add_column", "name": payload2.get("name"), "expression": payload2.get("expression")})
                                    summary_parts.append(f"Added column '{payload2.get('name')}'.")
                                elif a2 == "highlight_rows":
                                    rows2 = payload2.get("rows", []) or []
                                    actions.append({"type": "highlight_rows", "rows": rows2, "condition": payload2.get("condition")})
                                    summary_parts.append(f"Highlighted {len(rows2)} row(s) where {payload2.get('condition')}.")
                                elif a2 == "summary":
                                    actions.append({"type": "summary", "column": payload2.get("column"), "total": payload2.get("total")})
                                    summary_parts.append(f"Total {payload2.get('column')}: {payload2.get('total')}.")
                                elif a2 == "plot":
                                    actions.append({"type": "plot", "kind": payload2.get("kind"), "x": payload2.get("x"), "y": payload2.get("y"), "figure": payload2.get("figure")})
                                    summary_parts.append(f"Generated {payload2.get('kind')} chart of {payload2.get('y')} by {payload2.get('x')}.")
                                elif a2 == "update_cell":
                                    actions.append({"type": "update_cell", "rowIndex": payload2.get("rowIndex"), "field": payload2.get("field"), "value": payload2.get("value")})
                                    summary_parts.append(f"Updated row {payload2.get('rowIndex')} • {payload2.get('field')} = {payload2.get('value')}.")
                                elif a2 == "add_row":
                                    actions.append({"type": "add_row", "rowIndex": payload2.get("rowIndex"), "values": payload2.get("values")})
                                    summary_parts.append(f"Added row at index {payload2.get('rowIndex')}.")
                                elif a2 == "add_empty_column":
                                    actions.append({"type": "add_empty_column", "name": payload2.get("name"), "fill": payload2.get("fill")})
                                    summary_parts.append(f"Added empty column '{payload2.get('name')}'.")
                                elif a2 == "delete_rows":
                                    rows2 = payload2.get("rows", []) or []
                                    actions.append({"type": "delete_rows", "rows": rows2})
                                    summary_parts.append(f"Deleted {len(rows2)} row(s).")
                                elif a2 == "delete_column":
                                    actions.append({"type": "delete_column", "name": payload2.get("name"), "deleted": payload2.get("deleted")})
                                    summary_parts.append(f"Deleted column '{payload2.get('name')}'." if payload2.get("deleted") else f"Column '{payload2.get('name')}' not found.")
                        # produce final content for second pass
                        try:
                            ai_final2: Any = self.llm.invoke([*msgs, ai2])  # type: ignore
                            content2 = (getattr(ai_final2, "content", "") or "").strip()
                        except Exception:
                            content2 = ""
                        if content2:
                            content = f"{content} {content2}".strip() if content else content2
                except Exception as e:
                    logger.exception("second-pass tool forcing failed: %s", e)

            # As a last resort for chart-like queries, synthesize a plot using LLM-based column inference
            if not actions and any(k in message.lower() for k in ["chart", "plot", "graph", "pie", "bar", "line"]):
                try:
                    kind = "pie" if "pie" in message.lower() else ("line" if "line" in message.lower() else "bar")
                    cols = [str(c) for c in data_service.df.columns]
                    x, y = self._infer_xy_llm(message, cols)
                    if x and y:
                        fig = data_service.plot(x=x, y=y, kind=kind)
                        actions.append({"type": "plot", "kind": kind, "x": x, "y": y, "figure": fig})
                        summary_parts.append(f"Generated {kind} chart of {y} by {x}.")
                except Exception as e:
                    logger.exception("synthesized plot failed: %s", e)

            # If we added summary parts during synthesis and content didn't include them, append now
            if summary_parts and (not content or not all(p in content for p in summary_parts)):
                content = f"{content} {' '.join(summary_parts)}".strip() if content else " ".join(summary_parts)

            try:
                logger.info("llm final actions=%s", [a.get("type") for a in actions])
            except Exception:
                pass

            return ChatResponse(content=content, actions=actions, suggestions=self._suggest_next_queries(message, actions))
 
        # No tool calls: attempt a second pass coercing tool use; if still none, return an explicit error
        try:
            nudge = SystemMessage(("You MUST call at least one tool to satisfy the user's request. Choose the best tool and call it now."))  # type: ignore
            ai2: Any = self.llm_with_tools.invoke([*msgs, nudge])
            tcalls2 = getattr(ai2, "tool_calls", None)
            if tcalls2:
                # Minimal fallback: ask the model for a final short confirmation
                try:
                    ai_final2: Any = self.llm.invoke([*msgs, ai2])  # type: ignore
                    content2 = (getattr(ai_final2, "content", "") or "").strip()
                except Exception:
                    content2 = "Completed requested action."
                return ChatResponse(content=content2, actions=[], suggestions=self._suggest_next_queries(message, []))
        except Exception:
            pass
        return ChatResponse(content="Unable to determine a tool-based action. Please rephrase the request.", actions=[], suggestions=self._suggest_next_queries(message, []))

    def chat(self, message: str, mode: Optional[str] = "auto") -> ChatResponse:
        # Simple admin command to clear memory
        if message.strip().lower() in {"reset memory", "clear memory", "/reset"}:
            self.memory.clear()
            return ChatResponse(content="Memory cleared.", actions=[])

        mode_l = (mode or "auto").lower()
        resp: ChatResponse
        if mode_l == "llm":
            if not self.use_llm:
                raise RuntimeError("LLM mode requested but OPENAI_API_KEY is not configured")
            resp = self._chat_llm(message)
            self._remember(message, resp.content)
            return resp
        if mode_l == "rules":
            if not self.use_llm:
                raise RuntimeError("LLM mode requested but OPENAI_API_KEY is not configured")
            resp = self._chat_llm(message)
            self._remember(message, resp.content)
            return resp

        # auto: require LLM (no regex/rule fallback)
        if self.use_llm:
            resp = self._chat_llm(message)
        else:
            raise RuntimeError("LLM is required for AI inference. Set OPENAI_API_KEY in backend/.env")

        self._remember(message, resp.content)
        return resp

    # ----------------------------- Rule-based Fallback -----------------------------
    def _chat_rule_based(self, message: str, fallback_error: Optional[str] = None) -> ChatResponse:
        text = message.strip()
        low = text.lower()
        actions: List[Dict[str, Any]] = []
        preface = "" if not fallback_error else f"(LLM unavailable: {fallback_error})\n"

        # 1) Add column with expression
        if "add" in low and "column" in low and "=" in text:
            name = self._extract_column_name(text) or "new_column"
            expr = self._extract_expression_after_equals(text)
            if expr:
                data_service.add_column_expression(name, expr)
                actions.append({"type": "add_column", "name": name, "expression": expr})
                return ChatResponse(content=preface + f"Added column '{name}' with expression: {expr}", actions=actions)
            return ChatResponse(content=preface + "Could not find an expression after '='.", actions=[{"type": "error"}])

        # 2) Growth rate convenience
        if "growth rate" in low and ("q1" in low and "q2" in low):
            name = "growth_rate"
            expr = "(q2 - q1) / q1"
            data_service.add_column_expression(name, expr)
            actions.append({"type": "add_column", "name": name, "expression": expr})
            return ChatResponse(content=preface + "Added 'growth_rate' as (q2 - q1) / q1.", actions=actions)

        # 3) Summarize totals
        if "summarize" in low or "total" in low:
            target_col = None
            if "expense" in low or "expenses" in low:
                target_col = "cost" if "cost" in data_service.df.columns else None
            if target_col is None:
                for c in data_service.df.columns:
                    if str(c).lower() in {"revenue", "sales", "amount", "cost"}:
                        target_col = str(c)
                        break
            if target_col and target_col in data_service.df.columns:
                total_val = float(data_service.df[target_col].fillna(0).sum())
                return ChatResponse(content=preface + f"Total {target_col}: {total_val}", actions=[{"type": "summary", "column": target_col, "total": total_val}])

        # 4) Update a single cell (supports multiple phrasings)
        # a) "set revenue of row 2 to 300"
        # b) "update row 3 cost=500"
        # c) "set cell B3 to 42" (Excel-style)
        # d) "set row 3 column 2 to 5" (1-based column index)
        cols_list = [str(c) for c in data_service.df.columns]

        m = re.search(r"\b(?:set|update)\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:of|in)\s+row\s+(\d+)\s+to\s+(.+)$", text, flags=re.IGNORECASE)
        m2 = re.search(r"\b(?:set|update)\s+row\s+(\d+)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$", text, flags=re.IGNORECASE)
        m3 = re.search(r"\b(?:set|update)\s+cell\s+([A-Za-z]+)(\d+)\s+to\s+(.+)$", text, flags=re.IGNORECASE)
        m4 = re.search(r"\b(?:set|update)\s+row\s+(\d+)\s+column\s+(\d+)\s+to\s+(.+)$", text, flags=re.IGNORECASE)

        if m is not None:
            field = m.group(1).strip()
            row_n = int(m.group(2))
            row_index = self._to_zero_based_row(row_n)
            val_str = m.group(3).strip()
            value = self._parse_value(val_str)
            data_service.update_cell(row_index, field, value, {"source": "ai"})
            actions.append({"type": "update_cell", "rowIndex": row_index, "field": field, "value": value})
            return ChatResponse(content=preface + f"Updated row {row_n} • {field} = {value}", actions=actions)

        if m2 is not None:
            row_n = int(m2.group(1))
            row_index = self._to_zero_based_row(row_n)
            field = m2.group(2).strip()
            val_str = m2.group(3).strip()
            value = self._parse_value(val_str)
            data_service.update_cell(row_index, field, value, {"source": "ai"})
            actions.append({"type": "update_cell", "rowIndex": row_index, "field": field, "value": value})
            return ChatResponse(content=preface + f"Updated row {row_n} • {field} = {value}", actions=actions)

        if m3 is not None:
            letters = m3.group(1).strip()
            row_n = int(m3.group(2))
            row_index = self._to_zero_based_row(row_n)
            val_str = m3.group(3).strip()
            field = self._column_by_excel_letters(letters, cols_list)
            if not field:
                return ChatResponse(content=preface + f"Invalid cell reference: {letters}{row_n}.", actions=[{"type": "error"}])
            value = self._parse_value(val_str)
            data_service.update_cell(row_index, field, value, {"source": "ai"})
            actions.append({"type": "update_cell", "rowIndex": row_index, "field": field, "value": value})
            return ChatResponse(content=preface + f"Updated cell {letters}{row_n} • {field} = {value}", actions=actions)

        if m4 is not None:
            row_n = int(m4.group(1))
            col_n = int(m4.group(2))
            row_index = self._to_zero_based_row(row_n)
            col_index = max(0, col_n - 1)
            if col_index < 0 or col_index >= len(cols_list):
                return ChatResponse(content=preface + f"Column index {col_n} out of range.", actions=[{"type": "error"}])
            field = cols_list[col_index]
            val_str = m4.group(3).strip()
            value = self._parse_value(val_str)
            data_service.update_cell(row_index, field, value, {"source": "ai"})
            actions.append({"type": "update_cell", "rowIndex": row_index, "field": field, "value": value})
            return ChatResponse(content=preface + f"Updated row {row_n} • {field} = {value}", actions=actions)

        # 5) Add row, optionally with values: "add row with a=1, b='c'"
        m = re.search(r"\badd\s+row(?:\s+with\s+(.+))?$", text, flags=re.IGNORECASE)
        if m:
            values_raw = (m.group(1) or "").strip()
            values = self._parse_kv_pairs(values_raw) if values_raw else {}
            idx = data_service.add_row(values or None)
            actions.append({"type": "add_row", "rowIndex": idx, "values": values})
            return ChatResponse(content=preface + (f"Added row {idx}." if not values else f"Added row {idx} with values."), actions=actions)

        # 6) Add empty column: "add empty column notes" or "add column notes"
        if "=" not in text:
            m = re.search(r"\badd\s+(?:empty\s+)?column\s+([A-Za-z_][A-Za-z0-9_ ]*)(?:\s+filled\s+with\s+(.+))?$", text, flags=re.IGNORECASE)
            if m:
                name = m.group(1).strip().replace(" ", "_")
                fill = self._parse_value(m.group(2).strip()) if m.group(2) else None
                data_service.add_empty_column(name, fill)
                actions.append({"type": "add_empty_column", "name": name, "fill": fill})
                return ChatResponse(content=preface + f"Added empty column '{name}'.", actions=actions)

        # 7) Delete rows: e.g. "delete row 3", "delete rows 2,4,5", "delete rows 2-5" (1-based input)
        m = re.search(r"\bdelete\s+row\s+(\d+)\b", text, flags=re.IGNORECASE)
        mlist = re.search(r"\bdelete\s+rows?\s+([0-9,\-\s]+)\b", text, flags=re.IGNORECASE)
        if m is not None:
            rows_1_based: List[int] = [int(m.group(1))]
            rows_zero = [self._to_zero_based_row(n) for n in rows_1_based]
            deleted = data_service.delete_rows(rows_zero)
            actions.append({"type": "delete_rows", "rows": deleted})
            return ChatResponse(content=preface + f"Deleted {len(deleted)} row(s).", actions=actions)
        elif mlist is not None:
            token = (mlist.group(1) or "").strip()
            rows_1_based = self._parse_row_list_1_based(token)
            rows_zero = [self._to_zero_based_row(n) for n in rows_1_based]
            deleted = data_service.delete_rows(rows_zero)
            actions.append({"type": "delete_rows", "rows": deleted})
            return ChatResponse(content=preface + f"Deleted {len(deleted)} row(s).", actions=actions)

        # 8) Delete column by name/index/letter: "delete column notes", "delete column 2", "delete column B"
        m = re.search(r"\bdelete\s+column\s+([A-Za-z_][A-Za-z0-9_ ]+|\d+|[A-Za-z]+)\b", text, flags=re.IGNORECASE)
        if m:
            ident = m.group(1).strip()
            cols_list = [str(c) for c in data_service.df.columns]
            name = self._resolve_column_identifier(ident, cols_list)
            if not name:
                return ChatResponse(content=preface + f"Column '{ident}' not found.", actions=[{"type": "error"}])
            ok = data_service.delete_column(name)
            actions.append({"type": "delete_column", "name": name, "deleted": ok})
            return ChatResponse(content=preface + (f"Deleted column '{name}'." if ok else f"Column '{name}' not found."), actions=actions)

        # 9) Highlight condition
        if "highlight" in low:
            cond = self._extract_condition(text) or ("profit < 0" if ("profit" in low and "<" in low) else None)
            if not cond:
                return ChatResponse(content=preface + "Provide a condition to highlight, e.g., 'highlight rows where profit < 0'.", actions=[{"type": "error"}])
            rows = data_service.highlight_rows(cond)
            actions.append({"type": "highlight_rows", "rows": rows, "condition": cond})
            return ChatResponse(content=preface + f"Highlighted {len(rows)} row(s) where {cond}.", actions=actions)

        # 5) Plot/chart requests, e.g., "pie chart of revenue by region"
        if any(k in low for k in ["chart", "plot", "graph"]):
            kind = "bar"
            if "pie" in low:
                kind = "pie"
            elif "line" in low:
                kind = "line"
            elif "bar" in low:
                kind = "bar"

            cols = [str(c) for c in data_service.df.columns]
            logger.info("chart request: kind=%s text=%s cols=%s", kind, text, cols)
            x, y = self._infer_xy_llm(text, cols)

            if x and y:
                logger.info("chart inferred x=%s y=%s", x, y)
                try:
                    fig = data_service.plot(x=x, y=y, kind=kind)
                except Exception as e:
                    logger.exception("plot failed: %s", e)
                    return ChatResponse(content=preface + f"Plot failed: {e}", actions=[{"type": "error"}])
                actions.append({"type": "plot", "kind": kind, "x": x, "y": y, "figure": fig})
                return ChatResponse(content=preface + f"Generated {kind} chart of {y} by {x}.", actions=actions)
            else:
                return ChatResponse(
                    content=preface + f"Unable to infer columns for chart. Available columns: {', '.join(cols)}. "
                    "Try: 'pie chart of events by src_ip' or 'use events as values and src_ip as label'.",
                    actions=[{"type": "error"}],
                )

        # 6) Profit convenience without equals
        if "add" in low and "column" in low and "profit" in low and ("revenue" in low and "cost" in low):
            name = "profit"
            expr = "revenue - cost"
            data_service.add_column_expression(name, expr)
            actions.append({"type": "add_column", "name": name, "expression": expr})
            return ChatResponse(content=preface + "Added column 'profit' as revenue - cost.", actions=actions)

        return ChatResponse(
            content=preface + (
                "I can add computed columns, highlight rows by a condition, summarize totals, and make charts. "
                "Try: “add a column for profit = revenue - cost”, “highlight rows where profit < 0”, "
                "“summarize total revenue”, or “pie chart of revenue by region”."
            ),
            actions=[]
        )

    # ----------------------------- helpers -----------------------------
    def _extract_expression_after_equals(self, text: str) -> Optional[str]:
        if "=" not in text:
            return None
        return text.split("=", 1)[1].strip()

    def _extract_column_name(self, text: str) -> Optional[str]:
        m = re.search(r"\b(?:for|named|called)\s+([A-Za-z_][A-Za-z0-9_ ]*)", text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip().replace(" ", "_")
        if "=" in text:
            left = text.split("=", 1)[0]
            m2 = re.search(r"\bcolumn(?:\s+for)?\s+([A-Za-z_][A-Za-z0-9_ ]*)$", left.strip(), flags=re.IGNORECASE)
            if m2:
                return m2.group(1).strip().replace(" ", "_")
        return None

    def _extract_condition(self, text: str) -> Optional[str]:
        m = re.search(r"\bwhere\s+(.+)$", text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m2 = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*([<>!=]=?|==)\s*([A-Za-z0-9_().+\-*/ ]+)", text)
        if m2:
            return f"{m2.group(1)} {m2.group(2)} {m2.group(3)}"
        return None

    # ---------- value parsing helpers ----------
    def _parse_value(self, s: str) -> Any:
        """Coerce a string token into int/float/bool/None/str (strip surrounding quotes)."""
        t = s.strip()
        if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
            t = t[1:-1]
        low = t.lower()
        if low in {"true", "false"}:
            return low == "true"
        if low in {"null", "none"}:
            return None
        try:
            if "." in t:
                return float(t)
            return int(t)
        except Exception:
            return t

    def _parse_kv_pairs(self, s: str) -> Dict[str, Any]:
        """
        Parse comma-separated key=value pairs into a dict, e.g. a=1, b='x'.
        """
        out: Dict[str, Any] = {}
        if not s:
            return out
        parts = [p.strip() for p in s.split(",") if p.strip()]
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                out[k.strip()] = self._parse_value(v.strip())
        return out

    def _parse_row_list_1_based(self, token: str) -> List[int]:
        """
        Parse a row list token like '2,4,5' or '2-5' or '2 - 5' into a list of 1-based integers.
        """
        token = (token or "").strip()
        rows: List[int] = []
        if not token:
            return rows
        # Split by comma
        parts = [p.strip() for p in token.split(",") if p.strip()]
        for p in parts:
            # Range like 2-5
            m = re.match(r"^(\d+)\s*-\s*(\d+)$", p)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                if a <= b:
                    rows.extend(list(range(a, b + 1)))
                else:
                    rows.extend(list(range(b, a + 1)))
            else:
                # Single number
                try:
                    rows.append(int(p))
                except Exception:
                    continue
        # Deduplicate, keep order
        seen = set()
        uniq: List[int] = []
        for r in rows:
            if r not in seen:
                seen.add(r)
                uniq.append(r)
        return uniq

    def _json_sanitized(self, obj: Any) -> Any:
        """
        Ensure an object is JSON-serializable by round-tripping through json dumps/loads.
        Falls back to original object if conversion fails.
        """
        try:
            return json.loads(json.dumps(obj, default=str))
        except Exception:
            return obj

    # Index/coordinate helpers
    def _to_zero_based_row(self, n: int) -> int:
        """Treat user-provided row numbers as 1-based and convert to 0-based (clamped at 0)."""
        try:
            return max(0, int(n) - 1)
        except Exception:
            return 0

    def _excel_col_to_index(self, letters: str) -> Optional[int]:
        """Convert Excel letters (A,B,...,AA,AB,...) to 0-based column index."""
        t = (letters or "").strip().upper()
        if not t.isalpha():
            return None
        idx = 0
        for ch in t:
            idx = idx * 26 + (ord(ch) - ord("A") + 1)
        return idx - 1

    def _column_by_excel_letters(self, letters: str, columns: List[str]) -> Optional[str]:
        """Map Excel letters to a column name based on current DataFrame column order."""
        i = self._excel_col_to_index(letters)
        if i is None or i < 0 or i >= len(columns):
            return None
        return columns[i]

    def _resolve_column_identifier(self, ident: str, columns: List[str]) -> Optional[str]:
        """
        Resolve a column identifier that can be:
        - exact name (case-insensitive match)
        - 1-based numeric index (e.g., 2)
        - Excel-style letters (e.g., B, AA)
        """
        ident = (ident or "").strip()
        # numeric index?
        if ident.isdigit():
            idx = max(0, int(ident) - 1)
            return columns[idx] if 0 <= idx < len(columns) else None
        # letters?
        cidx = self._excel_col_to_index(ident)
        if cidx is not None and 0 <= cidx < len(columns):
            return columns[cidx]
        # name match (case-insensitive)
        low = ident.lower().replace(" ", "_")
        # try exact and underscored variants
        for c in columns:
            if c.lower() == ident.lower() or c.lower().replace(" ", "_") == low:
                return c
        return None

    def _infer_xy_llm(self, message: str, columns: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        Choose (x,y) using the configured LLM when available, constrained to provided columns.
        Falls back to a simple heuristic: first non-numeric column as x and first numeric column as y.
        """
        # Prepare dtype information to hint the LLM and for fallback
        try:
            import pandas as pd  # type: ignore
            dtypes_map = {str(c): str(data_service.df[c].dtype) for c in data_service.df.columns}
        except Exception:
            dtypes_map = {str(c): "unknown" for c in columns}

        # Prefer LLM-guided inference if available
        if self.use_llm:
            try:
                kind = "pie" if "pie" in message.lower() else ("line" if "line" in message.lower() else "bar")
                sys = SystemMessage((
                    "You select chart columns from a provided schema. "
                    "Return ONLY compact JSON of the form {\"x\":\"<column>\", \"y\":\"<column>\"} using EXACT names "
                    "from the provided columns. x must be a categorical label column; y must be a numeric values column "
                    "appropriate for the requested chart. No extra text."
                ))  # type: ignore
                human = HumanMessage((
                    f"User query: {message}\n"
                    f"Available columns: {columns}\n"
                    f"Column dtypes: {dtypes_map}\n"
                    f"Chart type: {kind}\n"
                    "Respond with JSON {\"x\":\"...\",\"y\":\"...\"}."
                ))  # type: ignore
                ai = self.llm.invoke([sys, human])  # type: ignore
                raw = getattr(ai, "content", "") or ""
                m = re.search(r"\{.*\}", raw, flags=re.S)
                if m:
                    obj = json.loads(m.group(0))
                    x = obj.get("x")
                    y = obj.get("y")
                    if x in columns and y in columns:
                        logger.info("infer_xy_llm picked x=%s y=%s", x, y)
                        return (x, y)
                logger.info("infer_xy_llm invalid/empty response: %s", raw[:200])
            except Exception as e:
                logger.exception("infer_xy_llm failed: %s", e)

        # Fallback: first non-numeric as x (label), first numeric as y (values)
        try:
            import pandas as pd  # type: ignore
            dtypes = data_service.df.dtypes
            x_fallback: Optional[str] = None
            y_fallback: Optional[str] = None
            for c in data_service.df.columns:
                if not pd.api.types.is_numeric_dtype(dtypes[c]):  # type: ignore
                    x_fallback = str(c)
                    break
            for c in data_service.df.columns:
                if pd.api.types.is_numeric_dtype(dtypes[c]):  # type: ignore
                    y_fallback = str(c)
                    break
            logger.info("infer_xy_llm fallback x=%s y=%s", x_fallback, y_fallback)
            return (x_fallback, y_fallback)
        except Exception:
            # Last resort: pick first two columns
            x0 = columns[0] if columns else None
            y0 = columns[1] if len(columns) > 1 else None
            logger.info("infer_xy_llm last-resort x=%s y=%s", x0, y0)
            return (x0, y0)

    # ---------- helpers for chart column inference ----------
    def _normalize_name(self, s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

    def _match_column(self, token: str, columns: List[str]) -> Optional[str]:
        """Match token against available columns using normalization and fuzzy contains."""
        norm_token = self._normalize_name(token)
        mapping = {self._normalize_name(c): c for c in columns}
        # exact normalized
        if norm_token in mapping:
            return mapping[norm_token]
        # exact case-insensitive
        for c in columns:
            if c.lower() == token.lower():
                return c
        # startswith or contains normalized
        for c in columns:
            norm_col = self._normalize_name(c)
            if norm_col.startswith(norm_token) or norm_token in norm_col:
                return c
        return None

    def _infer_xy(self, text: str, columns: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        Infer (x,y) for charts from flexible phrasing.
        x is the categorical/label column; y is numeric values.
        """
        lower = text.lower()

        # Pattern: "use <y> as values and <x> as label(s)"
        m = re.search(r"use\s+([a-z0-9_ ]+)\s+as\s+values?.*?\b(?:and|,)\s+([a-z0-9_ ]+)\s+as\s+(?:labels?|names?)", lower)
        if m:
            y_tok = m.group(1).strip()
            x_tok = m.group(2).strip()
            y = self._match_column(y_tok, columns)
            x = self._match_column(x_tok, columns)
            return (x, y)

        # Reverse: "use <x> as label(s) and <y> as values"
        m = re.search(r"use\s+([a-z0-9_ ]+)\s+as\s+(?:labels?|names?).*?\b(?:and|,)\s+([a-z0-9_ ]+)\s+as\s+values?", lower)
        if m:
            x_tok = m.group(1).strip()
            y_tok = m.group(2).strip()
            y = self._match_column(y_tok, columns)
            x = self._match_column(x_tok, columns)
            return (x, y)

        # Pattern: "of <y> by <x>"
        m = re.search(r"\b(?:of\s+)?([a-z0-9_ ]+)\s+by\s+([a-z0-9_ ]+)", lower)
        if m:
            y = self._match_column(m.group(1).strip(), columns)
            x = self._match_column(m.group(2).strip(), columns)
            return (x, y)

        # Pattern: "with <a> and <b>" -> if one looks numeric-like (events), treat as y
        m = re.search(r"\bwith\s+([a-z0-9_ ]+)\s+and\s+([a-z0-9_ ]+)", lower)
        if m:
            a = self._match_column(m.group(1).strip(), columns)
            b = self._match_column(m.group(2).strip(), columns)
            # Heuristic: prefer events/count/sum as y, src_ip/source/ip/region as x
            if a and ("event" in self._normalize_name(a) or "count" in self._normalize_name(a)):
                return (b, a)
            if b and ("event" in self._normalize_name(b) or "count" in self._normalize_name(b)):
                return (a, b)
            # Fallback: pick numeric candidate for y
            try:
                import pandas as pd  # type: ignore
                # Access df via data_service
                dtypes = getattr(data_service, "df").dtypes  # type: ignore
                def is_numeric(col: Optional[str]) -> bool:
                    try:
                        return col is not None and pd.api.types.is_numeric_dtype(dtypes[col])  # type: ignore
                    except Exception:
                        return False
                if a and is_numeric(a):
                    return (b, a)
                if b and is_numeric(b):
                    return (a, b)
            except Exception:
                pass
            return (a, b)

        # Pattern: "for <x>"
        m = re.search(r"\bfor\s+([a-z0-9_ ]+)", lower)
        if m:
            x = self._match_column(m.group(1).strip(), columns)
            # choose y by common names or first numeric
            y = self._match_column("events", columns) or self._match_column("revenue", columns)
            if not y:
                try:
                    import pandas as pd  # type: ignore
                    dtypes = getattr(data_service, "df").dtypes  # type: ignore
                    for c in columns:
                        try:
                            if pd.api.types.is_numeric_dtype(dtypes[c]):  # type: ignore
                                y = c
                                break
                        except Exception:
                            continue
                except Exception:
                    pass
            return (x, y)

        # Direct common defaults: src_ip/events, region/revenue etc.
        y = self._match_column("events", columns) or self._match_column("revenue", columns)
        x = self._match_column("src_ip", columns) or self._match_column("source ip", columns) or self._match_column("region", columns)
        return (x, y)
 
    def _suggest_next_queries(self, last_user_text: str, actions: List[Dict[str, Any]]) -> List[str]:
        """
        Return up to 2 short, context-aware next query suggestions.
        Prefer LLM suggestions when available; fall back to heuristics using current dataframe schema.
        """
        # Heuristic fallback builder
        def heuristic() -> List[str]:
            try:
                import pandas as pd  # type: ignore
                cols = [str(c) for c in data_service.df.columns]
                dtypes = data_service.df.dtypes
                x_label = None
                y_val = None
                # pick first non-numeric for x, first numeric for y
                for c in data_service.df.columns:
                    try:
                        if not pd.api.types.is_numeric_dtype(dtypes[c]):  # type: ignore
                            x_label = str(c)
                            break
                    except Exception:
                        continue
                for c in data_service.df.columns:
                    try:
                        if pd.api.types.is_numeric_dtype(dtypes[c]):  # type: ignore
                            y_val = str(c)
                            break
                    except Exception:
                        continue
                suggestions: List[str] = []
                if y_val:
                    suggestions.append(f"summarize total {y_val}")
                if x_label and y_val:
                    suggestions.append(f"pie chart of {y_val} by {x_label}")
                # ensure at least one generic option
                if not suggestions:
                    suggestions = ["highlight rows where profit < 0", "add a column for profit = revenue - cost"]
                return suggestions[:2]
            except Exception:
                return ["summarize total revenue", "pie chart of revenue by region"]
 
        # If an LLM is configured, ask it for 2 concise suggestions
        if self.use_llm:
            try:
                cols = [str(c) for c in data_service.df.columns]
                try:
                    import pandas as pd  # type: ignore
                    dtypes_map = {str(c): str(data_service.df[c].dtype) for c in data_service.df.columns}
                except Exception:
                    dtypes_map = {str(c): "unknown" for c in cols}
                acts_summary = ", ".join(sorted({a.get("type", "") for a in (actions or []) if isinstance(a, dict)})) or "none"
                sys = SystemMessage(  # type: ignore
                    "You suggest the next two helpful spreadsheet queries for the user. "
                    "Return exactly two short suggestions, each on its own line, no bullets, no extra text."
                )
                human = HumanMessage(  # type: ignore
                    f"User just asked: {last_user_text}\n"
                    f"Recent actions: {acts_summary}\n"
                    f"Available columns: {cols}\n"
                    f"Column dtypes: {dtypes_map}\n"
                    "Output exactly two suggestions, one per line."
                )
                ai = self.llm.invoke([sys, human])  # type: ignore
                raw = (getattr(ai, "content", "") or "").strip()
                lines = [l.strip(" -•\t") for l in raw.splitlines() if l.strip()]
                # keep unique and short
                uniq: List[str] = []
                for l in lines:
                    if l not in uniq:
                        uniq.append(l)
                    if len(uniq) == 2:
                        break
                if len(uniq) >= 1:
                    # if only 1 line returned, supplement with heuristic second
                    if len(uniq) == 1:
                        for h in heuristic():
                            if h not in uniq:
                                uniq.append(h)
                                break
                    return uniq[:2]
            except Exception:
                pass
 
        # Fallback
        return heuristic()
 
 # Global singleton
ai_service = AIService()
