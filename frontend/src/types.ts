export type Scalar = string | number | boolean | null;

export interface ColumnDef {
    field: string;
    type?: "numeric" | "string" | "date" | "formula" | string;
}

export type Row = Record<string, any>;

export interface DataResponse {
    columns: ColumnDef[];
    rows: Row[];
    metadata: {
        rowCount: number;
        columnCount: number;
        [k: string]: any;
    };
}

export interface CellAttributes {
    type?: string;
    tags?: string[];
    source?: "manual" | "ai" | string;
    formula?: string;
}

export interface CellUpdateRequest {
    rowIndex: number;
    field: string;
    value: any;
    attributes?: CellAttributes;
}

export interface AddColumnRequest {
    name: string;
    expression: string;
}

export interface HighlightRequest {
    condition: string;
}

export interface PlotRequest {
    x: string;
    y?: string;
    kind?: "bar" | "line" | "pie" | string;
}

export interface ChatRequest {
    message: string;
    mode?: "auto" | "llm" | "rules";
}

export interface ChatAction {
    type: string;
    // flexible payload depending on action
    [k: string]: any;
}

export interface ChatResponse {
    content: string;
    actions: ChatAction[];
    suggestions?: string[];
}

export type Role = "user" | "assistant";

export interface ChatMessage {
    role: Role;
    content: string;
    actions?: ChatAction[];
}
