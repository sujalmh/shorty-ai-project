import { api } from "./api";
import type {
    DataResponse,
    CellUpdateRequest,
    AddColumnRequest,
    HighlightRequest,
    PlotRequest,
    ChatRequest,
    ChatResponse,
} from "../types";

export async function getData(): Promise<DataResponse> {
    const { data } = await api.get<DataResponse>("/data");
    return data;
}

export async function importData(rows: Record<string, any>[]): Promise<DataResponse> {
    const { data } = await api.post<DataResponse>("/import", { rows });
    return data;
}

export async function updateCell(payload: CellUpdateRequest): Promise<DataResponse> {
    const { data } = await api.post<DataResponse>("/cell", payload);
    return data;
}

export async function addColumn(payload: AddColumnRequest): Promise<DataResponse> {
    const { data } = await api.post<DataResponse>("/add-column", payload);
    return data;
}

export async function importCsv(file: File, encoding?: string): Promise<DataResponse> {
    const fd = new FormData();
    fd.append("file", file);
    if (encoding) fd.append("encoding", encoding);
    const { data } = await api.post<DataResponse>("/import-csv", fd, {
        headers: { "Content-Type": "multipart/form-data" },
    });
    return data;
}

export async function importExcel(file: File, sheetName?: string): Promise<DataResponse> {
    const fd = new FormData();
    fd.append("file", file);
    if (sheetName) fd.append("sheet_name", sheetName);
    const { data } = await api.post<DataResponse>("/import-excel", fd, {
        headers: { "Content-Type": "multipart/form-data" },
    });
    return data;
}

export async function addRowSimple(): Promise<DataResponse> {
    const { data } = await api.post<DataResponse>("/rows/add");
    return data;
}

export async function addEmptyColumn(name: string, fill?: string): Promise<DataResponse> {
    const fd = new FormData();
    fd.append("name", name);
    if (fill !== undefined) fd.append("fill", fill);
    const { data } = await api.post<DataResponse>("/columns/add-empty", fd);
    return data;
}

export async function highlight(payload: HighlightRequest): Promise<{ rows: number[] }> {
    const { data } = await api.post<{ rows: number[] }>("/highlight", payload);
    return data;
}

export async function plot(payload: PlotRequest): Promise<any> {
    const { data } = await api.post<any>("/plot", payload);
    return data;
}

export async function chat(payload: ChatRequest): Promise<ChatResponse> {
    const { data } = await api.post<ChatResponse>("/chat", payload);
    return data;
}
