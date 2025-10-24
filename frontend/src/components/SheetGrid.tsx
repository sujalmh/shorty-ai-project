import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AgGridReact } from "ag-grid-react";
import type { ColDef, ColGroupDef, ValueSetterParams } from "ag-grid-community";
import { ModuleRegistry, AllCommunityModule } from "ag-grid-community";
import { addColumn, getData, updateCell, importCsv, importExcel, addRowSimple, addEmptyColumn } from "../lib/endpoints";
import type { ColumnDef as ColMeta, DataResponse } from "../types";

// AG Grid styles
import "ag-grid-community/styles/ag-theme-quartz.css";

// Register AG Grid Community modules
ModuleRegistry.registerModules([AllCommunityModule]);

type Props = {
  reloadToken: number;
  highlightedRows: number[];
  onRefetch?: () => void;
};

export default function SheetGrid({ reloadToken, highlightedRows, onRefetch }: Props) {
  const [rowData, setRowData] = useState<any[]>([]);
  const [columnDefs, setColumnDefs] = useState<(ColDef | ColGroupDef)[]>([]);
  const defaultColDef = useMemo<ColDef>(
    () => ({
      editable: true,
      sortable: true,
      filter: true,
      resizable: true,
      floatingFilter: true,
      flex: 1,
    }),
    []
  );

  // simple add column controls
  const [newColName, setNewColName] = useState("profit");
  const [newColExpr, setNewColExpr] = useState("revenue - cost");
  const [busy, setBusy] = useState(false);
  const [importBusy, setImportBusy] = useState(false);
  const [newEmptyCol, setNewEmptyCol] = useState("");
  const csvInputRef = useRef<HTMLInputElement | null>(null);
  const excelInputRef = useRef<HTMLInputElement | null>(null);

  const onPickCsv = useCallback(() => csvInputRef.current?.click(), []);
  const onPickExcel = useCallback(() => excelInputRef.current?.click(), []);

  const buildColDefs = useCallback((cols: ColMeta[]): ColDef[] => {
    return cols.map((c) => ({
      headerName: c.field,
      field: c.field,
      editable: true,
      // valueSetter to keep grid state responsive while backend update applies
      valueSetter: (params: ValueSetterParams) => {
        params.data[c.field] = params.newValue;
        return true;
      },
      cellClass: (p) => (c.type === "numeric" ? "text-right" : ""),
    }));
  }, []);

  const refetch = useCallback(async () => {
    const data: DataResponse = await getData();
    setColumnDefs(buildColDefs(data.columns));
    setRowData(data.rows);
    onRefetch?.();
  }, [buildColDefs, onRefetch]);

  // Handlers that depend on refetch
  const onCsvChange = useCallback(
    async (e: any) => {
      const f: File | undefined = e.target.files?.[0];
      if (!f) return;
      setImportBusy(true);
      try {
        await importCsv(f);
        await refetch();
      } finally {
        setImportBusy(false);
        e.target.value = "";
      }
    },
    [refetch]
  );

  const onExcelChange = useCallback(
    async (e: any) => {
      const f: File | undefined = e.target.files?.[0];
      if (!f) return;
      setImportBusy(true);
      try {
        await importExcel(f);
        await refetch();
      } finally {
        setImportBusy(false);
        e.target.value = "";
      }
    },
    [refetch]
  );

  const handleAddRow = useCallback(async () => {
    setBusy(true);
    try {
      await addRowSimple();
      await refetch();
    } finally {
      setBusy(false);
    }
  }, [refetch]);

  const handleAddEmptyColumn = useCallback(async () => {
    if (!newEmptyCol.trim()) return;
    setBusy(true);
    try {
      await addEmptyColumn(newEmptyCol.trim());
      setNewEmptyCol("");
      await refetch();
    } finally {
      setBusy(false);
    }
  }, [newEmptyCol, refetch]);

  useEffect(() => {
    refetch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reloadToken]);

  const onCellValueChanged = useCallback(async (p: any) => {
    const rowIndex: number = p.node?.rowIndex ?? 0;
    const field: string = p.colDef.field!;
    const value = p.newValue;
    try {
      await updateCell({
        rowIndex,
        field,
        value,
        attributes: { source: "manual" },
      });
      // optional: re-fetch to ensure server truth
      await refetch();
    } catch (e) {
      console.error("Update failed", e);
    }
  }, [refetch]);

  const rowClassRules = useMemo(
    () => ({
      "bg-red-100 dark:bg-red-900/40": (params: any) => highlightedRows.includes(params.node.rowIndex),
    }),
    [highlightedRows]
  );

  const handleAddColumn = useCallback(async () => {
    if (!newColName.trim() || !newColExpr.trim()) return;
    setBusy(true);
    try {
      await addColumn({ name: newColName.trim(), expression: newColExpr.trim() });
      await refetch();
    } finally {
      setBusy(false);
    }
  }, [newColName, newColExpr, refetch]);

  return (
    <div className="h-full w-full flex flex-col">
      <div className="sticky top-0 z-10 header-glass px-3 py-2 space-y-2">
        {/* Row 1: Import + manual edits */}
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">Data:</span>
          <button
            className="btn-soft btn-glow bg-white hover:bg-slate-50 text-gray-700 text-sm disabled:opacity-50"
            onClick={onPickCsv}
            disabled={importBusy}
            title="Import CSV"
          >
            {importBusy ? "Importing..." : "Import CSV"}
          </button>
          <button
            className="btn-soft btn-glow bg-white hover:bg-slate-50 text-gray-700 text-sm disabled:opacity-50"
            onClick={onPickExcel}
            disabled={importBusy}
            title="Import Excel"
          >
            {importBusy ? "Importing..." : "Import Excel"}
          </button>
          <input ref={csvInputRef} type="file" accept=".csv,text/csv" className="hidden" onChange={onCsvChange} />
          <input ref={excelInputRef} type="file" accept=".xlsx,.xls,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel" className="hidden" onChange={onExcelChange} />

          <div className="mx-2 h-5 w-px bg-gray-300 dark:bg-gray-700" />
          <button
            className="btn-soft btn-glow bg-white hover:bg-slate-50 text-gray-700 text-sm disabled:opacity-50"
            onClick={handleAddRow}
            disabled={busy}
            title="Append empty row"
          >
            Add Row
          </button>

          <div className="flex items-center gap-2">
            <input
              className="px-2 py-1 text-sm rounded border border-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-400/40 bg-white text-gray-900"
              placeholder="new column name"
              value={newEmptyCol}
              onChange={(e) => setNewEmptyCol(e.target.value)}
            />
            <button
              className="btn-soft btn-glow bg-white hover:bg-slate-50 text-gray-700 text-sm disabled:opacity-50"
              onClick={handleAddEmptyColumn}
              disabled={busy}
              title="Add empty column"
            >
              Add Empty Column
            </button>
          </div>
        </div>

        {/* Row 2: Add computed column */}
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">Add Column:</span>
          <input
            className="px-2 py-1 text-sm rounded border border-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-400/40 bg-white text-gray-900"
            placeholder="name"
            value={newColName}
            onChange={(e) => setNewColName(e.target.value)}
          />
          <input
            className="px-2 py-1 text-sm rounded border border-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-400/40 bg-white text-gray-900 flex-1"
            placeholder="expression e.g. revenue - cost"
            value={newColExpr}
            onChange={(e) => setNewColExpr(e.target.value)}
          />
          <button
            className="btn-soft btn-glow bg-indigo-600 text-white text-sm rounded disabled:opacity-50"
            onClick={handleAddColumn}
            disabled={busy}
          >
            {busy ? "Adding..." : "Add"}
          </button>
        </div>
      </div>
      <div className="ag-theme-quartz flex-1 min-h-0" role="region" aria-label="Spreadsheet grid" style={{ width: "100%", height: "100%" }}>
        <AgGridReact
          rowData={rowData}
          columnDefs={columnDefs}
          defaultColDef={defaultColDef}
          onCellValueChanged={onCellValueChanged}
          rowClassRules={rowClassRules}
          animateRows
        />
      </div>
    </div>
  );
}
