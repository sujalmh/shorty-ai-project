import { useCallback, useState } from "react";
import ChatPanel from "./components/ChatPanel";
import SheetGrid from "./components/SheetGrid";
import DiagramGallery from "./components/DiagramGallery";

function App() {
  const [highlightedRows, setHighlightedRows] = useState<number[]>([]);
  const [reloadToken, setReloadToken] = useState<number>(0);
  const [activeTab, setActiveTab] = useState<"sheet" | "diagrams">("sheet");
  const [plots, setPlots] = useState<Array<{ id: string; figure: any; title?: string }>>([]);

  const handleHighlightRows = useCallback((rows: number[]) => {
    setHighlightedRows(rows);
  }, []);

  const handleReloadSheet = useCallback(() => {
    setReloadToken((t) => t + 1);
  }, []);

  const handleAddPlot = useCallback((plot: { figure: any; title?: string }) => {
    const id =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? (crypto as any).randomUUID()
        : `${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
    setPlots((prev) => [{ id, figure: plot.figure, title: plot.title }, ...prev]);
    setActiveTab("diagrams");
  }, []);

  const handleClosePlot = useCallback((id: string) => {
    setPlots((prev) => prev.filter((p) => p.id !== id));
  }, []);

  const handleClearPlots = useCallback(() => {
    setPlots([]);
  }, []);

  return (
    <div className="h-screen w-screen overflow-hidden">
      <div className="h-full grid grid-cols-1 md:grid-cols-3 gap-4 p-4">
        {/* Left: Chat */}
        <div className="h-full">
          <div className="h-full glass-dark flex flex-col min-h-0 component-padding">
            <ChatPanel
              onHighlightRows={handleHighlightRows}
              onReloadSheet={handleReloadSheet}
              onAddPlot={handleAddPlot}
            />
          </div>
        </div>

        {/* Right: Sheet / Diagrams */}
        <div className="h-full md:col-span-2 min-h-0">
          <div className="h-full glass-light flex flex-col min-h-0 overflow-hidden">
            <div className="sticky top-0 z-10 header-glass flex items-center justify-between px-4 py-3">
              <div className="flex items-center gap-2">
                <button
                  className={`tab-btn ${
                    activeTab === "sheet"
                      ? "tab-btn-active"
                      : "tab-btn-inactive"
                  }`}
                  onClick={() => setActiveTab("sheet")}
                >
                  Sheet
                </button>
                <button
                  className={`tab-btn ${
                    activeTab === "diagrams"
                      ? "tab-btn-active"
                      : "tab-btn-inactive"
                  }`}
                  onClick={() => setActiveTab("diagrams")}
                >
                  Diagrams
                  {plots.length > 0 && (
                    <span className="badge ml-1.5">{plots.length}</span>
                  )}
                </button>
              </div>
              {activeTab === "diagrams" && plots.length > 0 && (
                <button
                  className="btn-secondary"
                  onClick={handleClearPlots}
                  title="Remove all diagrams"
                >
                  Clear All
                </button>
              )}
            </div>

            <div className="flex-1 min-h-0 p-4 overflow-y-auto custom-scrollbar">
              {activeTab === "sheet" ? (
                <SheetGrid
                  reloadToken={reloadToken}
                  highlightedRows={highlightedRows}
                  onRefetch={() => {}}
                />
              ) : (
                <DiagramGallery plots={plots} onClosePlot={handleClosePlot} onClearAll={handleClearPlots} />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
