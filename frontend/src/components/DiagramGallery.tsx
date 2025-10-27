import PlotViewer from "./PlotViewer";

type PlotItem = {
  id: string;
  figure: any;
  title?: string;
};

type Props = {
  plots: PlotItem[];
  onClosePlot: (id: string) => void;
  onClearAll?: () => void; // retained for compatibility; header controls live in App
};

export default function DiagramGallery({ plots, onClosePlot }: Props) {
  return (
    <div className="h-full w-full flex flex-col min-h-0">
      <div className="flex-1 min-h-0 overflow-y-auto space-y-4 custom-scrollbar">
        {plots.length === 0 ? (
          <div className="text-center p-8">
            <div className="text-gray-500 text-sm">
              No diagrams yet. Ask in chat, e.g., "pie chart of revenue by region".
            </div>
          </div>
        ) : (
          plots.map((p) => (
            <PlotViewer
              key={p.id}
              figure={p.figure}
              title={p.title}
              onClose={() => onClosePlot(p.id)}
            />
          ))
        )}
      </div>
    </div>
  );
}
