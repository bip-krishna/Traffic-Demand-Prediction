"use client";

import { useMemo } from "react";

interface HeatmapPoint {
  geohash: string;
  demand: number;
  latitude: number;
  longitude: number;
}

interface Props {
  data: HeatmapPoint[];
}

export default function TrafficHeatmap({ data }: Props) {
  const stats = useMemo(() => {
    if (!data || data.length === 0) return null;
    const demands = data.map((d) => d.demand);
    const maxDemand = Math.max(...demands);
    const minDemand = Math.min(...demands);
    const avgDemand = demands.reduce((a, b) => a + b, 0) / demands.length;
    // Sort by demand descending
    const sorted = [...data].sort((a, b) => b.demand - a.demand);
    const hotspots = sorted.slice(0, 10);
    const coldspots = sorted.slice(-5);
    return { maxDemand, minDemand, avgDemand, hotspots, coldspots, total: data.length };
  }, [data]);

  if (!data || data.length === 0 || !stats) {
    return (
      <div className="glass-card p-6">
        <h2 className="text-xl font-bold gradient-text mb-4">Traffic Heatmap</h2>
        <div className="h-64 flex items-center justify-center text-slate-500">
          <p>No heatmap data available. Run the ML pipeline and start the API.</p>
        </div>
      </div>
    );
  }

  const getColor = (demand: number) => {
    const ratio = (demand - stats.minDemand) / (stats.maxDemand - stats.minDemand + 0.0001);
    if (ratio > 0.8) return "from-red-500 to-red-600";
    if (ratio > 0.6) return "from-orange-500 to-orange-600";
    if (ratio > 0.4) return "from-yellow-500 to-yellow-600";
    if (ratio > 0.2) return "from-green-500 to-green-600";
    return "from-emerald-500 to-emerald-600";
  };

  const getBarWidth = (demand: number) => {
    const ratio = (demand - stats.minDemand) / (stats.maxDemand - stats.minDemand + 0.0001);
    return `${Math.max(8, ratio * 100)}%`;
  };

  return (
    <div className="glass-card p-6 chart-container">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold gradient-text">Traffic Demand Heatmap</h2>
        <span className="text-sm text-slate-500">{stats.total} locations</span>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-3 mb-5">
        <div className="bg-red-500/10 rounded-xl p-3 text-center border border-red-500/20">
          <p className="text-xs text-red-400 mb-1">Highest</p>
          <p className="text-lg font-bold text-red-400">{stats.maxDemand.toFixed(4)}</p>
        </div>
        <div className="bg-indigo-500/10 rounded-xl p-3 text-center border border-indigo-500/20">
          <p className="text-xs text-indigo-400 mb-1">Average</p>
          <p className="text-lg font-bold text-indigo-400">{stats.avgDemand.toFixed(4)}</p>
        </div>
        <div className="bg-emerald-500/10 rounded-xl p-3 text-center border border-emerald-500/20">
          <p className="text-xs text-emerald-400 mb-1">Lowest</p>
          <p className="text-lg font-bold text-emerald-400">{stats.minDemand.toFixed(4)}</p>
        </div>
      </div>

      {/* Top Hotspots */}
      <h3 className="text-sm font-semibold text-slate-400 mb-3 uppercase tracking-wider">
        🔥 Top 10 Hotspots
      </h3>
      <div className="space-y-2 max-h-[260px] overflow-y-auto pr-2">
        {stats.hotspots.map((point, i) => (
          <div key={point.geohash} className="flex items-center gap-3">
            <span className="text-xs text-slate-500 w-5 text-right">{i + 1}</span>
            <code className="text-xs text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded font-mono w-20 text-center">
              {point.geohash}
            </code>
            <div className="flex-1 h-5 bg-slate-800/50 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full bg-gradient-to-r ${getColor(point.demand)} transition-all duration-500`}
                style={{ width: getBarWidth(point.demand) }}
              />
            </div>
            <span className="text-xs text-slate-300 font-mono w-16 text-right">
              {point.demand.toFixed(4)}
            </span>
          </div>
        ))}
      </div>

      {/* Gradient Legend */}
      <div className="mt-4 flex items-center gap-2">
        <span className="text-xs text-slate-500">Low</span>
        <div className="flex-1 h-2 rounded-full bg-gradient-to-r from-emerald-500 via-yellow-500 to-red-500" />
        <span className="text-xs text-slate-500">High</span>
      </div>
    </div>
  );
}
