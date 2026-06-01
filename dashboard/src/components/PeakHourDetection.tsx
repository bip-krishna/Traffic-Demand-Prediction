"use client";

import { useMemo } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell
} from "recharts";

interface PeakData {
  peak: number[];
  off_peak: number[];
  peak_avg_demand: number;
  off_peak_avg_demand: number;
}

interface HourlyData {
  hour: number;
  mean: number;
  std: number;
  count: number;
}

interface Props {
  peakData: PeakData | null;
  hourlyData: HourlyData[];
}

export default function PeakHourDetection({ peakData, hourlyData }: Props) {
  const chartData = useMemo(() => {
    if (!hourlyData || hourlyData.length === 0) return [];
    const peaks = new Set(peakData?.peak || []);
    return hourlyData.map((d) => ({
      ...d,
      isPeak: peaks.has(d.hour),
      label: `${d.hour}:00`,
    }));
  }, [hourlyData, peakData]);

  if (!peakData || !hourlyData || hourlyData.length === 0) {
    return (
      <div className="glass-card p-6">
        <h2 className="text-xl font-bold gradient-text mb-4">Peak Hour Detection</h2>
        <div className="h-64 flex items-center justify-center text-slate-500">
          <p>No peak hour data available. Run the ML pipeline and start the API.</p>
        </div>
      </div>
    );
  }

  const peakRatio = peakData.peak_avg_demand / (peakData.off_peak_avg_demand || 1);

  return (
    <div className="glass-card p-6 chart-container">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold gradient-text">Peak Hour Detection</h2>
        <div className="flex items-center gap-3 text-sm">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-sm bg-red-500"></span>
            <span className="text-slate-400">Peak</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-sm bg-indigo-500"></span>
            <span className="text-slate-400">Normal</span>
          </span>
        </div>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-3 gap-3 mb-5">
        <div className="bg-red-500/10 rounded-xl p-3 text-center border border-red-500/20">
          <p className="text-xs text-red-400 mb-1">Peak Hours</p>
          <p className="text-sm font-bold text-red-400">
            {peakData.peak.map((h) => `${h}:00`).join(", ")}
          </p>
        </div>
        <div className="bg-indigo-500/10 rounded-xl p-3 text-center border border-indigo-500/20">
          <p className="text-xs text-indigo-400 mb-1">Peak Avg</p>
          <p className="text-lg font-bold text-indigo-400">
            {peakData.peak_avg_demand.toFixed(4)}
          </p>
        </div>
        <div className="bg-amber-500/10 rounded-xl p-3 text-center border border-amber-500/20">
          <p className="text-xs text-amber-400 mb-1">Peak Ratio</p>
          <p className="text-lg font-bold text-amber-400">{peakRatio.toFixed(1)}×</p>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,100,255,0.1)" />
          <XAxis dataKey="label" stroke="#94a3b8" tick={{ fontSize: 10 }} />
          <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }}
            tickFormatter={(v) => v.toFixed(3)} />
          <Tooltip
            contentStyle={{
              backgroundColor: "rgba(15,15,40,0.95)",
              border: "1px solid rgba(100,100,255,0.2)",
              borderRadius: "12px",
              color: "#e0e0ff",
            }}
            formatter={(value: number) => [value.toFixed(4), "Avg Demand"]}
          />
          <Bar dataKey="mean" radius={[4, 4, 0, 0]}>
            {chartData.map((entry, idx) => (
              <Cell
                key={idx}
                fill={entry.isPeak ? "#ef4444" : "#6366f1"}
                fillOpacity={entry.isPeak ? 0.9 : 0.6}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
