"use client";

import { useMemo } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer
} from "recharts";

interface RoadData {
  by_road_type: Array<{
    RoadType: string;
    mean: number;
    median: number;
    std: number;
    count: number;
  }>;
  by_lanes: Array<{
    NumberofLanes: number;
    mean: number;
    std: number;
    count: number;
  }>;
}

interface Props {
  data: RoadData | null;
}

const ROAD_COLORS: Record<string, string> = {
  Residential: "#10b981",
  Street: "#f59e0b",
  Highway: "#ef4444",
};

export default function RoadCapacity({ data }: Props) {
  const utilization = useMemo(() => {
    if (!data?.by_road_type) return [];
    const maxDemand = Math.max(...data.by_road_type.map((d) => d.mean), 0.001);
    return data.by_road_type.map((d) => ({
      ...d,
      utilization: Math.min(100, (d.mean / maxDemand) * 100),
    }));
  }, [data]);

  if (!data || (!data.by_road_type?.length && !data.by_lanes?.length)) {
    return (
      <div className="glass-card p-6">
        <h2 className="text-xl font-bold gradient-text mb-4">Road Capacity</h2>
        <div className="h-64 flex items-center justify-center text-slate-500">
          <p>No road data available. Run the ML pipeline and start the API.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-card p-6 chart-container">
      <h2 className="text-xl font-bold gradient-text mb-4">Road Capacity Insights</h2>

      {/* Utilization Gauges */}
      <div className="space-y-4 mb-6">
        {utilization.map((road) => (
          <div key={road.RoadType}>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-slate-300 font-medium">
                {road.RoadType}
              </span>
              <div className="flex items-center gap-3">
                <span className="text-slate-500">
                  {road.count.toLocaleString()} segments
                </span>
                <span className="font-semibold" style={{ color: ROAD_COLORS[road.RoadType] || "#6366f1" }}>
                  {road.mean.toFixed(4)}
                </span>
              </div>
            </div>
            <div className="h-3 bg-slate-800/50 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-1000 ease-out"
                style={{
                  width: `${road.utilization}%`,
                  background: `linear-gradient(90deg, ${ROAD_COLORS[road.RoadType] || "#6366f1"}80, ${ROAD_COLORS[road.RoadType] || "#6366f1"})`,
                  boxShadow: `0 0 10px ${ROAD_COLORS[road.RoadType] || "#6366f1"}40`,
                }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Lane Analysis */}
      {data.by_lanes && data.by_lanes.length > 0 && (
        <>
          <h3 className="text-sm font-semibold text-slate-400 mb-3 uppercase tracking-wider">
            Demand by Number of Lanes
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={data.by_lanes} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,100,255,0.1)" />
              <XAxis dataKey="NumberofLanes" stroke="#94a3b8" tick={{ fontSize: 12 }}
                tickFormatter={(v) => `${v} Lane${v > 1 ? "s" : ""}`} />
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
                labelFormatter={(v) => `${v} Lane${Number(v) > 1 ? "s" : ""}`}
              />
              <Bar dataKey="mean" fill="#06b6d4" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  );
}
