"use client";

import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine
} from "recharts";

interface ForecastData {
  hour: number;
  predicted_demand: number;
  confidence_low: number;
  confidence_high: number;
}

interface Props {
  data: ForecastData[];
}

export default function DemandForecast({ data }: Props) {
  if (!data || data.length === 0) {
    return (
      <div className="glass-card p-6">
        <h2 className="text-xl font-bold gradient-text mb-4">Demand Forecast</h2>
        <div className="h-64 flex items-center justify-center text-slate-500">
          <p>No forecast data available. Run the ML pipeline and start the API.</p>
        </div>
      </div>
    );
  }

  const peakHour = data.reduce((max, d) =>
    d.predicted_demand > max.predicted_demand ? d : max, data[0]);

  return (
    <div className="glass-card p-6 chart-container">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold gradient-text">24-Hour Demand Forecast</h2>
        <div className="flex items-center gap-2 text-sm">
          <span className="inline-block w-3 h-3 rounded-full bg-indigo-500"></span>
          <span className="text-slate-400">Predicted</span>
          <span className="inline-block w-3 h-3 rounded-full bg-indigo-500/20 ml-2"></span>
          <span className="text-slate-400">Confidence</span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="forecastGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#6366f1" stopOpacity={0.4} />
              <stop offset="100%" stopColor="#6366f1" stopOpacity={0.05} />
            </linearGradient>
            <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.15} />
              <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,100,255,0.1)" />
          <XAxis dataKey="hour" stroke="#94a3b8" tick={{ fontSize: 12 }}
            tickFormatter={(v) => `${v}:00`} />
          <YAxis stroke="#94a3b8" tick={{ fontSize: 12 }}
            tickFormatter={(v) => v.toFixed(3)} />
          <Tooltip
            contentStyle={{
              backgroundColor: "rgba(15,15,40,0.95)",
              border: "1px solid rgba(100,100,255,0.2)",
              borderRadius: "12px",
              color: "#e0e0ff",
            }}
            formatter={(value: number) => [value.toFixed(4), ""]}
            labelFormatter={(label) => `${label}:00`}
          />
          <Area type="monotone" dataKey="confidence_high" stroke="none"
            fill="url(#confidenceGradient)" />
          <Area type="monotone" dataKey="confidence_low" stroke="none"
            fill="transparent" />
          <Area type="monotone" dataKey="predicted_demand" stroke="#6366f1"
            strokeWidth={2.5} fill="url(#forecastGradient)" dot={false} />
          <ReferenceLine x={peakHour.hour} stroke="#ef4444" strokeDasharray="5 5"
            label={{ value: "Peak", position: "top", fill: "#ef4444", fontSize: 11 }} />
        </AreaChart>
      </ResponsiveContainer>
      <div className="mt-4 flex gap-6 text-sm text-slate-400">
        <div>
          <span className="text-slate-500">Peak Hour:</span>{" "}
          <span className="text-indigo-400 font-semibold">{peakHour.hour}:00</span>
        </div>
        <div>
          <span className="text-slate-500">Peak Demand:</span>{" "}
          <span className="text-indigo-400 font-semibold">{peakHour.predicted_demand.toFixed(4)}</span>
        </div>
      </div>
    </div>
  );
}
