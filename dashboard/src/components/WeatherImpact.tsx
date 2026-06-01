"use client";

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar, Legend
} from "recharts";

interface WeatherData {
  Weather: string;
  mean: number;
  median: number;
  std: number;
  count: number;
}

interface Props {
  data: WeatherData[];
}

const WEATHER_ICONS: Record<string, string> = {
  Sunny: "☀️",
  Foggy: "🌫️",
  Rainy: "🌧️",
  Snowy: "❄️",
};

const WEATHER_COLORS: Record<string, string> = {
  Sunny: "#fbbf24",
  Foggy: "#94a3b8",
  Rainy: "#3b82f6",
  Snowy: "#a5f3fc",
};

export default function WeatherImpact({ data }: Props) {
  if (!data || data.length === 0) {
    return (
      <div className="glass-card p-6">
        <h2 className="text-xl font-bold gradient-text mb-4">Weather Impact</h2>
        <div className="h-64 flex items-center justify-center text-slate-500">
          <p>No weather data available. Run the ML pipeline and start the API.</p>
        </div>
      </div>
    );
  }

  const radarData = data.map((d) => ({
    weather: `${WEATHER_ICONS[d.Weather] || ""} ${d.Weather}`,
    demand: d.mean,
    variability: d.std,
  }));

  const maxDemand = Math.max(...data.map((d) => d.mean));
  const minDemand = Math.min(...data.map((d) => d.mean));
  const bestWeather = data.find((d) => d.mean === maxDemand)?.Weather || "";
  const worstWeather = data.find((d) => d.mean === minDemand)?.Weather || "";

  return (
    <div className="glass-card p-6 chart-container">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold gradient-text">Weather Impact Analysis</h2>
      </div>

      {/* Weather stat cards */}
      <div className="grid grid-cols-4 gap-2 mb-5">
        {data.map((d) => (
          <div
            key={d.Weather}
            className="rounded-xl p-3 text-center border transition-all duration-300 hover:scale-105"
            style={{
              borderColor: `${WEATHER_COLORS[d.Weather] || "#6366f1"}40`,
              backgroundColor: `${WEATHER_COLORS[d.Weather] || "#6366f1"}10`,
            }}
          >
            <p className="text-2xl mb-1">{WEATHER_ICONS[d.Weather] || "🌤️"}</p>
            <p className="text-xs text-slate-400">{d.Weather}</p>
            <p className="text-sm font-bold mt-1" style={{ color: WEATHER_COLORS[d.Weather] }}>
              {d.mean.toFixed(4)}
            </p>
            <p className="text-xs text-slate-500">{d.count.toLocaleString()} obs</p>
          </div>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,100,255,0.1)" />
          <XAxis dataKey="Weather" stroke="#94a3b8" tick={{ fontSize: 12 }} />
          <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }}
            tickFormatter={(v) => v.toFixed(3)} />
          <Tooltip
            contentStyle={{
              backgroundColor: "rgba(15,15,40,0.95)",
              border: "1px solid rgba(100,100,255,0.2)",
              borderRadius: "12px",
              color: "#e0e0ff",
            }}
            formatter={(value: number, name: string) => [
              value.toFixed(4),
              name === "mean" ? "Avg Demand" : name === "std" ? "Std Dev" : name,
            ]}
          />
          <Bar dataKey="mean" fill="#6366f1" radius={[6, 6, 0, 0]} name="mean" />
          <Bar dataKey="std" fill="#8b5cf6" radius={[6, 6, 0, 0]} opacity={0.5} name="std" />
        </BarChart>
      </ResponsiveContainer>

      <div className="mt-3 flex justify-between text-sm text-slate-500">
        <span>
          Highest: <span className="text-amber-400">{WEATHER_ICONS[bestWeather]} {bestWeather}</span>
        </span>
        <span>
          Lowest: <span className="text-cyan-400">{WEATHER_ICONS[worstWeather]} {worstWeather}</span>
        </span>
      </div>
    </div>
  );
}
