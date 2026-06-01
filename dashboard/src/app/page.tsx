"use client";

import { useEffect, useState } from "react";
import DemandForecast from "@/components/DemandForecast";
import TrafficHeatmap from "@/components/TrafficHeatmap";
import PeakHourDetection from "@/components/PeakHourDetection";
import WeatherImpact from "@/components/WeatherImpact";
import RoadCapacity from "@/components/RoadCapacity";
import type {
  ForecastPoint, HeatmapPoint, HourlyDemand,
  PeakHours, WeatherImpact as WeatherImpactType, RoadCapacity as RoadCapacityType
} from "@/lib/api";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function DashboardPage() {
  const [forecast, setForecast] = useState<ForecastPoint[]>([]);
  const [heatmap, setHeatmap] = useState<HeatmapPoint[]>([]);
  const [hourly, setHourly] = useState<HourlyDemand[]>([]);
  const [peakHours, setPeakHours] = useState<PeakHours | null>(null);
  const [weather, setWeather] = useState<WeatherImpactType[]>([]);
  const [road, setRoad] = useState<RoadCapacityType | null>(null);
  const [apiStatus, setApiStatus] = useState<"loading" | "connected" | "offline">("loading");
  const [lastUpdate, setLastUpdate] = useState<string>("");

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const healthRes = await fetch(`${API_URL}/api/health`);
        if (!healthRes.ok) throw new Error("API not available");
        setApiStatus("connected");

        const [forecastRes, heatmapRes, hourlyRes, peakRes, weatherRes, roadRes] =
          await Promise.all([
            fetch(`${API_URL}/api/forecast`).then((r) => r.json()).catch(() => []),
            fetch(`${API_URL}/api/analytics/heatmap`).then((r) => r.json()).catch(() => []),
            fetch(`${API_URL}/api/analytics/demand-by-hour`).then((r) => r.json()).catch(() => []),
            fetch(`${API_URL}/api/analytics/peak-hours`).then((r) => r.json()).catch(() => null),
            fetch(`${API_URL}/api/analytics/demand-by-weather`).then((r) => r.json()).catch(() => []),
            fetch(`${API_URL}/api/analytics/road-capacity`).then((r) => r.json()).catch(() => null),
          ]);

        setForecast(forecastRes);
        setHeatmap(heatmapRes);
        setHourly(hourlyRes);
        setPeakHours(peakRes);
        setWeather(weatherRes);
        setRoad(roadRes);
        setLastUpdate(new Date().toLocaleTimeString());
      } catch {
        setApiStatus("offline");
      }
    };

    fetchAll();
    const interval = setInterval(fetchAll, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-[#0a0a1a]">
      {/* Header */}
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-[#0a0a1a]/80 border-b border-indigo-500/10">
        <div className="max-w-[1440px] mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-xl shadow-lg shadow-indigo-500/20">
                🚦
              </div>
              <div>
                <h1 className="text-xl font-bold gradient-text">Traffic Intelligence</h1>
                <p className="text-xs text-slate-500">AI-Powered Demand Prediction Dashboard</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              {lastUpdate && (
                <span className="text-xs text-slate-500">
                  Updated: {lastUpdate}
                </span>
              )}
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium ${
                apiStatus === "connected"
                  ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                  : apiStatus === "offline"
                  ? "bg-red-500/10 text-red-400 border border-red-500/20"
                  : "bg-amber-500/10 text-amber-400 border border-amber-500/20"
              }`}>
                <span className={`w-2 h-2 rounded-full ${
                  apiStatus === "connected" ? "bg-emerald-400 animate-pulse"
                  : apiStatus === "offline" ? "bg-red-400"
                  : "bg-amber-400 animate-pulse"
                }`} />
                {apiStatus === "connected" ? "API Connected" : apiStatus === "offline" ? "API Offline" : "Connecting..."}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-[1440px] mx-auto px-6 py-6">
        {/* Stat Overview */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <StatCard
            label="Total Locations"
            value={heatmap.length.toLocaleString()}
            icon="📍"
            color="indigo"
          />
          <StatCard
            label="Peak Demand"
            value={forecast.length > 0 ? Math.max(...forecast.map(f => f.predicted_demand)).toFixed(4) : "—"}
            icon="📈"
            color="red"
          />
          <StatCard
            label="Weather Types"
            value={weather.length.toString()}
            icon="🌤️"
            color="amber"
          />
          <StatCard
            label="Road Types"
            value={road?.by_road_type?.length?.toString() || "—"}
            icon="🛣️"
            color="emerald"
          />
        </div>

        {apiStatus === "offline" && (
          <div className="glass-card p-8 mb-6 text-center">
            <div className="text-4xl mb-4">🔌</div>
            <h2 className="text-xl font-bold text-slate-300 mb-2">API Server Offline</h2>
            <p className="text-slate-500 max-w-md mx-auto mb-4">
              The FastAPI backend is not running. Start it to see live data:
            </p>
            <code className="inline-block bg-slate-800/50 px-4 py-2 rounded-lg text-indigo-400 text-sm font-mono">
              uvicorn api.main:app --port 8000 --reload
            </code>
          </div>
        )}

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Row 1: Forecast + Peak Hours */}
          <DemandForecast data={forecast} />
          <PeakHourDetection peakData={peakHours} hourlyData={hourly} />

          {/* Row 2: Heatmap + Weather */}
          <TrafficHeatmap data={heatmap} />
          <WeatherImpact data={weather} />

          {/* Row 3: Road Capacity (full width) */}
          <div className="lg:col-span-2">
            <RoadCapacity data={road} />
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-8 text-center text-sm text-slate-600 py-4 border-t border-slate-800/50">
          <p>
            Traffic Intelligence Dashboard • Powered by CatBoost, LightGBM, XGBoost Ensemble
          </p>
        </footer>
      </main>
    </div>
  );
}

function StatCard({
  label,
  value,
  icon,
  color,
}: {
  label: string;
  value: string;
  icon: string;
  color: "indigo" | "red" | "amber" | "emerald";
}) {
  const colorClasses = {
    indigo: "bg-indigo-500/10 border-indigo-500/20 text-indigo-400",
    red: "bg-red-500/10 border-red-500/20 text-red-400",
    amber: "bg-amber-500/10 border-amber-500/20 text-amber-400",
    emerald: "bg-emerald-500/10 border-emerald-500/20 text-emerald-400",
  };

  return (
    <div className={`stat-card glass-card p-4 border ${colorClasses[color]}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-2xl">{icon}</span>
      </div>
      <p className="text-2xl font-bold">{value}</p>
      <p className="text-xs text-slate-500 mt-1">{label}</p>
    </div>
  );
}
