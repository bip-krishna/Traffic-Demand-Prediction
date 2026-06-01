const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface HourlyDemand {
  hour: number;
  mean: number;
  std: number;
  count: number;
}

export interface WeatherImpact {
  Weather: string;
  mean: number;
  median: number;
  std: number;
  count: number;
}

export interface WeekdayDemand {
  weekday: number;
  mean: number;
  std: number;
  name: string;
}

export interface HeatmapPoint {
  geohash: string;
  demand: number;
  latitude: number;
  longitude: number;
}

export interface RoadCapacity {
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

export interface PeakHours {
  peak: number[];
  off_peak: number[];
  peak_avg_demand: number;
  off_peak_avg_demand: number;
}

export interface ForecastPoint {
  hour: number;
  predicted_demand: number;
  confidence_low: number;
  confidence_high: number;
}

async function fetchAPI<T>(endpoint: string): Promise<T | null> {
  try {
    const res = await fetch(`${API_URL}${endpoint}`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export const api = {
  getHealth: () => fetchAPI<{ status: string; models_loaded: string[] }>("/api/health"),
  getHourlyDemand: () => fetchAPI<HourlyDemand[]>("/api/analytics/demand-by-hour"),
  getWeatherImpact: () => fetchAPI<WeatherImpact[]>("/api/analytics/demand-by-weather"),
  getWeekdayDemand: () => fetchAPI<WeekdayDemand[]>("/api/analytics/demand-by-weekday"),
  getHeatmap: () => fetchAPI<HeatmapPoint[]>("/api/analytics/heatmap"),
  getRoadCapacity: () => fetchAPI<RoadCapacity>("/api/analytics/road-capacity"),
  getPeakHours: () => fetchAPI<PeakHours>("/api/analytics/peak-hours"),
  getForecast: () => fetchAPI<ForecastPoint[]>("/api/forecast"),
};
