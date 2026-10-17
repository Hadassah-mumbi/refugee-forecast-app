const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail ?? `Request failed (${response.status})`);
  }
  return payload;
}

export const fetchYears = (options) => request('/api/years', options);

export const fetchMapData = ({ year, mode, country, signal }) => {
  const params = new URLSearchParams({ year: String(year), mode });
  if (country) params.set('country', country);
  return request(`/api/map?${params}`, { signal });
};

export const fetchCountryDetail = ({ country, mode, year, signal }) => {
  const params = new URLSearchParams({ mode });
  if (year !== undefined && year !== null) params.set('year', String(year));
  return request(`/api/country/${encodeURIComponent(country)}?${params}`, { signal });
};

export const fetchForecast = ({ country, startYear, years, signal }) => {
  const params = new URLSearchParams({ country, start_year: String(startYear), years: String(years) });
  return request(`/api/forecast?${params}`, { signal });
};
