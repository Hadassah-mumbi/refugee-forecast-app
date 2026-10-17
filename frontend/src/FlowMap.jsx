import { useEffect, useMemo, useRef, useState } from 'react';
import { Map as MapLibreMap, NavigationControl, Popup } from 'maplibre-gl';
import { geoCentroid, geoInterpolate } from 'd3-geo';

const WORLD_GEOJSON = 'https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson';

export default function FlowMap({ data, mode, selectedCountry, onCountrySelect, onYearClick }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const popupRef = useRef(null);
  const hoverPopupRef = useRef(null);
  const [geojson, setGeojson] = useState(null);
  const [mapError, setMapError] = useState('');

  useEffect(() => {
    let cancelled = false;
    fetch(WORLD_GEOJSON)
      .then((response) => {
        if (!response.ok) throw new Error('Could not load country boundary data.');
        return response.json();
      })
      .then((json) => { if (!cancelled) setGeojson(json); })
      .catch((error) => { if (!cancelled) setMapError(error.message); });
    return () => { cancelled = true; };
  }, []);

  const mapFeatures = useMemo(() => {
    if (!geojson || !data) return null;
    const totals = new Map(data.countries.map((row) => [row.name, row.total]));
    const geoNames = new Map();
    for (const feature of geojson.features) {
      const props = feature.properties ?? {};
      const rawName = props.ADMIN ?? props.name ?? props.name_en ?? props.NAME ?? '';
      const name = normalizeCountryName(rawName);
      if (name) geoNames.set(name, feature);
    }
    const values = data.countries.map((row) => row.total).filter((value) => value > 0);
    const max = Math.max(...values, 1);
    const centers = new Map();
    for (const [name, countryFeature] of geoNames) {
      const center = geoCentroid(countryFeature);
      if (!Number.isFinite(center[0]) || !Number.isFinite(center[1])) continue;
      centers.set(name, center);
    }
    const features = [];
    for (const [name, total] of totals) {
      const center = centers.get(name);
      if (!center || !total) continue;
      features.push({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: center },
        properties: { name, total, radius: 7 + 22 * Math.sqrt(total / max) },
      });
    }
    const routes = [...(data.flows ?? [])]
      .sort((a, b) => b.total - a.total)
      .slice(0, 35)
      .flatMap((flow) => {
        const origin = centers.get(normalizeCountryName(flow.origin));
        const asylum = centers.get(normalizeCountryName(flow.asylum));
        if (!origin || !asylum || flow.origin === flow.asylum) return [];
        const interpolate = geoInterpolate(origin, asylum);
        const coordinates = Array.from({ length: 49 }, (_, index) => interpolate(index / 48));
        return [{
          type: 'Feature',
          geometry: { type: 'LineString', coordinates },
          properties: { origin: flow.origin, asylum: flow.asylum, total: flow.total },
        }];
      });
    return {
      points: { type: 'FeatureCollection', features },
      routes: { type: 'FeatureCollection', features: routes },
      boundaries: geojson,
    };
  }, [geojson, data]);

  useEffect(() => {
    popupRef.current?.remove();
    hoverPopupRef.current?.remove();
  }, [mapFeatures]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return undefined;
    const map = new MapLibreMap({
      container: containerRef.current,
      style: {
        version: 8,
        glyphs: 'https://fonts.openmaptiles.org/{fontstack}/{range}.pbf',
        sources: {
        },
        layers: [{ id: 'map-background', type: 'background', paint: { 'background-color': '#18242b' } }],
      },
      center: [37, 3],
      zoom: 3.25,
      minZoom: 1.5,
      maxZoom: 8,
      maxBounds: [[-30, -40], [95, 48]],
      attributionControl: {
        compact: true,
        customAttribution: 'Country boundaries: Natural Earth via datasets/geo-countries',
      },
    });
    map.addControl(new NavigationControl({ showCompass: false }), 'bottom-right');
    mapRef.current = map;
    return () => {
      popupRef.current?.remove();
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapFeatures) return undefined;
    const update = () => {
      if (!map.getSource('country-boundaries')) {
        map.addSource('country-boundaries', { type: 'geojson', data: mapFeatures.boundaries });
        map.addLayer({
          id: 'country-fills',
          type: 'fill',
          source: 'country-boundaries',
          paint: { 'fill-color': '#25343c', 'fill-opacity': 0.88 },
        });
        map.addLayer({
          id: 'country-outlines',
          type: 'line',
          source: 'country-boundaries',
          paint: { 'line-color': '#879398', 'line-opacity': 0.26, 'line-width': 0.7 },
        });
      } else {
        map.getSource('country-boundaries').setData(mapFeatures.boundaries);
      }
      if (!map.getSource('refugee-routes')) {
        map.addSource('refugee-routes', { type: 'geojson', data: mapFeatures.routes });
        map.addLayer({
          id: 'refugee-route-lines',
          type: 'line',
          source: 'refugee-routes',
          paint: {
            'line-color': mode === 'origin' ? '#ff6546' : '#04c5dd',
            'line-width': ['interpolate', ['linear'], ['get', 'total'], 1, 0.7, 100000, 2.4],
            'line-opacity': 0.42,
          },
        });
      } else {
        map.getSource('refugee-routes').setData(mapFeatures.routes);
        map.setPaintProperty('refugee-route-lines', 'line-color', mode === 'origin' ? '#ff6546' : '#04c5dd');
      }
      if (!map.getSource('route-particles')) {
        map.addSource('route-particles', { type: 'geojson', data: buildParticleCollection(mapFeatures.routes.features, 0) });
        map.addLayer({
          id: 'route-particle-dots',
          type: 'circle',
          source: 'route-particles',
          paint: {
            'circle-radius': 3.2,
            'circle-color': mode === 'origin' ? '#ffd0a8' : '#b5f8ff',
            'circle-opacity': 0.95,
            'circle-blur': 0.25,
            'circle-stroke-color': mode === 'origin' ? '#ff6546' : '#04c5dd',
            'circle-stroke-width': 1.2,
          },
        });
      } else {
        map.getSource('route-particles').setData(buildParticleCollection(mapFeatures.routes.features, 0));
        map.setPaintProperty('route-particle-dots', 'circle-color', mode === 'origin' ? '#ffd0a8' : '#b5f8ff');
        map.setPaintProperty('route-particle-dots', 'circle-stroke-color', mode === 'origin' ? '#ff6546' : '#04c5dd');
      }
      if (!map.getSource('refugee-totals')) {
        map.addSource('refugee-totals', { type: 'geojson', data: mapFeatures.points });
        map.addLayer({
          id: 'refugee-circles',
          type: 'circle',
          source: 'refugee-totals',
          paint: {
            'circle-radius': ['get', 'radius'],
            'circle-color': mode === 'origin' ? '#ff5635' : '#04c5dd',
            'circle-opacity': 0.8,
            'circle-stroke-color': ['case', ['==', ['get', 'name'], selectedCountry], '#ffffff', '#f6e7dc'],
            'circle-stroke-width': ['case', ['==', ['get', 'name'], selectedCountry], 3, 1.2],
          },
        });
        map.addLayer({
          id: 'refugee-labels',
          type: 'symbol',
          source: 'refugee-totals',
          minzoom: 3.3,
          layout: {
            'text-field': ['get', 'name'],
            'text-size': 11,
            'text-offset': [0, 1.8],
            'text-font': ['Open Sans Regular'],
          },
          paint: { 'text-color': '#d8e5e8', 'text-halo-color': '#152128', 'text-halo-width': 1.5 },
        });
        map.on('click', 'refugee-circles', (event) => {
          const feature = event.features?.[0];
          if (!feature) return;
          const country = feature.properties.name;
          onCountrySelect(country);
          hoverPopupRef.current?.remove();
          popupRef.current?.remove();
          popupRef.current = new Popup({ closeButton: true, offset: 14 })
            .setLngLat(event.lngLat)
            .setHTML(`<strong>${escapeHtml(country)}</strong><br/>${Number(feature.properties.total).toLocaleString()} refugees`)
            .addTo(map);
        });
        map.on('mouseenter', 'refugee-circles', () => { map.getCanvas().style.cursor = 'pointer'; });
        map.on('mousemove', 'refugee-circles', (event) => {
          const feature = event.features?.[0];
          if (!feature) return;
          hoverPopupRef.current?.remove();
          hoverPopupRef.current = new Popup({ closeButton: false, closeOnClick: false, offset: 12 })
            .setLngLat(event.lngLat)
            .setHTML(`<strong>${escapeHtml(feature.properties.name)}</strong><br/>${Number(feature.properties.total).toLocaleString()} refugees`)
            .addTo(map);
        });
        map.on('mouseleave', 'refugee-circles', () => {
          map.getCanvas().style.cursor = '';
          hoverPopupRef.current?.remove();
        });
      } else {
        map.getSource('country-boundaries').setData(mapFeatures.boundaries);
        map.getSource('refugee-routes').setData(mapFeatures.routes);
        map.getSource('route-particles').setData(buildParticleCollection(mapFeatures.routes.features, 0));
        map.getSource('refugee-totals').setData(mapFeatures.points);
        map.setPaintProperty('refugee-circles', 'circle-stroke-color', ['case', ['==', ['get', 'name'], selectedCountry], '#ffffff', '#f6e7dc']);
        map.setPaintProperty('refugee-circles', 'circle-stroke-width', ['case', ['==', ['get', 'name'], selectedCountry], 3, 1.2]);
      }
    };
    if (map.isStyleLoaded()) update();
    else map.once('load', update);
    return () => map.off('load', update);
  }, [mapFeatures, mode, selectedCountry, onCountrySelect, onYearClick]);

  useEffect(() => {
    const map = mapRef.current;
    const routeFeatures = mapFeatures?.routes.features ?? [];
    if (!map || !map.getSource('route-particles') || routeFeatures.length === 0) return undefined;
    let progress = 0;
    const animation = window.setInterval(() => {
      progress = (progress + 0.012) % 1;
      const source = map.getSource('route-particles');
      if (source) source.setData(buildParticleCollection(routeFeatures, progress));
    }, 90);
    return () => window.clearInterval(animation);
  }, [mapFeatures]);

  useEffect(() => {
    if (selectedCountry && mapRef.current) {
      const match = mapFeatures?.points.features.find((feature) => feature.properties.name === selectedCountry);
      if (match) mapRef.current.flyTo({ center: match.geometry.coordinates, zoom: Math.max(mapRef.current.getZoom(), 4.1), speed: 0.9 });
    }
  }, [selectedCountry, mapFeatures]);

  return (
    <div className="map-frame">
      <div ref={containerRef} className="map-canvas" aria-label="Interactive East Africa refugee flow map" />
      {!geojson && !mapError && <div className="map-status">Loading world boundaries…</div>}
      {mapError && <div className="map-status map-status-error">{mapError} Circle markers still show totals when boundary data is available.</div>}
      <div className="map-legend"><span className={mode === 'origin' ? 'legend-dot origin' : 'legend-dot asylum'} />{mode === 'origin' ? 'Origin population' : 'Hosted refugee population'}<small>Circle size represents reported total</small></div>
      <div className="map-vignette" />
    </div>
  );
}

function normalizeCountryName(value) {
  const name = String(value).trim();
  if (name === 'United Rep. of Tanzania' || name === 'United Republic of Tanzania') return 'Tanzania';
  if (name === 'Dem. Rep. Congo' || name === 'Democratic Republic of the Congo') return 'Democratic Republic of the Congo';
  return name;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
}

function buildParticleCollection(routeFeatures, progress) {
  const features = routeFeatures.slice(0, 24).flatMap((route, index) => {
    const coordinates = route.geometry.coordinates;
    if (coordinates.length < 2) return [];
    const offset = index / Math.min(routeFeatures.length, 24);
    const position = (progress + offset) % 1;
    const coordinateIndex = Math.min(coordinates.length - 1, Math.floor(position * coordinates.length));
    return [{
      type: 'Feature',
      geometry: { type: 'Point', coordinates: coordinates[coordinateIndex] },
      properties: { route: index },
    }];
  });
  return { type: 'FeatureCollection', features };
}
