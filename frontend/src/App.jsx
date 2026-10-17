import { useCallback, useEffect, useMemo, useState } from 'react';
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import FlowMap from './FlowMap.jsx';
import { fetchCountryDetail, fetchForecast, fetchMapData, fetchYears } from './api.js';

const fmt = new Intl.NumberFormat('en-US');
const money = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });

export default function App() {
  const [years, setYears] = useState([]);
  const [year, setYear] = useState(null);
  const [mode, setMode] = useState('origin');
  const [mapData, setMapData] = useState(null);
  const [selectedCountry, setSelectedCountry] = useState('');
  const [countryDetail, setCountryDetail] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [annualCost, setAnnualCost] = useState(350);
  const [shelterCapacity, setShelterCapacity] = useState(20000);
  const [hostOnly, setHostOnly] = useState('');
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [searchMessage, setSearchMessage] = useState('');
  const [authPrompt, setAuthPrompt] = useState('');

  useEffect(() => {
    fetchYears()
      .then((result) => {
        setYears(result.years);
        setYear(result.max_year);
      })
      .catch((reason) => setError(reason.message));
  }, []);

  useEffect(() => {
    if (year === null) return undefined;
    const controller = new AbortController();
    setBusy(true);
    setError('');
    fetchMapData({ year, mode, country: hostOnly || undefined, signal: controller.signal })
      .then((result) => setMapData(result))
      .catch((reason) => { if (reason.name !== 'AbortError') setError(reason.message); })
      .finally(() => { if (!controller.signal.aborted) setBusy(false); });
    return () => controller.abort();
  }, [year, mode, hostOnly]);

  useEffect(() => {
    if (!playing || years.length === 0) return undefined;
    const timer = window.setInterval(() => {
      setYear((current) => {
        const index = years.indexOf(current);
        return years[(index + 1) % years.length];
      });
    }, 1300);
    return () => window.clearInterval(timer);
  }, [playing, years]);

  const chooseCountry = useCallback((name) => {
    setSelectedCountry(name);
  }, []);

  useEffect(() => {
    if (!selectedCountry) {
      setCountryDetail(null);
      setForecast(null);
      return undefined;
    }
    let alive = true;
    const controller = new AbortController();
    fetchCountryDetail({ country: selectedCountry, mode, year, signal: controller.signal })
      .then((result) => { if (alive) setCountryDetail(result); })
      .catch(() => { if (alive) setCountryDetail(null); });
    if (mode === 'asylum') {
      fetchForecast({ country: selectedCountry, startYear: Math.max(2025, year + 1), years: 3, signal: controller.signal })
        .then((result) => { if (alive) setForecast(result); })
        .catch((reason) => { if (alive && reason.name !== 'AbortError') setForecast({ error: reason.message }); });
    } else {
      setForecast(null);
    }
    return () => { alive = false; controller.abort(); };
  }, [selectedCountry, mode, year]);

  const yearIndex = years.indexOf(year);
  const hostOptions = useMemo(() => mapData?.countries.map(({ name }) => name) ?? [], [mapData]);
  const latestAnnual = countryDetail?.annual.at(-1)?.total ?? null;
  const capacityGuess = shelterCapacity > 0 ? Math.ceil((latestAnnual ?? 0) / shelterCapacity) : 0;
  const forecastAnnual = forecast?.predictions?.[0]?.total ?? 0;
  const forecastCost = forecastAnnual * annualCost;
  const forecastShelters = shelterCapacity > 0 ? Math.ceil(forecastAnnual / shelterCapacity) : 0;
  const capacityRecord = forecast?.latest_host_capacity;
  const scenarioGap = capacityRecord ? Math.max(0, forecastAnnual - capacityRecord.total_capacity) : null;

  function searchCountry(event) {
    event.preventDefault();
    const match = hostOptions.find((name) => name.toLowerCase().includes(searchTerm.trim().toLowerCase()));
    if (!match) {
      setSearchMessage(searchTerm.trim() ? 'No country found in the loaded historical data.' : 'Enter a country name to search.');
      return;
    }
    setSearchMessage('');
    setSearchOpen(false);
    setMode('asylum');
    setHostOnly('');
    chooseCountry(match);
    setTimeout(() => document.getElementById('atlas')?.scrollIntoView({ behavior: 'smooth' }), 0);
  }

  return (
    <>
      <header className="masthead">
        <a href="#home" className="wordmark" aria-label="Refugee Flow home">
          <svg className="brand-mark" viewBox="0 0 42 42" aria-hidden="true">
            <path d="M21 3.5 34.5 11v15.5L21 34 7.5 26.5V11L21 3.5Z" />
            <path d="M21 3.5v15.7m13.5-8.2L21 19.2 7.5 11m0 15.5 13.5-7.3 13.5 7.3M21 19.2V34" />
            <circle cx="21" cy="19.2" r="2.4" />
          </svg>
          <span className="brand-name">Refugee<span>Flow</span><small>FORECASTING &amp; RESOURCE ALLOCATION</small></span>
        </a>
        <nav className="site-nav" aria-label="Main navigation">
          <a className="nav-home" href="#home">Home</a>
          <a href="#atlas">Our Data</a>
          <a href="#contact">Contact</a>
        </nav>
        <div className="header-actions">
          <button className="search-toggle" type="button" aria-label={searchOpen ? 'Close search' : 'Search countries'} aria-expanded={searchOpen} onClick={() => { setSearchOpen(!searchOpen); setSearchMessage(''); }}>
            <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.8" cy="10.8" r="6.3"/><path d="m15.4 15.4 4.2 4.2"/></svg>
          </button>
          <button className="auth-button login-button" type="button" onClick={() => setAuthPrompt('Log in')}>Log in</button>
          <button className="auth-button signup-button" type="button" onClick={() => setAuthPrompt('Sign up')}>Sign up</button>
          {searchOpen && <form className="header-search" role="search" onSubmit={searchCountry}>
            <label className="sr-only" htmlFor="country-search">Search historical data by country</label>
            <input id="country-search" autoFocus value={searchTerm} onChange={(event) => { setSearchTerm(event.target.value); setSearchMessage(''); }} placeholder="Search a country" />
            <button type="submit">Search</button>
            {searchMessage && <span role="status">{searchMessage}</span>}
          </form>}
        </div>
      </header>

      {authPrompt && <div className="auth-backdrop" role="presentation" onClick={() => setAuthPrompt('')}>
        <section className="auth-notice" role="dialog" aria-modal="true" aria-labelledby="auth-notice-title" onClick={(event) => event.stopPropagation()}>
          <button className="auth-close" type="button" aria-label="Close" onClick={() => setAuthPrompt('')}>×</button>
          <span className="eyebrow">REFUGEE FLOW</span>
          <h2 id="auth-notice-title">{authPrompt} is not available yet</h2>
          <p>This project does not currently include a sign-in or registration workflow.</p>
          <button className="auth-dismiss" type="button" onClick={() => setAuthPrompt('')}>Got it</button>
        </section>
      </div>}

      <main>
      <section className="intro" id="home" aria-labelledby="hero-title">
        <div className="hero-inner">
          <div className="intro-copy">
            <div className="eyebrow"><i /> HUMANITARIAN INSIGHT <span aria-hidden="true">·</span> DISPLACEMENT FORECASTING</div>
            <h1 id="hero-title">Forecasting refugee flows for stronger, <em>more resilient communities.</em></h1>
            <p>Explore historical refugee movements, uncover emerging displacement patterns, and forecast future flows to help humanitarian teams plan resources where they are needed most.</p>
            <div className="hero-actions">
              <a className="explore-button" href="#atlas">Explore the map <span aria-hidden="true">→</span></a>
            </div>
            <div className="hero-proof" aria-label="Platform capabilities">
              <span>Historical data</span><i aria-hidden="true"/><span>Geographic insight</span><i aria-hidden="true"/><span>Future scenarios</span>
            </div>
          </div>
          <div className="hero-visual">
            <svg className="east-africa-map" viewBox="0 0 500 560" role="img" aria-label="Illustrative outline of East Africa, not an analytical map" preserveAspectRatio="xMidYMid meet">
              <g className="map-land">
                <path d="m198 28 44 8 25 26-6 34 23 31-8 29 18 23-8 32 25 25-9 32-24 19-4 44-21 21-13 42-28 32-4 42-27 20-22-19-13-37-24-11-7-34-21-20 6-32-19-24 9-28-20-24 11-32-14-31 17-24-3-29 28-10 12-26z" />
                <path d="m174 150 43 13 35-8m-89 67 34-6 38 12 43-11m-89 91 35-19 33 11 37-14m-84 73 28-23 29 7 25-14m-59 81 25-23 30-6" />
                <path d="m265 120 49-7 41 19 37-12 34 16-3 31 24 23-12 26-42 3-27 20-38-6-31 13-23-22" />
                <path d="m265 265 31-20 36 9 34-10 36 20-7 39-24 28-8 41-29 25-29-14-23-36-21-16 8-36-19-20" />
                <path d="m245 368 47 3 35 26 25 40-16 52-33 39-29-5-19-37-28-25 11-35-19-29z" />
                <path d="m400 189 33-20 29 12 14 30-14 23-37-5" />
              </g>
              <g className="map-routes">
                <path d="M224 177 Q260 234 306 278 T328 402" />
                <path d="M224 177 Q265 204 304 278 Q277 331 328 402" />
                <path d="M224 177 Q223 285 191 345 Q259 376 328 402" />
              </g>
              <g className="map-points">
                <circle cx="224" cy="177" r="5"/><circle cx="304" cy="278" r="5"/><circle cx="328" cy="402" r="5"/><circle cx="191" cy="345" r="4"/>
              </g>
              <g className="map-labels"><text x="184" y="112">SOUTH SUDAN</text><text x="246" y="244">UGANDA</text><text x="334" y="316">KENYA</text><text x="271" y="467">TANZANIA</text><text x="408" y="229">SOMALIA</text></g>
            </svg>
            <figure className="hero-photo">
              <img src="/assets/dadaab-arrivals-oxfam.jpg" alt="Families who have arrived at Dadaab refugee camp in East Africa" fetchPriority="high" onError={(event) => { event.currentTarget.hidden = true; event.currentTarget.nextElementSibling.hidden = false; }} />
              <div className="photo-fallback" hidden>East Africa · people and place</div>
              <figcaption><span>PEOPLE ON THE MOVE IN EAST AFRICA</span><a href="https://commons.wikimedia.org/wiki/File:Oxfam_East_Africa_-_Hundreds_of_families_are_arriving_in_Dadaab_camp_every_day_01.jpg" target="_blank" rel="noreferrer">Photo: Oxfam East Africa · CC BY 2.0</a></figcaption>
            </figure>
            <span className="visual-note">EAST AFRICA <span>·</span> ILLUSTRATIVE REGION</span>
          </div>
        </div>
      </section>

      <section className="atlas-section" id="atlas">
        <div className="atlas-heading"><div><div className="eyebrow"><i /> REFUGEE POPULATION</div><h2>Movement, over time.</h2></div><span className="source-chip">SOURCE · UNHCR DATA PROVIDED</span></div>
        {error && <div className="error-banner" role="alert">{error} <span>Check that the API is running at localhost:8000.</span></div>}
        <div className="atlas-layout">
          <aside className="story-panel">
            <div className="mode-toggle" role="group" aria-label="Population view">
              <button className={mode === 'origin' ? 'active origin-active' : ''} onClick={() => { setMode('origin'); setHostOnly(''); }}>ORIGIN COUNTRY</button>
              <button className={mode === 'asylum' ? 'active asylum-active' : ''} onClick={() => { setMode('asylum'); setHostOnly(''); }}>ASYLUM COUNTRY</button>
            </div>
            <div className="year-and-region"><strong>{year ?? '—'}</strong><span> · EAST AFRICA</span></div>
            <div className="hero-total">{mapData ? fmt.format(mapData.total_refugees) : '—'}</div>
            <p className="total-label">refugees represented<br />in {mapData?.countries.length ?? '—'} countries</p>
            <div className="mini-chart-wrap"><ResponsiveContainer width="100%" height={104}><AreaChart data={countryDetail?.annual ?? []} margin={{ top: 9, right: 2, bottom: 0, left: 2 }}><defs><linearGradient id="chartFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#fb5838" stopOpacity={0.55}/><stop offset="100%" stopColor="#fb5838" stopOpacity={0.02}/></linearGradient></defs><CartesianGrid stroke="#273139" vertical={false}/><XAxis dataKey="year" tick={{ fill: '#75838d', fontSize: 9 }} axisLine={false} tickLine={false} minTickGap={20}/><YAxis hide/><Tooltip contentStyle={{ background: '#151d22', border: '1px solid #3a474e', borderRadius: 5, color: '#fff' }} formatter={(value) => [fmt.format(value), 'Refugees']}/><Area type="monotone" dataKey="total" stroke="#fb5838" strokeWidth={2} fill="url(#chartFill)" isAnimationActive/></AreaChart></ResponsiveContainer></div>
            <p className="chart-caption">{selectedCountry ? `${selectedCountry} · annual recorded totals` : 'Select a country circle to explore its history'}</p>
            <div className="panel-divider" />
            <label className="filter-label" htmlFor="country-filter">FILTER {mode === 'origin' ? 'ORIGIN' : 'ASYLUM'} COUNTRY</label>
            <select id="country-filter" value={hostOnly} onChange={(event) => setHostOnly(event.target.value)}><option value="">All countries</option>{hostOptions.map((name) => <option key={name}>{name}</option>)}</select>
            {selectedCountry && <div className="country-readout"><span>SELECTED COUNTRY</span><strong>{selectedCountry}</strong><b>{countryDetail ? fmt.format(countryDetail.total_refugees) : '…'}</b><small>{year} recorded refugees</small></div>}
            <p className="data-note">Counts reflect refugee records in the supplied dataset. They are not live movement estimates. Country-centre map markers and connecting lines are illustrative, not individual locations or verified travel paths.</p>
          </aside>

          <div className="map-column">
            <div className="map-topline"><span>{mode === 'origin' ? 'COUNTRIES OF ORIGIN' : 'COUNTRIES OF ASYLUM'}</span><span>{busy ? 'UPDATING…' : mapData ? `${mapData.flows.length} RECORDED COUNTRY PAIRS` : 'CONNECTING TO DATA'}</span></div>
            <FlowMap data={mapData} mode={mode} selectedCountry={selectedCountry} onCountrySelect={chooseCountry} onYearClick={chooseCountry} />
            <div className="timeline">
              <div className="timeline-control"><button className="play-button" aria-label={playing ? 'Pause timeline' : 'Play timeline'} onClick={() => setPlaying(!playing)}>{playing ? 'Ⅱ' : '▶'}</button><button aria-label="Previous year" className="year-step" onClick={() => setYear(years[Math.max(0, yearIndex - 1)])} disabled={yearIndex <= 0}>‹</button><strong>{year ?? '—'}</strong><button aria-label="Next year" className="year-step" onClick={() => setYear(years[Math.min(years.length - 1, yearIndex + 1)])} disabled={yearIndex >= years.length - 1}>›</button></div>
              <div className="range-wrap"><input aria-label="Historical year" type="range" min="0" max={Math.max(years.length - 1, 0)} value={Math.max(yearIndex, 0)} onChange={(event) => { setPlaying(false); setYear(years[Number(event.target.value)]); }} style={{ '--range-progress': `${years.length > 1 ? (yearIndex / (years.length - 1)) * 100 : 0}%` }} /><div className="range-labels"><span>{years[0] ?? '—'}</span><span>YEAR</span><span>{years.at(-1) ?? '—'}</span></div></div>
              <div className="timeline-total"><span>RECORDED TOTAL</span><b>{mapData ? fmt.format(mapData.total_refugees) : '—'}</b></div>
            </div>
          </div>
        </div>
      </section>

      <section className="forecast-section" id="forecast-section">
        <div className="forecast-copy"><div className="eyebrow"><i /> FUTURE SCENARIO · MODEL OUTPUT</div><h2>Planning starts<br />with a <em>possibility.</em></h2><p>Select a country in asylum mode to view the existing model's country-level scenario. Historical observations and model outputs are kept distinct.</p><div className="forecast-caveat">FORECASTS ARE SCENARIOS — NOT OBSERVED COUNTS</div></div>
        <div className="resource-card">
          <div className="resource-card-head"><span>RESOURCE ALLOCATION SCENARIO</span><span className="scenario-tag">CONFIGURABLE</span></div>
          {mode !== 'asylum' || !selectedCountry ? <div className="empty-forecast"><span>↗</span><p>Switch to <b>Asylum Country</b> above and select a country marker to see its model scenario and planning assumptions.</p></div> : forecast?.error ? <div className="empty-forecast error-text">Forecast unavailable: {forecast.error}</div> : <><div className="forecast-title"><span>{selectedCountry} · HOST COUNTRY</span><small>{forecast ? 'MODEL SCENARIO' : 'RUNNING MODEL…'}</small></div><div className="forecast-stats"><div><span>NEXT-YEAR SCENARIO</span><b>{forecast ? fmt.format(forecastAnnual) : '—'}</b><small>refugees · not an observation</small></div><div><span>EST. HOSTING COST</span><b>{forecast ? money.format(forecastCost) : '—'}</b><small>at configured unit cost</small></div><div><span>PLANNING CAPACITY UNITS</span><b>{forecast ? fmt.format(forecastShelters) : '—'}</b><small>at configured unit capacity</small></div></div><div className="assumption-grid"><label>Cost per person / year <span>USD</span><input type="number" min="0" step="50" value={annualCost} onChange={(event) => setAnnualCost(Math.max(0, Number(event.target.value)))}/></label><label>People per capacity unit <span>PEOPLE</span><input type="number" min="1" step="1000" value={shelterCapacity} onChange={(event) => setShelterCapacity(Math.max(1, Number(event.target.value)))}/></label></div>{forecast?.predictions?.length > 0 && <div className="forecast-years">{forecast.predictions.map((item) => <div key={item.year}><span>{item.year} · SCENARIO</span><b>{fmt.format(item.total)}</b></div>)}</div>}{capacityRecord && <div className="capacity-comparison"><div><span>OBSERVED CAPACITY · {capacityRecord.year}</span><b>{fmt.format(capacityRecord.total_capacity)}</b><small>recorded occupancy {fmt.format(capacityRecord.estimated_occupancy)} · {fmt.format(capacityRecord.available_capacity)} available</small></div><div className={scenarioGap > 0 ? 'shortage-warning' : 'capacity-ok'}><span>SCENARIO VS RECORDED CAPACITY</span><b>{scenarioGap > 0 ? `${fmt.format(scenarioGap)} over` : 'Within recorded capacity'}</b><small>{scenarioGap > 0 ? 'illustrative gap, not an operational assessment' : 'compares model output with latest recorded total capacity'}</small></div></div>}<p className="model-note">{forecast?.input_basis ?? 'Uses the existing trained model and supplied conflict/climate inputs.'} Model is not origin-specific and is not independently validated. Capacity figures are historical observations.</p></>}
        </div>
      </section>
      <footer id="contact"><span>REFUGEE FLOW ATLAS · ACADEMIC PROJECT</span><span>HISTORICAL RECORDS ≠ ROUTES · MODEL SCENARIOS ≠ OBSERVATIONS</span></footer>
      </main>
    </>
  );
}
