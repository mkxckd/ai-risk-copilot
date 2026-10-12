"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const state = { overview: null, selected: null, playing: false, busy: false, ticksSinceStress: 0 };
  const FEATURE_NAMES = {
    top_weight: "Top position weight", hhi: "Concentration (HHI)", var_95_pct: "VaR % of NAV", vol_ann: "Volatility",
    leverage: "Leverage", beta: "Beta", sector_max: "Largest sector weight", drawdown: "Drawdown",
  };
  const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
  const signedPct = (x, d = 1) => (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x * 100).toFixed(d) + "%";
  const money = (x) => {
    const a = Math.abs(x), s = x < 0 ? "−" : "";
    return s + (a >= 1e6 ? (a / 1e6).toFixed(2) + "M" : a >= 1e3 ? (a / 1e3).toFixed(0) + "k" : a.toFixed(0));
  };
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  async function api(path, opts = {}) {
    const res = await fetch(path, { headers: { "content-type": "application/json" }, ...opts });
    if (!res.ok) throw new Error(`${res.status} ${(await res.text()).slice(0, 160)}`);
    return res.json();
  }

  // ---------- status of a book ----------
  function status(book) {
    const s = book.snapshot, a = book.latest_alert;
    if (s.is_anomaly) return { cls: "alert", text: a && a.severity === "critical" ? "Critical" : "Alert" };
    if (a && s.tick - a.tick <= 20) return { cls: "watch", text: "Watch" };
    return { cls: "calm", text: "Calm" };
  }

  // ---------- risk tape (the seismograph strip) ----------
  function mix(a, b, t) {
    const pa = a.match(/\w\w/g).map((h) => parseInt(h, 16)), pb = b.match(/\w\w/g).map((h) => parseInt(h, 16));
    return "rgb(" + pa.map((v, i) => Math.round(v + (pb[i] - v) * t)).join(",") + ")";
  }
  function shade(t) { return t < 0.5 ? mix("c5d3dc", "e3ad4f", t * 2) : mix("e3ad4f", "c23b22", (t - 0.5) * 2); }

  // Bar height: VaR on a scale shared by every book, so rows compare. Bar colour: VaR relative to the book's
  // own median over the window, so a book is only "hot" when it is running above its normal level.
  function strip(history, scaleMax) {
    const W = 600, H = 44, n = history.length || 1, bw = W / n;
    const vals = history.map((r) => r.var_95_pct);
    const sorted = [...vals].sort((a, b) => a - b), median = sorted[Math.floor(sorted.length / 2)] || 1;
    let bars = "", notches = "";
    history.forEach((r, i) => {
      const h = 4 + (r.var_95_pct / scaleMax) * (H - 10), x = i * bw;
      const t = Math.min(Math.max((r.var_95_pct / median - 1.08) / 0.5, 0), 1);
      bars += `<rect x="${x.toFixed(2)}" y="${(H - h).toFixed(2)}" width="${Math.max(bw - 0.6, 0.6).toFixed(2)}" height="${h.toFixed(2)}" fill="${shade(t)}"/>`;
      if (r.is_anomaly) notches += `<rect x="${x.toFixed(2)}" y="0" width="${Math.max(bw, 2).toFixed(2)}" height="5" fill="#c23b22"/>`;
    });
    return `<svg class="strip" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">${bars}${notches}</svg>`;
  }

  function renderBooks() {
    const ol = $("books");
    const scaleMax = Math.max(...state.overview.portfolios.flatMap((b) => b.history.map((r) => r.var_95_pct)), 0.0001);
    ol.innerHTML = state.overview.portfolios.map((b) => {
      const st = status(b), s = b.snapshot;
      return `<li><button class="book" type="button" data-pid="${b.portfolio_id}" aria-current="${b.portfolio_id === state.selected}">
        <span class="id"><span class="name">${esc(b.name)}</span><br><span class="strategy">${esc(b.strategy)}</span></span>
        <span class="var"><span class="figure-big">${pct(s.var_95_pct, 2)}</span><br><span class="figure-cap">1-day VaR of NAV</span></span>
        <span class="lev"><span class="figure-big">${s.leverage.toFixed(2)}x</span><br><span class="figure-cap">leverage</span></span>
        <span class="strip-wrap">${strip(b.history, scaleMax)}</span>
        <span class="status ${st.cls}">${st.text}</span>
      </button></li>`;
    }).join("");
    ol.querySelectorAll(".book").forEach((el) => el.addEventListener("click", () => select(el.dataset.pid)));
  }

  // ---------- detail ----------
  function book() { return state.overview.portfolios.find((b) => b.portfolio_id === state.selected); }

  function renderDetail() {
    const b = book(); if (!b) return;
    const s = b.snapshot;
    $("detail-title").textContent = b.name;
    $("detail-sub").textContent = `${b.strategy}. ${status(b).text === "Calm" ? "No recent alerts." : "Recent alert: see the feed."}`;
    const figs = [
      ["NAV", money(s.nav)], ["1-day VaR 95%", money(s.var_95)], ["Leverage", s.leverage.toFixed(2) + "x"],
      ["Beta", s.beta.toFixed(2)], [`Top position, ${s.top_symbol}`, pct(s.top_weight)],
      [`Largest sector, ${s.sector_max_name}`, pct(s.sector_max)], ["Annual volatility", pct(s.vol_ann)],
    ];
    $("figures").innerHTML = figs.map(([k, v]) => `<div><div class="v">${esc(v)}</div><div class="k">${esc(k)}</div></div>`).join("");
    renderChart(b.history);
  }

  function renderChart(history) {
    const svg = $("var-chart"), W = svg.clientWidth || 800, H = 240, L = 48, R = 12, T = 12, B = 26;
    const vals = history.map((r) => r.var_95_pct);
    const ma = vals.map((_, i) => { const w = vals.slice(Math.max(0, i - 39), i + 1); return w.reduce((a, c) => a + c, 0) / w.length; });
    const lo = Math.min(...vals, ...ma) * 0.95, hi = Math.max(...vals, ...ma) * 1.05;
    const x = (i) => L + (i / Math.max(vals.length - 1, 1)) * (W - L - R);
    const y = (v) => T + (1 - (v - lo) / (hi - lo || 1)) * (H - T - B);
    const path = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
    let grid = "";
    for (let k = 0; k <= 3; k++) {
      const v = lo + ((hi - lo) * k) / 3, yy = y(v);
      grid += `<line x1="${L}" x2="${W - R}" y1="${yy}" y2="${yy}" stroke="#d3dce4"/><text x="${L - 8}" y="${yy + 4}" text-anchor="end" font-size="11" fill="#5a6b7b">${pct(v, 2)}</text>`;
    }
    const hours = (vals.length * 5) / 60;
    grid += `<text x="${L}" y="${H - 6}" font-size="11" fill="#5a6b7b">${hours.toFixed(0)} hours ago</text><text x="${W - R}" y="${H - 6}" text-anchor="end" font-size="11" fill="#5a6b7b">now</text>`;
    const dots = history.map((r, i) => r.is_anomaly ? `<circle cx="${x(i)}" cy="${y(r.var_95_pct)}" r="4" fill="#c23b22"/>` : "").join("");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.innerHTML = `${grid}<path d="${path(ma)}" fill="none" stroke="#9aa8b5" stroke-width="2" stroke-dasharray="5 4"/>
      <path d="${path(vals)}" fill="none" stroke="#17212b" stroke-width="2.2"/>${dots}`;
  }

  function renderAlerts() {
    const alerts = state.overview.alerts;
    if (!alerts.length) return;
    $("alerts").innerHTML = alerts.map((a) => {
      const drivers = (a.drivers || []).slice(0, 2).map((d) => `${FEATURE_NAMES[d.feature] || d.feature} (z ${d.z > 0 ? "+" : ""}${d.z.toFixed(1)})`).join(", ");
      const name = state.overview.portfolios.find((p) => p.portfolio_id === a.portfolio_id)?.name || a.portfolio_id;
      return `<li class="${a.severity}"><span class="who">${esc(name)}</span>, ${a.severity} at tick ${a.tick}<div class="drivers">${esc(drivers)}</div></li>`;
    }).join("");
  }

  function renderClock() {
    const b = state.overview.portfolios[0];
    const ts = new Date(b.snapshot.ts);
    const when = ts.toLocaleString("en-GB", { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
    $("clock").textContent = `Simulated market time ${when}, tick ${state.overview.tick}`;
  }

  // ---------- explanation ----------
  function markdownLite(text) {
    const lines = esc(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").split("\n").filter((l) => l.trim());
    let html = "", inList = false;
    for (const l of lines) {
      if (l.startsWith("- ")) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${l.slice(2)}</li>`; }
      else { if (inList) { html += "</ul>"; inList = false; } html += `<p>${l}</p>`; }
    }
    return html + (inList ? "</ul>" : "");
  }

  async function explain() {
    const box = $("explanation"), btn = $("explain");
    btn.disabled = true; box.hidden = false; box.innerHTML = "<p class='sub'>Gathering evidence from the risk tools…</p>";
    try {
      const out = await api(`/portfolios/${state.selected}/explain`, { method: "POST", body: "{}" });
      const how = out.mode === "llm" ? `Written by ${esc(out.model)} using ${out.evidence.length} tool calls.` : `Rule-based explanation from ${out.evidence.length} tool calls (no LLM key configured).`;
      box.innerHTML = markdownLite(out.explanation) + `<p class="mode">${how}</p>
        <details><summary>Evidence the explanation is built on</summary>${out.evidence.map((e) => `<p><strong>${esc(e.tool)}</strong></p><pre>${esc(JSON.stringify(e.result, null, 1).slice(0, 3000))}</pre>`).join("")}</details>`;
    } catch (e) { box.innerHTML = `<p>Couldn't build the explanation: ${esc(e.message)}</p>`; }
    btn.disabled = false;
  }

  // ---------- stress ----------
  async function loadStress() {
    if (!state.selected) return;
    const data = await api(`/portfolios/${state.selected}/stress`);
    state.ticksSinceStress = 0;
    const max = Math.max(0.02, ...data.scenarios.map((s) => Math.abs(s.pnl_pct_nav)));
    $("scenarios").innerHTML = data.scenarios.map((s) => {
      const w = (Math.abs(s.pnl_pct_nav) / max) * 50, cls = s.pnl < 0 ? "loss" : "gain";
      const worst = s.worst_contributors.filter((c) => c.pnl < 0).map((c) => `${c.symbol} ${signedPct(c.pnl_pct_nav, 2)}`).join(", ");
      return `<li class="scenario"><span>${esc(s.name)}</span>
        <span class="bar-track" role="img" aria-label="${signedPct(s.pnl_pct_nav)} of NAV"><span class="fill ${cls}" style="width:${w.toFixed(1)}%"></span></span>
        <span class="pnl">${signedPct(s.pnl_pct_nav)}</span>
        ${worst ? `<span class="worst">Biggest losses: ${esc(worst)}</span>` : ""}</li>`;
    }).join("");
    const sel = $("sector");
    if (!sel.options.length) sel.innerHTML = data.sectors.map((s) => `<option>${esc(s)}</option>`).join("");
    $("stress-sub").textContent = `Instant P&L of ${book().name}'s current positions under market and sector shocks, as a share of NAV.`;
  }

  async function customStress(ev) {
    ev.preventDefault();
    const market = +$("mkt").value / 100, sector = $("sector").value, sectorMove = +$("sec").value / 100;
    const name = `Market ${signedPct(market, 0)}, ${sector} ${signedPct(sectorMove, 0)}`;
    try {
      const out = await api(`/portfolios/${state.selected}/stress`, { method: "POST",
        body: JSON.stringify({ scenarios: [{ name, market, sectors: { [sector]: sectorMove } }] }) });
      const s = out.scenarios[0];
      $("custom-result").innerHTML = `${esc(name)}: <strong>${signedPct(s.pnl_pct_nav)}</strong> of NAV (${money(s.pnl)})`;
    } catch (e) { $("custom-result").textContent = `Couldn't run the shock: ${e.message}`; }
  }

  // ---------- simulation ----------
  async function refresh() {
    state.overview = await api("/dashboard/overview");
    if (!state.selected) state.selected = state.overview.portfolios[0].portfolio_id;
    renderClock(); renderBooks(); renderDetail(); renderAlerts();
    if (state.pendingEvent && state.overview.tick >= state.pendingEvent) {
      $("event-note").textContent = `Event applied at tick ${state.pendingEvent}.`;
      state.pendingEvent = null;
    }
  }

  async function step(n) {
    if (state.busy) return;
    state.busy = true;
    try {
      await api(`/simulation/step?n=${n}`, { method: "POST" });
      await refresh();
      state.ticksSinceStress += n;
      if (state.ticksSinceStress >= 6) await loadStress();
    } catch (e) { $("clock").textContent = `Lost contact with the risk engine: ${e.message}`; stop(); }
    state.busy = false;
  }

  let timer = null;
  function stop() { clearInterval(timer); timer = null; state.playing = false; $("play").setAttribute("aria-pressed", "false"); $("play").textContent = "Run market"; }
  function togglePlay() {
    if (state.playing) return stop();
    state.playing = true; $("play").setAttribute("aria-pressed", "true"); $("play").textContent = "Pause market";
    timer = setInterval(() => step(1), 1200);
  }

  async function inject(kind) {
    const bodies = {
      concentrate: { kind, portfolio_id: "TECH", symbol: "QBIT", target_weight: 0.45 },
      vol_spike: { kind, mult: 4, duration: 30 },
      sector_crash: { kind, sector: "Energy", drift: -0.02, duration: 10 },
    };
    try {
      const out = await api("/simulation/inject", { method: "POST", body: JSON.stringify(bodies[kind]) });
      state.pendingEvent = out.at_tick;
      $("event-note").textContent = `Scheduled for tick ${out.at_tick}. Run the market or advance time to see the detector respond.`;
      if (kind === "concentrate") select("TECH");
    } catch (e) { $("event-note").textContent = `Couldn't schedule the event: ${e.message}`; }
  }

  function select(pid) {
    state.selected = pid;
    $("explanation").hidden = true;
    renderBooks(); renderDetail(); loadStress();
  }

  // ---------- wiring ----------
  $("play").addEventListener("click", togglePlay);
  $("step1").addEventListener("click", () => step(1));
  $("step12").addEventListener("click", () => step(12));
  $("explain").addEventListener("click", explain);
  document.querySelectorAll(".event").forEach((b) => b.addEventListener("click", () => inject(b.dataset.kind)));
  $("custom-stress").addEventListener("submit", customStress);
  $("mkt").addEventListener("input", () => { $("mkt-out").textContent = signedPct(+$("mkt").value / 100, 0); });
  $("sec").addEventListener("input", () => { $("sec-out").textContent = signedPct(+$("sec").value / 100, 0); });
  window.addEventListener("resize", () => state.overview && renderChart(book().history));

  refresh().then(loadStress).catch((e) => { $("clock").textContent = `Couldn't reach the risk engine: ${e.message}`; });
})();
