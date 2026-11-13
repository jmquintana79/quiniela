// Shared helpers for the dashboards (inlined at build time).

/** Escape text before inserting it into HTML. */
function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Format a 0-1 share as a percentage. */
function pct(value, digits = 0) {
  return value == null ? "–" : `${(value * 100).toFixed(digits)}%`;
}

/** Format "YYYY-MM-DD HH:MM" as a short date (and time when requested). */
function fmtDate(value, withTime = false) {
  const d = new Date(value.replace(" ", "T"));
  const date = d.toLocaleDateString("en-GB", { weekday: "short", day: "2-digit", month: "short" });
  return withTime ? `${date} ${value.slice(11, 16)}` : date;
}

/**
 * Stacked 1X2 bar. Segments wide enough show their percentage; every
 * segment shows the exact value in the hover tooltip.
 */
function outcomeBar(home, draw, away, labels = ["Home win", "Draw", "Away win"]) {
  if (home == null) return '<span class="muted">No odds</span>';
  const parts = [["h", home, labels[0]], ["d", draw, labels[1]], ["a", away, labels[2]]];
  return `<div class="bar">${parts.map(([cls, v, label]) =>
    `<div class="${cls}" style="width:${(v * 100).toFixed(2)}%" data-tip="${esc(label)}: ${pct(v, 1)}">` +
    `${v >= 0.12 ? pct(v) : ""}</div>`).join("")}</div>`;
}

/** Legend for the 1X2 bars. */
function outcomeLegend(labels = ["Home win (1)", "Draw (X)", "Away win (2)"]) {
  return `<div class="legend">${[["--home", labels[0]], ["--draw", labels[1]], ["--away", labels[2]]]
    .map(([v, l]) => `<span><i style="background:var(${v})"></i>${esc(l)}</span>`).join("")}</div>`;
}

/** Last results as W/D/L chips (oldest first). */
function formChips(form) {
  const names = { W: "Win", D: "Draw", L: "Loss" };
  return `<span class="form">${[...(form || "")].map((r) =>
    `<b class="${r}" data-tip="${names[r]}">${r}</b>`).join("")}</span>`;
}

/** League table (the form column is optional). */
function standingsTable(rows) {
  const hasForm = rows.length && rows[0].form !== undefined;
  return `<table><thead><tr>
      <th class="num">#</th><th>Team</th><th class="num">P</th><th class="num">W</th>
      <th class="num">D</th><th class="num">L</th><th class="num">GF</th><th class="num">GA</th>
      <th class="num">GD</th><th class="num">Pts</th>${hasForm ? "<th>Form</th>" : ""}
    </tr></thead><tbody>${rows.map((r, i) => `<tr>
      <td class="num muted">${i + 1}</td><td class="strong">${esc(r.team)}</td>
      <td class="num">${r.played}</td><td class="num">${r.won}</td><td class="num">${r.drawn}</td>
      <td class="num">${r.lost}</td><td class="num">${r.gf}</td><td class="num">${r.ga}</td>
      <td class="num">${r.gd > 0 ? "+" : ""}${r.gd}</td><td class="num strong">${r.points}</td>
      ${hasForm ? `<td>${formChips(r.form)}</td>` : ""}
    </tr>`).join("")}</tbody></table>`;
}

/** Match category chip (M, F, O, U) with its meaning on hover. */
function categoryChip(category, legend = {}) {
  const c = category || "U";
  return `<span class="cat cat-${esc(c)}" data-tip="${esc(legend[c] ?? c)}">${esc(c)}</span>`;
}

/** Whether a match was on an official quiniela ticket (and which jornada). */
function quinielaCell(jornada) {
  if (jornada == null) return '<span class="muted">–</span>';
  const j = Math.round(jornada);
  return `<span class="badge q" data-tip="On the official quiniela ticket of jornada ${j}">Q ${j}</span>`;
}

/** Legend of the categories and of the quiniela flag. */
function categoryLegend(legend = {}) {
  const items = ["M", "F", "O", "U"].filter((c) => legend[c])
    .map((c) => `${categoryChip(c, legend)} ${esc(legend[c])}`);
  return `<div class="legend" style="flex-wrap:wrap;gap:6px 16px">${items.map((i) => `<span>${i}</span>`).join("")}` +
    `<span>${quinielaCell(21).replace(/21/g, "N")} on the official quiniela ticket of jornada N</span></div>`;
}

/**
 * An official quiniela ticket: its 15 matches in order with category, result
 * and, when given, the v3 sign (hit or miss) of the La Liga / Segunda ones.
 */
function officialTicketTable(ticket, legend = {}, withV3 = false) {
  if (!ticket) return '<p class="empty">No official ticket.</p>';
  return `<table><thead><tr><th class="num">#</th><th>Cat.</th><th class="num">Home</th><th>Away</th>
      <th class="num">Score</th><th class="num">1X2</th>${withV3 ? '<th class="num">v3</th>' : ""}</tr></thead>
    <tbody>${ticket.matches.map((m) => {
      const sign = { H: "1", D: "X", A: "2" }[m.sign] ?? (m.position === 15 ? "Pleno" : "–");
      let v3 = "";
      if (withV3) {
        v3 = !m.v3 ? '<td class="num muted">–</td>' : `<td class="num"><span class="pick ${m.sign ? (m.v3 === m.sign ? "hit" : "miss") : "pending"}">` +
          `${{ H: "1", D: "X", A: "2" }[m.v3]}${m.sign ? (m.v3 === m.sign ? " ✓" : " ✗") : ""}</span></td>`;
      }
      return `<tr><td class="num muted">${m.position}</td><td>${categoryChip(m.category, legend)}</td>
        <td class="num strong">${esc(m.home)}</td><td class="strong">${esc(m.away)}</td>
        <td class="num">${m.score ?? "–"}</td><td class="num">${sign}</td>${v3}</tr>`;
    }).join("")}</tbody></table>`;
}

/** Played matches as a results table. */
function resultsTable(rows, legend = {}) {
  if (!rows.length) return '<p class="empty">No matches.</p>';
  return `<table><thead><tr>
      <th>Date</th><th>Cat.</th><th>Quiniela</th><th class="num">Home</th><th class="num">Score</th><th>Away</th><th class="num">1X2</th>
    </tr></thead><tbody>${rows.map((m) => `<tr>
      <td class="muted">${fmtDate(m.date)}</td><td>${categoryChip(m.category ?? "M", legend)}</td><td>${quinielaCell(m.quiniela)}</td>
      <td class="num ${m.FTR === "H" ? "strong" : ""}">${esc(m.home_team)}</td>
      <td class="num strong">${m.FTHG} – ${m.FTAG}</td>
      <td class="${m.FTR === "A" ? "strong" : ""}">${esc(m.away_team)}</td>
      <td class="num">${{ H: "1", D: "X", A: "2" }[m.FTR] ?? "–"}</td>
    </tr>`).join("")}</tbody></table>`;
}

/** Segmented control; calls onSelect(index) and marks the pressed button. */
function segmented(container, labels, selected, onSelect) {
  container.innerHTML = labels.map((l, i) =>
    `<button type="button" aria-pressed="${i === selected}" data-i="${i}">${esc(l)}</button>`).join("");
  container.onclick = (e) => {
    const button = e.target.closest("button");
    if (!button) return;
    container.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b === button));
    onSelect(Number(button.dataset.i));
  };
}

// Tooltip for any element with a data-tip attribute
(() => {
  const tip = document.createElement("div");
  tip.id = "tooltip";
  document.body.appendChild(tip);
  document.addEventListener("mousemove", (e) => {
    const target = e.target.closest("[data-tip]");
    if (!target) { tip.style.display = "none"; return; }
    tip.textContent = target.dataset.tip;
    tip.style.display = "block";
    const x = Math.min(e.clientX + 12, window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = `${x}px`;
    tip.style.top = `${e.clientY + 14}px`;
  });
})();

// ---------------------------------------------------------------------------
// Model prediction tables (current matchday and season dashboards)
// ---------------------------------------------------------------------------

/** Result codes as quiniela signs. */
const SIGN = { H: "1", D: "X", A: "2" };

/** Column label of a model: name plus version, or "(reference)" for the market. */
function modelLabel(m) {
  return `${esc(m.name)} <span class="muted">${m.reference ? "(reference)" : esc(m.version)}</span>`;
}

/** Confidence tier of a value ("high", "medium", "low"), from DATA.confidence_tiers. */
function confidenceTier(value, tiers) {
  if (value == null) return null;
  return Object.entries(tiers).find(([, low]) => value >= low)?.[0] ?? null;
}

/** Confidence chip of a match: probability of the predicted sign, coloured by tier. */
function confidenceCell(match, key, tiers) {
  const value = match[`${key}_conf`];
  const tier = confidenceTier(value, tiers);
  if (!tier) return '<span class="muted">–</span>';
  const tip = `Confidence ${pct(value, 1)} (${tier}): probability of the predicted sign · ` +
    `margin over the second sign ${pct(match[`${key}_margin`], 1)}`;
  return `<span class="conf ${tier}" data-tip="${esc(tip)}">${pct(value)}</span>`;
}

/**
 * Prediction cell: the predicted sign only (probability, score and Pleno al 15
 * in the tooltip, to keep columns narrow); green check for a hit, red cross
 * for a miss, outline while pending, dashed outline without a prediction.
 * A pick can cover several signs (doubles "1X" and triples "1X2" of v4).
 */
function pickCell(match, key) {
  const sign = match[`${key}_sign`];
  if (!sign) return '<span class="pick none" data-tip="No prediction available for this match">–</span>';
  const prob = match[`${key}_prob`];
  const score = match[`${key}_score`];
  const extra = score ? ` · score ${score} · Pleno al 15 ${match[`${key}_pleno`]}` : "";
  const label = [...sign].map((s) => SIGN[s]).join("");
  const kind = { 2: "Double", 3: "Triple" }[sign.length];
  const conf = match[`${key}_margin`] != null && !kind ? ` · margin ${pct(match[`${key}_margin`], 1)}` : "";
  const source = match[`${key}_source`] === "public" ? " · from the public's picks (no model for this match)" : "";
  const tip = kind ? `${kind} ${label}: covers ${pct(prob, 1)} of the outcomes`
    : `Predicted ${label} (${pct(prob, 1)})${conf}${extra}${source}`;
  if (!match.FTR) return `<span class="pick pending ${kind ? "multi" : ""}" data-tip="${tip}">${label}</span>`;
  const hit = sign.includes(match.FTR);
  return `<span class="pick ${hit ? "hit" : "miss"} ${kind ? "multi" : ""}" data-tip="${tip} · ${hit ? "hit" : "miss"}">` +
    `${label} ${hit ? "✓" : "✗"}</span>`;
}

/** Note for matchdays where some matches have no prediction (shown as an asterisk). */
function missingNote(summary) {
  if (!summary.missing) return "";
  const n = summary.missing;
  return `Not every prediction is available: ${n} match${n > 1 ? "es" : ""} without prediction` +
    (summary.missing_in_ticket ? ` (${summary.missing_in_ticket} on the ticket, counted as misses).` : ".");
}

/** Quiniela outcome of a model in one matchday, as a small badge. */
function prizeBadge(summary) {
  const ticket = summary.ticket;
  const label = {
    prize: esc(ticket.category),
    no_prize: "No prize",
    pending: "Pending",
    incomplete: "No ticket",
  }[ticket.status];
  const note = missingNote(summary);
  const tip = [ticket.message, note].filter(Boolean).join(" ");
  return `<span class="badge ${ticket.status === "prize" ? "prize" : ""}" data-tip="${esc(tip)}">` +
    `${label}${note ? " *" : ""}</span>`;
}

/** Chance of each prize category before the matchday, as a short line. */
function oddsLine(summary) {
  const odds = summary.odds;
  if (!odds) return "";
  const plan = summary.plan;
  const cost = plan ? `${plan.doubles} double${plan.doubles === 1 ? "" : "s"} + ${plan.triples} ` +
    `triple${plan.triples === 1 ? "" : "s"} (${plan.bets} bets) · ` : "";
  const parts = ["10", "11", "12", "13"].map((k) => `${k}+ hits ${pct(odds[k], 1)}`);
  parts.push(`14 hits ${pct(odds["14"], 2)}`);
  return `<br><span class="muted">${cost}Chance before the matchday: ${parts.join(" · ")}</span>`;
}

/** Euros with two decimals. */
function eur(value) {
  return value == null ? "–" : `${value.toFixed(2)} €`;
}

/** Chance as text, with "1 in N" for small chances. */
function chanceText(p) {
  if (p == null) return "–";
  if (p === 0) return "0%";
  if (p >= 0.01) return pct(p, 1);
  return `${(p * 100).toFixed(p >= 0.001 ? 2 : 3)}% (1 in ${Math.round(1 / p).toLocaleString("en-GB")})`;
}

/**
 * v5 big-win ticket: what can happen with it, as a horizontal bar chart of the
 * chance of each payout range (one series, values always shown), plus the key
 * figures, so the decision to play is the reader's.
 */
function moneyLine(summary) {
  const m = summary.money;
  if (!m) return "";
  const rows = m.outcomes;
  const rowH = 24, labelW = 150, barW = 250, valueW = 150, top = 4;
  const width = labelW + barW + valueW, height = top + rows.length * rowH;
  const bars = rows.map((o, i) => {
    const y = top + i * rowH;
    const w = Math.max(o.chance * barW, o.chance > 0 ? 2 : 0);
    const big = o.low >= m.threshold;
    const tip = `${o.label}: ${chanceText(o.chance)} of the matchdays`;
    return `<g data-tip="${esc(tip)}">
      <rect x="0" y="${y}" width="${width}" height="${rowH}" fill="transparent"/>
      <text x="${labelW - 8}" y="${y + rowH / 2 + 4}" text-anchor="end" font-size="11"
        fill="var(--text-secondary)" font-weight="${big ? 600 : 400}">${esc(o.label)}</text>
      <rect x="${labelW}" y="${y + 5}" width="${w.toFixed(1)}" height="${rowH - 10}" rx="4"
        fill="var(--accent)" opacity="${big ? 1 : 0.75}"/>
      <text x="${labelW + w + 6}" y="${y + rowH / 2 + 4}" font-size="11"
        fill="var(--text-primary)">${chanceText(o.chance)}</text>
    </g>`;
  }).join("");
  const back = m.cost ? m.expected_payout / m.cost : null;
  const jackpot = m.jackpot ? ` · estimated jackpot of the Pleno al 15 ${Math.round(m.jackpot).toLocaleString("en-GB")} €` : "";
  const every = m.matchdays_per_win ? ` (about once every ${Math.round(m.matchdays_per_win).toLocaleString("en-GB")} matchdays)` : "";
  return `<div style="margin-top:8px">
      <span class="muted">What can happen with this ticket (${eur(m.cost)}), chance per matchday:</span>
      <svg viewBox="0 0 ${width} ${height}" width="100%" style="max-width:${width}px;display:block;margin-top:4px"
        role="img" aria-label="Chance of each payout range of the v5 ticket">${bars}</svg>
      <span class="muted">Prize of ${Math.round(m.threshold).toLocaleString("en-GB")} € or more: ${chanceText(m.chance)}${every} ·
        13+ hits ${chanceText(m.chance_13)} · 14 hits ${chanceText(m.chance_14)} ·
        on average ${eur(m.expected_payout)} back per ${eur(m.cost)} spent (${pct(back, 0)})${jackpot} ·
        predictability (average v3 confidence) ${pct(m.predictability, 1)}.</span>
    </div>`;
}

/** Euros spent and won by a ticket in a played matchday. */
function eurosLine(summary) {
  const e = summary.euros;
  if (!e) return "";
  const source = e.source === "draw" ? `real prizes of the official draw, jornada ${e.jornada}`
    : "prizes estimated with the prize model";
  return `<br><span class="muted">Euros: spent ${eur(e.cost)}, won ${eur(e.payout)} ` +
    `(net ${e.payout - e.cost >= 0 ? "+" : ""}${eur(e.payout - e.cost)}; ${source}).</span>`;
}

/** One banner per model with the outcome of its 14-match quiniela ticket. */
function quinielaBanners(md, models) {
  return models.map((m) => {
    const s = md.summary[m.key];
    const t = s.ticket;
    const note = missingNote(s);
    const title = {
      prize: `Quiniela prize: ${t.category}`,
      no_prize: "Quiniela: no prize",
      pending: "Quiniela: matchday in progress",
      incomplete: "Quiniela: no full ticket",
    }[t.status];
    return `<div class="banner ${t.status}"><strong>${modelLabel(m)} · ${title}${note ? " *" : ""}</strong>` +
      `${esc(t.message)}${oddsLine(s)}${moneyLine(s)}${eurosLine(s)}${note ? `<br><span class="muted">* ${esc(note)}</span>` : ""}</div>`;
  }).join("");
}

/**
 * Matches of a matchday with their result, the confidence of the
 * confidence model (v3) and one column per model, plus a footer with each
 * model's hit rate on the played matches. ``sort`` is "ticket" (ticket
 * order) or "confidence" (most predictable first).
 */
function matchdayTable(md, models, leagues, confKey = null, tiers = {}, sort = "ticket", legend = {}) {
  const rows = sort === "confidence" && confKey
    ? [...md.matches].sort((a, b) => (b[`${confKey}_conf`] ?? -1) - (a[`${confKey}_conf`] ?? -1))
    : md.matches;
  const confModel = models.find((m) => m.key === confKey);
  const confHeader = confModel
    ? `<th class="num" data-tip="Probability of the predicted sign of ${esc(confModel.name)} ${esc(confModel.version)}: high ≥ ${pct(tiers.high)}, medium ≥ ${pct(tiers.medium)}, low below">Confidence</th>`
    : "";
  const result = (m) => m.FTR
    ? `<span class="strong">${m.FTHG} – ${m.FTAG}</span> <span class="muted">(${SIGN[m.FTR]})</span>`
    : '<span class="muted">Pending</span>';
  const footer = models.map((m) => {
    const s = md.summary[m.key];
    const played = s.hits + s.misses;
    const details = [`${s.hits}/${played} ✓`];
    if (s.missing) details.push(`${s.missing} –`);
    if (s.pending) details.push(`${s.pending} pending`);
    return `<td class="num"><span class="strong">${played ? pct(s.rate, 1) : "–"}</span>` +
      `<br><span class="muted">${details.join(" · ")}</span></td>`;
  }).join("");
  return `<table><thead><tr>
      <th class="num">Ticket</th><th>Cat.</th><th>Quiniela</th><th>League</th><th>Kick-off</th><th class="num">Home</th><th>Away</th>
      <th>Result</th>${confHeader}${models.map((m) => `<th class="num" data-tip="${esc(m.description)}">${modelLabel(m)}</th>`).join("")}
    </tr></thead><tbody>${rows.map((m) => `<tr>
      <td class="num muted">${m.ticket ?? ""}</td>
      <td>${categoryChip(m.category, legend)}</td><td>${quinielaCell(m.quiniela)}</td>
      <td class="muted">${esc(leagues[m.league] ?? m.league)}${m.round != null ? ` · R${m.round}` : ""}</td>
      <td class="muted">${m.kickoff_known === false || !m.date ? "–" : fmtDate(m.date, true)}</td>
      <td class="num strong">${esc(m.home_team)}</td><td class="strong">${esc(m.away_team)}</td>
      <td>${result(m)}</td>
      ${confModel ? `<td class="num">${confidenceCell(m, confKey, tiers)}</td>` : ""}
      ${models.map((model) => `<td class="num">${pickCell(m, model.key)}</td>`).join("")}
    </tr>`).join("")}</tbody>
    <tfoot><tr><td colspan="${confModel ? 9 : 8}">Hit rate <span class="muted">(played matches)</span></td>${footer}</tr></tfoot></table>`;
}

/** Legend under the prediction tables. */
const MATCHDAY_LEGEND = `Ticket: the 14 matches of the quiniela check: the official ticket when it is
  saved (data/lae/upcoming.json), otherwise an approximation (La Liga first, then Segunda, in kick-off
  order). Cat.: category of the match (M, F, O, U; hover for the meaning). Quiniela: the official
  ticket the match was on (Q and its jornada). Prizes: 14 hits 1st, 13 2nd, 12 3rd, 11 4th, 10 5th category; the
  Pleno al 15 is not simulated. No ticket: the matchday has fewer than 14 matches (e.g.
  international breaks, with only Segunda). Each cell shows the predicted sign (hover it for its
  probability): ✓ hit · ✗ miss · outline: pending · – no prediction. The market reference (bookmakers'
  odds) is not shown here: v3 follows it closely; it is compared in evaluation.html. Confidence: probability of the v3 sign, green high
  (60% or more), amber medium (45-60%), red low (below 45%). Planned ticket (v4): the v3 signs with
  the doubles (1X) and triples (1X2) of a 48-bet budget placed to maximise the chance of a prize;
  the chance of each prize before the matchday is under each ticket banner. Big-win ticket (v5): the
  doubles and triples (at most 50 €) with the best chance of a prize of 10,000 € or more; its banner
  shows what can happen with it (chance of each payout range), so the decision is yours. Euros of played matchdays use the real prizes of
  the official draw when it is saved in data/lae (otherwise an estimate).`;
