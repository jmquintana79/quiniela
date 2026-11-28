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
  // A missing date must not break the table it is in
  if (!value) return "–";
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
      <th class="num">GD</th><th class="num">Pts</th>${hasForm ? '<th data-tip="Results of the team\'s last 5 league matches, oldest first: W win, D draw, L loss">Last 5 league matches</th>' : ""}
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

/** A model's mark on an official ticket match: sign(s), with ✓ / ✗ once played. */
function ticketPick(picks, result, reduced = false) {
  if (!picks) return '<span class="muted">–</span>';
  // Signs always in 1, X, 2 order, whatever order the planner marked them in
  const label = [...picks].sort((a, b) => "HDA".indexOf(a) - "HDA".indexOf(b)).map((p) => ({ H: "1", D: "X", A: "2" }[p])).join("") +
    (reduced ? '<sup class="muted" data-tip="In the reduction (not every combination is played)">R</sup>' : "");
  if (!result) return `<span class="pick pending ${picks.length > 1 ? "multi" : ""}">${label}</span>`;
  const hit = picks.includes(result);
  if (hit && reduced) {
    // Covered by the reduction, but not every combination is played: the best column may miss it
    return `<span class="pick covered ${picks.length > 1 ? "multi" : ""}" data-tip="${COVERED_TIP}">${label} ✓</span>`;
  }
  return `<span class="pick ${hit ? "hit" : "miss"} ${picks.length > 1 ? "multi" : ""}">${label} ${hit ? "✓" : "✗"}</span>`;
}

/** Tooltip of a reduced match whose result was covered (grey, not green). */
const COVERED_TIP = "Covered by the reduction: not every combination is played, so the best column can still miss one of the R matches (the footer gives the hits of the best column)";

/**
 * An official quiniela ticket: its 15 matches in order with category, result
 * and, when ``models`` are given, what each one marked (✓ hit, ✗ miss) and
 * the Pleno al 15 pick; the footer gives each model's hits and ticket price.
 */
function officialTicketTable(ticket, legend = {}, models = null) {
  if (!ticket) return '<p class="empty">No official ticket.</p>';
  const cols = models || [];
  const rows = ticket.matches.map((m) => {
    const sign = { H: "1", D: "X", A: "2" }[m.sign] ?? (m.position === 15 ? "Pleno" : "–");
    const marks = cols.map((model) => {
      if (m.position === 15) {
        // The Pleno al 15 pick (Dixon-Coles' goals categories) is on the v2, v3, v4, v3+v3F·D, v3+v3F·R and v3+v3F·B tickets
        if (!m.pleno_pick || model.key === "head_to_head") return '<td class="num muted">–</td>';
        const pick = `${m.pleno_pick[0]}-${m.pleno_pick[1]}`;
        if (!m.score) return `<td class="num"><span class="pick pending">${pick}</span></td>`;
        const goals = m.score.split("-").map((g) => ["0", "1", "2"][Number(g)] ?? "M");
        const hit = goals[0] === m.pleno_pick[0] && goals[1] === m.pleno_pick[1];
        return `<td class="num"><span class="pick ${hit ? "hit" : "miss"}">${pick} ${hit ? "✓" : "✗"}</span></td>`;
      }
      const reduced = (ticket.plans?.[model.key]?.reduced || []).includes(m.position);
      return `<td class="num">${ticketPick(m.picks?.[model.key], m.sign, reduced)}</td>`;
    }).join("");
    return `<tr><td class="num muted">${m.position}</td><td>${categoryChip(m.category, legend)}</td>
      <td class="num strong">${esc(m.home)}</td><td class="strong">${esc(m.away)}</td>
      <td class="num">${m.score ?? "–"}</td><td class="num">${sign}</td>${marks}</tr>`;
  }).join("");
  const footer = cols.length ? `<tfoot><tr><td colspan="6">Hits <span class="muted">(14 matches) · price</span></td>${cols.map((model) => {
    const first14 = ticket.matches.filter((m) => m.position <= 14 && m.sign);
    const marked = first14.filter((m) => m.picks?.[model.key]);
    const plan = ticket.plans?.[model.key];
    const outcome = ticket.outcomes?.[model.key];
    // A reduced ticket (v3+v3F·R) does not play every combination: hits of its best column
    const hits = outcome?.hits ?? plan?.hits ?? marked.filter((m) => m.picks[model.key].includes(m.sign)).length;
    const bets = plan ? plan.bets : 1;
    // Liga F models only mark the women's matches: hits, no ticket price
    const price = model.women ? "Liga F matches" : `${eur(bets * 0.75)}${plan ? ` · ${plan.bets} bets` : ""}`;
    return `<td class="num"><span class="strong">${marked.length ? `${hits}/${marked.length}` : "–"}</span>` +
      `<br><span class="muted">${price}</span></td>`;
  }).join("")}</tr></tfoot>` : "";
  return `<table><thead><tr><th class="num">#</th><th>Cat.</th><th class="num">Home</th><th>Away</th>
      <th class="num">Score</th><th class="num">1X2</th>${cols.map((m) => `<th class="num">${modelLabel(m)}</th>`).join("")}</tr></thead>
    <tbody>${rows}</tbody>${footer}</table>`;
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
 * A pick can cover several signs (doubles "1X" and triples "1X2" of v3+v3F·D).
 */
function pickCell(match, key) {
  const sign = match[`${key}_sign`];
  if (!sign) return '<span class="pick none" data-tip="No prediction available for this match">–</span>';
  const prob = match[`${key}_prob`];
  const score = match[`${key}_score`];
  const extra = score ? ` · score ${score} · Pleno al 15 ${match[`${key}_pleno`]}` : "";
  const label = [...sign].sort((a, b) => "HDA".indexOf(a) - "HDA".indexOf(b)).map((s) => SIGN[s]).join("");
  const kind = { 2: "Double", 3: "Triple" }[sign.length];
  const conf = match[`${key}_margin`] != null && !kind ? ` · margin ${pct(match[`${key}_margin`], 1)}` : "";
  const source = match[`${key}_source`] === "v3F" ? " · Liga F: v3F, pi-ratings of the Liga F goals and shots on target"
    : match[`${key}_source`] === "v2F" ? " · Liga F: v2F, pi-ratings of the Liga F results"
    : match[`${key}_source`] === "v1F" ? " · Liga F: v1F, the public's picks"
    : match[`${key}_source`] === "v1N" ? " · national teams: v1N, Elo of every international"
    : match[`${key}_source`] === "v1C" ? " · other club competitions: v1C, Elo of every ESPN club match"
    : match[`${key}_source`] === "public" ? " · from the public's picks (no model for this match)"
    : match[`${key}_source`] === "reduced" ? " · in the reduction (not every combination is played)" : "";
  const mark = match[`${key}_source`] === "reduced" ? '<sup class="muted">R</sup>' : "";
  const tip = kind ? `${kind} ${label}: covers ${pct(prob, 1)} of the outcomes`
    : `Predicted ${label} (${pct(prob, 1)})${conf}${extra}${source}`;
  if (!match.FTR) return `<span class="pick pending ${kind ? "multi" : ""}" data-tip="${tip}">${label}${mark}</span>`;
  const hit = sign.includes(match.FTR);
  if (hit && match[`${key}_source`] === "reduced") {
    return `<span class="pick covered ${kind ? "multi" : ""}" data-tip="${tip} · ${COVERED_TIP}">${label}${mark} ✓</span>`;
  }
  return `<span class="pick ${hit ? "hit" : "miss"} ${kind ? "multi" : ""}" data-tip="${tip} · ${hit ? "hit" : "miss"}">` +
    `${label}${mark} ${hit ? "✓" : "✗"}</span>`;
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
  // Price of the ticket: 0.75 EUR per bet, 2^doubles * 3^triples bets
  const r = plan && plan.reduction;
  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
  // A reduced ticket: the reduction's bets times its direct doubles and triples
  const mix = r
    ? `reduction no. ${r.number} (${r.label}, ${r.bets} bets, 13 guaranteed if every covered sign is right` +
      ` · matches marked R)${r.direct_doubles || r.direct_triples ? ` × direct ${[r.direct_doubles ? plural(r.direct_doubles, "double") : "",
        r.direct_triples ? plural(r.direct_triples, "triple") : ""].filter(Boolean).join(" + ")}` : ""}`
    : plan ? `${plural(plan.doubles, "double")} + ${plural(plan.triples, "triple")}` : "";
  const price = plan ? `<br><strong style="display:inline">Price of the ticket: ${eur(plan.bets * 0.75)}</strong>` +
    `<span class="muted"> · ${mix} = ${plan.bets} bets × 0.75 €</span>` : "";
  const parts = ["10", "11", "12", "13"].map((k) => `${k}+ hits ${pct(odds[k], 1)}`);
  parts.push(`14 hits ${pct(odds["14"], 2)}`);
  // The traffic lights are in the recommendations table of current.html, not in the banners
  return `${price}<br><span class="muted">Chance before the matchday: ${parts.join(" · ")}</span>`;
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
 * v3+v3F·B big-win ticket: what can happen with it, as a horizontal bar chart of the
 * chance of each payout range (one series, values always shown), plus the key
 * figures, so the decision to play is the reader's.
 */
function moneyLine(summary) {
  const m = summary.money;
  if (!m) return "";
  const back = m.cost ? m.expected_payout / m.cost : null;
  const jackpot = m.jackpot ? ` · estimated jackpot of the Pleno al 15 ${Math.round(m.jackpot).toLocaleString("en-GB")} €` : "";
  const every = m.matchdays_per_win ? ` (about once every ${Math.round(m.matchdays_per_win).toLocaleString("en-GB")} matchdays)` : "";
  // Where the other winners of the estimate come from: the public's real shares of this
  // ticket, or the market (the past tickets of the traffic lights always use the market)
  const winners = m.winners === "public"
    ? " Winners estimated from the public's real picks of the 14 matches (the traffic lights compare with past tickets estimated from the market)."
    : m.winners === "market" ? " Winners estimated from the market odds." : "";
  // One line instead of the chart of payout ranges: the chance that matters and the money back
  return `<br><span class="muted">Prize of ${Math.round(m.threshold).toLocaleString("en-GB")} € or more: ${chanceText(m.chance)}${every} ·
      on average ${eur(m.expected_payout)} back per ${eur(m.cost)} spent (${pct(back, 0)})${jackpot}.${winners}</span>`;
}

/**
 * One traffic light: how the ticket's planned chance of something (a large
 * prize, many hits) ranks among the past real tickets planned the same way.
 */
function signalLine(s, measure) {
  if (!s) return "";
  const [icon, label, text] = {
    green: ["●", "Recommended to play", `high, above ${s.percentile}% of the ${s.draws} past tickets`],
    amber: ["◐", "Undetermined", `typical, above ${s.percentile}% of the ${s.draws} past tickets`],
    red: ["✕", "Not recommended", `low, below ${100 - s.percentile}% of the ${s.draws} past tickets`],
  }[s.level];
  return `<div class="signal signal-${s.level}"><span class="signal-dot">${icon}</span>` +
    `<span><strong style="display:inline">${label}</strong> · ${esc(measure)}: ${text}.</span></div>`;
}

/** The rule of the traffic lights (small print) and the warning that none makes money (red). */
function signalRule(signals) {
  const shown = signals.filter(([s]) => s);
  if (!shown.length) return "";
  const limits = shown.map(([s, measure]) => `${measure}: red at or below ${chanceText(s.low)}, green at or above ${chanceText(s.high)}`).join("; ");
  return help(`Light: this chance against past official tickets planned the same way (${esc(limits)}). No light makes the quiniela profitable.`, "margin:4px 0 0");
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
  // Liga F models (v1F) only predict the women's matches: no 14-match ticket of their own
  return models.filter((m) => !m.women).map((m) => {
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
  // Hit rates apart for the men's leagues and the Liga F (separate models and evaluation)
  // Only when models are shown: the tickets of current.html count every match together
  const hasWomen = md.matches.some((m) => m.category === "F") && models.some((m) => !m.ticket);
  const groupRate = (model, women) => {
    const rows = md.matches.filter((m) => (m.category === "F") === women);
    const marked = rows.filter((m) => m[`${model.key}_sign`]);
    const played = marked.filter((m) => m.FTR);
    const hits = played.filter((m) => m[`${model.key}_sign`].includes(m.FTR)).length;
    const details = [`${hits}/${played.length} ✓`];
    if (rows.length - marked.length) details.push(`${rows.length - marked.length} –`);
    if (marked.length - played.length) details.push(`${marked.length - played.length} pending`);
    return `<td class="num"><span class="strong">${played.length ? pct(hits / played.length, 1) : "–"}</span>` +
      `<br><span class="muted">${details.join(" · ")}</span></td>`;
  };
  const footer = models.map((m) => {
    if (hasWomen) return groupRate(m, false);
    const s = md.summary[m.key];
    const played = s.hits + s.misses;
    const details = [`${s.hits}/${played} ✓`];
    if (s.missing) details.push(`${s.missing} –`);
    if (s.pending) details.push(`${s.pending} pending`);
    return `<td class="num"><span class="strong">${played ? pct(s.rate, 1) : "–"}</span>` +
      `<br><span class="muted">${details.join(" · ")}</span></td>`;
  }).join("");
  const womenFooter = hasWomen ? `<tr><td colspan="${confModel ? 9 : 8}">Hit rate · Liga F <span class="muted">(F matches)</span></td>${models.map((m) => groupRate(m, true)).join("")}</tr>` : "";
  // Hits of the 14-match ticket
  const ticketFooter = models.map((m) => {
    const t = md.summary[m.key].ticket;
    if (t.status === "incomplete") return '<td class="num muted">–</td>';
    const pending = t.pending ? `<br><span class="muted">${t.pending} pending</span>` : "";
    return `<td class="num"><span class="strong">${t.hits}/${t.size}</span>${pending}</td>`;
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
    <tfoot>${models.every((m) => m.ticket) ? "" : `<tr><td colspan="${confModel ? 9 : 8}">Hit rate${hasWomen ? " · men's" : ""} <span class="muted">(${hasWomen ? "M and O matches" : "played matches"})</span></td>${footer}</tr>${womenFooter}`}
      <tr><td colspan="${confModel ? 9 : 8}">Hits <span class="muted">(of the played matches; 14 on the ticket)</span></td>${ticketFooter}</tr></tfoot></table>`;
}




/**
 * Table of contents after the dashboard header: one link per section marked
 * with data-toc (sections without an id get one).
 */
function tableOfContents() {
  const header = document.querySelector("main > header");
  const sections = [...document.querySelectorAll("[data-toc]")];
  if (!header || !sections.length) return;
  const links = sections.map((section) => {
    // Stable anchors from the section name (e.g. #toc-liga-f-metrics). The "toc-" prefix
    // keeps them apart from the ids the page scripts fill in (e.g. #standings): two equal ids
    // would make those scripts overwrite the whole section
    if (!section.id) {
      const slug = section.dataset.toc.toLowerCase().normalize("NFD").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
      let id = `toc-${slug}`;
      for (let n = 2; document.getElementById(id); n++) id = `toc-${slug}-${n}`;
      section.id = id;
    }
    return `<a href="#${section.id}">${esc(section.dataset.toc)}</a>`;
  });
  const nav = document.createElement("nav");
  nav.className = "toc";
  nav.setAttribute("aria-label", "Contents");
  nav.innerHTML = `<span class="muted">Contents</span>${links.join("")}`;
  header.after(nav);
}

/** Liga F matches of the official tickets with the result and the v1F, v2F and v3F picks (✓ hit, ✗ miss). */
function ligaFMatchesTable(matches) {
  if (!matches?.length) return '<p class="empty">No Liga F matches on the official tickets.</p>';
  return `<table><thead><tr><th>Season</th><th class="num">Jornada</th><th>Date</th><th class="num">Home</th><th>Away</th>
      <th class="num">Score</th><th class="num">1X2</th><th class="num">Public picks v1F</th><th class="num">Pi-ratings v2F</th><th class="num">Pi-ratings + shots v3F</th></tr></thead>
    <tbody>${matches.map((m) => {
      const tip = m.p ? `Public: 1 ${pct(m.p[0], 0)} · X ${pct(m.p[1], 0)} · 2 ${pct(m.p[2], 0)}` : "Betting page not saved: no v1F pick";
      return `<tr><td>${esc(m.season)}</td><td class="num">${m.jornada}</td><td class="muted">${m.date ? fmtDate(m.date) : "–"}</td>
        <td class="num strong">${esc(m.home)}</td><td class="strong">${esc(m.away)}</td>
        <td class="num">${m.score ?? "–"}</td><td class="num">${{ H: "1", D: "X", A: "2" }[m.FTR] ?? "–"}</td>
        <td class="num" data-tip="${esc(tip)}">${ticketPick(m.public_picks, m.FTR)}</td>
        <td class="num" data-tip="${m.pi_ratings_f ? "Pi-ratings of the Liga F results (ESPN), before the match" : "Not a Liga F match in ESPN"}">${ticketPick(m.pi_ratings_f, m.FTR)}</td>
        <td class="num" data-tip="${m.pi_ratings_shots_f ? "Pi-ratings learned from goals and shots on target (ESPN), before the match" : "Not a Liga F match in ESPN"}">${ticketPick(m.pi_ratings_shots_f, m.FTR)}</td></tr>`;
    }).join("")}</tbody></table>`;
}

/** Per-season Liga F data: league matches (ESPN), on the tickets, with public picks, result shares, goals. */
function ligaFDataTable(data) {
  if (!data?.length) return '<p class="empty">No Liga F data yet. Run <code>invoke update</code>.</p>';
  return `<table><thead><tr><th>Season</th><th class="num">Liga F matches (ESPN)</th><th class="num">Played</th>
      <th class="num">On official tickets</th><th class="num">With public picks (v1F)</th><th class="num">1</th><th class="num">X</th>
      <th class="num">2</th><th class="num">Goals per match</th></tr></thead>
    <tbody>${data.map((d) => `<tr><td>${esc(d.season)}</td><td class="num">${d.league_matches ?? 0}</td><td class="num">${d.played ?? 0}</td>
      <td class="num">${d.tickets ?? 0}</td><td class="num">${d.with_picks ?? 0}</td><td class="num">${pct(d.H, 1)}</td>
      <td class="num">${pct(d.D, 1)}</td><td class="num">${pct(d.A, 1)}</td>
      <td class="num">${d.goals == null ? "–" : d.goals.toFixed(2)}</td></tr>`).join("")}</tbody></table>`;
}


/**
 * Recommendations table: one traffic light per ticket and measure, with the chance of this
 * matchday, how it ranks among the past official tickets and a plain explanation.
 */
function recommendationsTable(rows, models) {
  if (!rows?.length) return '<p class="empty">No traffic lights: the past tickets are needed (invoke backtest.run) and a full ticket.</p>';
  const name = (key) => {
    const m = models.find((x) => x.key === key);
    return m ? modelLabel(m) : esc(key);
  };
  const words = {
    green: ["●", "Good matchday", (s) => `easier than in ${Math.min(9, Math.floor(s.percentile / 10))} of every 10 past matchdays`],
    amber: ["◐", "Normal matchday", () => "like most past matchdays"],
    red: ["✕", "Bad matchday", (s) => `harder than in ${Math.min(9, Math.floor((100 - s.percentile) / 10))} of every 10 past matchdays`],
  };
  const body = rows.map((r) => {
    const [icon, label, text] = words[r.signal.level];
    return `<tr><td>${name(r.key)}</td><td>${esc(r.measure)}${r.primary ? "" : ' <span class="muted">(also)</span>'}</td>
      <td class="num strong">${chanceText(r.value)}</td>
      <td class="muted">usually ${chanceText(r.signal.low)} to ${chanceText(r.signal.high)}</td>
      <td><span class="signal signal-${r.signal.level}" style="display:inline-flex;padding:2px 8px;margin:0">
        <span class="signal-dot">${icon}</span><strong style="display:inline">${label}</strong></span></td>
      <td>${esc(text(r.signal))}</td></tr>`;
  }).join("");
  // The two lights of the ·B ticket often disagree: say what kind of matchday it is
  const big = rows.filter((r) => r.key === "v3b");
  const prize = big.find((r) => r.primary)?.signal.level;
  const hits = big.find((r) => !r.primary)?.signal.level;
  let summary = "";
  if (prize && hits && prize !== hits) {
    summary = hits === "green" || prize === "red"
      ? "Matchday of favourites: many hits are likely, but prizes will be small (many other winners)."
      : "Open matchday: many hits are unlikely, but a prize, when it comes, can be larger (few other winners).";
  }
  // Only the table scrolls sideways: the notes below stay clear of its scroll bar
  return `<div class="scroll" style="padding-bottom:12px"><table><thead><tr><th>Ticket</th><th>Chance of</th>
      <th class="num">This matchday</th><th>Typical range</th><th>Light</th><th>In plain words</th></tr></thead>
      <tbody>${body}</tbody></table></div>
    ${summary ? `<p style="margin:12px 0 0"><strong>${esc(summary)}</strong></p>` : ""}
    <p class="help" style="margin:8px 0 8px">Typical range: what half of the past official tickets (since 2015-16) had,
      planned the same way. Green: above it (top 25%); red: below it (bottom 25%). <b>No light makes the quiniela profitable.</b></p>`;
}

/**
 * Glossary of every model, ticket, metric and abbreviation of the dashboards: one short,
 * plain definition each. Every page lists the terms it shows (renderGlossary).
 */
const GLOSSARY = {
  // Models: men's La Liga and Segunda
  v1: ["Models", "v1 · Head-to-head", "Past results between the same two teams at the same ground. A baseline."],
  v2: ["Models", "v2 · Dixon-Coles", "Goals model from each team's attack and defence, recent matches weighing more."],
  v3: ["Models", "v3 · DC + market", "The main model: Dixon-Coles mixed with the bookmakers' odds. Without odds yet, Dixon-Coles mixed with v4."],
  v4: ["Models", "v4 · Pi-ratings", "Home and away ratings of every team, learned from goal differences."],
  market: ["Models", "Market", "Probabilities implied by the bookmakers' odds (margin removed). The reference to beat."],
  // Other competitions
  v1F: ["Models", "v1F · Public picks (Liga F)", "Share of bettors marking 1, X and 2 in a Liga F match."],
  v2F: ["Models", "v2F · Pi-ratings (Liga F)", "v4 applied to the Liga F results."],
  v3F: ["Models", "v3F · Pi-ratings + shots (Liga F)", "v2F learning from shots on target too. The tickets' Liga F model."],
  v1N: ["Models", "v1N · National-team Elo", "Elo rating of every national team since 1872. The tickets' model for national teams."],
  v1C: ["Models", "v1C · Club Elo", "One Elo rating of every club of 38 leagues and cups (European cups, Copa del Rey, foreign leagues). The tickets' model for those matches."],
  v1P: ["Models", "v1P · Pleno al 15 (Dixon-Coles)", "Most likely goals categories (0, 1, 2, M) of the 15th match from Dixon-Coles."],
  v2P: ["Models", "v2P · Pleno al 15 (market)", "The same from the odds of 1X2 and over/under 2.5 goals. Used when the odds exist."],
  shares: ["Models", "Result shares", "Always the home win, with the shares of 1, X and 2 of earlier matches. The simplest reference."],
  public: ["Models", "Public", "What quiniela bettors mark: their share of 1, X and 2 on the betting page."],
  // Tickets: ways of playing the probabilities
  single: ["Tickets", "Single ticket", "One sign per match, the most likely. 0.75 € (a reference: the minimum ticket is 2 bets)."],
  D2: ["Tickets", "·D2 · Small ticket", "Like ·D with only 2 bets (1.50 €): one double where it helps most."],
  D: ["Tickets", "·D · Direct ticket", "Doubles (1X) and triples (1X2) within 48 bets (36 €), placed for the best chance of 11+ hits."],
  R: ["Tickets", "·R · Reduced ticket", "Like ·D, but may use an official reduction: more matches covered for the same money, not every combination played."],
  B: ["Tickets", "·B · 1,000 € ticket", "Up to 50 € placed for the best chance of a prize of 1,000 € or more (may use a reduction too)."],
  E: ["Tickets", "·E · Expected-prize ticket", "Backtest only: played only when its expected prize beats its cost."],
  "v3+v3F": ["Tickets", "v3+v3F", "The tickets use v3 on La Liga / Segunda, v3F on Liga F, v1N on national teams and v1C on other clubs."],
  // Metrics
  hit_rate: ["Metrics", "Hit rate", "Share of matches whose predicted sign was right. Higher is better."],
  log_loss: ["Metrics", "Log-loss", "Penalty for the probability given to what happened (0 perfect, 1.10 a blind guess). Lower is better; the main metric."],
  brier: ["Metrics", "Brier", "Squared error of the three probabilities. Lower is better."],
  rps: ["Metrics", "RPS", "Like Brier, but a home win predicted as a draw is less wrong than as an away win. Lower is better."],
  contingency: ["Metrics", "Contingency table", "Rows: what happened; columns: what the model said. The diagonal are the hits."],
  calibration: ["Metrics", "Calibration", "Whether things predicted at 60% happen 60% of the time (points on the diagonal)."],
  confidence: ["Metrics", "Confidence", "Probability of the predicted sign: green 60%+, amber 45-60%, red below 45%."],
  entropy: ["Metrics", "Openness (entropy)", "How open a match is: 0.80 clear favourite, 1.10 toss-up."],
  fav_prob: ["Metrics", "Favourite", "Probability of the most likely sign."],
  edge_public: ["Metrics", "Edge on the public", "How much more likely a sign is than the public thinks (above 0: underrated)."],
  value: ["Metrics", "Value", "Probability of a sign over the share of bettors marking it (above 1: worth more than the public thinks)."],
  expected_hits: ["Metrics", "Expected hits", "Hits the single ticket gets on average with these probabilities."],
  return: ["Metrics", "Back per €", "Euros won per euro spent (the quiniela returns 55% on average: below 100% loses money)."],
  net: ["Metrics", "Net", "Euros won minus euros spent."],
  strength: ["Metrics", "Strength gap", "Difference of the two teams' pi-ratings (v4): above 0, the home team is stronger."],
  edge_market: ["Metrics", "Edge on the market", "How far v3 is from the bookmakers on the most likely sign."],
  spread: ["Metrics", "Spread", "How much the bookmakers disagree on the odds. Informative only."],
  differentiation: ["Metrics", "Differentiation", "How far the ticket is from what the public marks (above 0: less marked than likely)."],
  co_winners: ["Metrics", "Co-winners", "Other bettors expected to share the prize if the ticket gets 14."],
  quintile: ["Metrics", "Quintile", "Matches or tickets split into five equal groups, from the lowest to the highest value of a metric."],
  // Marks and abbreviations
  light: ["Marks", "Traffic light", "This jornada's chance against past tickets: green top 25%, amber typical, red bottom 25%. None makes money."],
  verdict: ["Marks", "Verdict", "Big (play ·B), small (single or ·D) or skip, from the lights. A first rule, judged in metrics.html."],
  M: ["Marks", "M", "Men's La Liga or Segunda match."],
  F: ["Marks", "F", "Liga F (women's) match."],
  O: ["Marks", "O", "Other competition: national teams, cups, foreign leagues."],
  U: ["Marks", "U", "Unknown competition."],
  Q: ["Marks", "Q N", "The match was on the official quiniela ticket of jornada N."],
  Rmark: ["Marks", "R", "Match of an official reduction: not every combination of its signs is played."],
  signs: ["Marks", "1 · X · 2", "Home win · draw · away win. 1X, X2, 12 are doubles; 1X2 a triple."],
  marks: ["Marks", "✓ · ✗", "Hit · miss once the match is played (grey ✓: covered by a reduction, the best column may still miss it)."],
  pleno: ["Marks", "Pleno al 15", "Exact goals category of each team (0, 1, 2, M = 3 or more) in the 15th match. Only pays with 14 hits."],
  categories: ["Marks", "Prize categories", "14 hits 1st, 13 2nd, 12 3rd, 11 4th, 10 5th; plus the Pleno al 15 with 14 hits."],
  jornada: ["Marks", "Jornada (J.)", "Number of the official quiniela draw in its season."],
  walk_forward: ["Marks", "Walk-forward", "Every prediction made only with what was known before the match: an honest test."],
};

/** A short, muted explanation paragraph. */
function help(html, style = "") {
  return `<p class="help"${style ? ` style="${style}"` : ""}>${html}</p>`;
}


/** The glossary card of a page: only the given terms, grouped. */
function renderGlossary(keys, target = "glossary") {
  const box = document.getElementById(target);
  if (!box) return;
  const groups = {};
  for (const key of keys) {
    const entry = GLOSSARY[key];
    if (!entry) continue;
    (groups[entry[0]] = groups[entry[0]] || []).push(entry);
  }
  box.innerHTML = Object.entries(groups).map(([group, entries]) =>
    `<h3>${esc(group)}</h3>` + entries.map(([, term, text]) =>
      `<div><span class="term">${esc(term)}</span>${esc(text)}</div>`).join("")).join("");
}
