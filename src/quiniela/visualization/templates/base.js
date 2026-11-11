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

/** Played matches as a results table. */
function resultsTable(rows) {
  if (!rows.length) return '<p class="empty">No matches.</p>';
  return `<table><thead><tr>
      <th>Date</th><th class="num">Home</th><th class="num">Score</th><th>Away</th><th class="num">1X2</th>
    </tr></thead><tbody>${rows.map((m) => `<tr>
      <td class="muted">${fmtDate(m.date)}</td>
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

/**
 * Prediction cell: the predicted sign only (probability, score and Pleno al 15
 * in the tooltip, to keep columns narrow); green check for a hit, red cross
 * for a miss, outline while pending, dashed outline without a prediction.
 */
function pickCell(match, key) {
  const sign = match[`${key}_sign`];
  if (!sign) return '<span class="pick none" data-tip="No prediction available for this match">–</span>';
  const prob = match[`${key}_prob`];
  const score = match[`${key}_score`];
  const extra = score ? ` · score ${score} · Pleno al 15 ${match[`${key}_pleno`]}` : "";
  const tip = `Predicted ${SIGN[sign]} (${pct(prob, 1)})${extra}`;
  const label = SIGN[sign];
  if (!match.FTR) return `<span class="pick pending" data-tip="${tip}">${label}</span>`;
  const hit = sign === match.FTR;
  return `<span class="pick ${hit ? "hit" : "miss"}" data-tip="${tip} · ${hit ? "hit" : "miss"}">` +
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
      `${esc(t.message)}${note ? `<br><span class="muted">* ${esc(note)}</span>` : ""}</div>`;
  }).join("");
}

/**
 * Matches of a matchday with their result and one column per model, plus a
 * footer with each model's hit rate on the played matches.
 */
function matchdayTable(md, models, leagues) {
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
      <th class="num">Ticket</th><th>League</th><th>Kick-off</th><th class="num">Home</th><th>Away</th>
      <th>Result</th>${models.map((m) => `<th class="num" data-tip="${esc(m.description)}">${modelLabel(m)}</th>`).join("")}
    </tr></thead><tbody>${md.matches.map((m) => `<tr>
      <td class="num muted">${m.ticket ?? ""}</td>
      <td class="muted">${esc(leagues[m.league] ?? m.league)} · R${m.round}</td>
      <td class="muted">${fmtDate(m.date, true)}</td>
      <td class="num strong">${esc(m.home_team)}</td><td class="strong">${esc(m.away_team)}</td>
      <td>${result(m)}</td>
      ${models.map((model) => `<td class="num">${pickCell(m, model.key)}</td>`).join("")}
    </tr>`).join("")}</tbody>
    <tfoot><tr><td colspan="6">Hit rate <span class="muted">(played matches)</span></td>${footer}</tr></tfoot></table>`;
}

/** Legend under the prediction tables. */
const MATCHDAY_LEGEND = `Ticket: the 14 matches used for the quiniela check (La Liga first, then
  Segunda, in kick-off order). Prizes: 14 hits 1st, 13 2nd, 12 3rd, 11 4th, 10 5th category; the
  Pleno al 15 is not simulated. No ticket: the matchday has fewer than 14 matches (e.g.
  international breaks, with only Segunda). Each cell shows the predicted sign (hover it for its
  probability): ✓ hit · ✗ miss · outline: pending · – no prediction. The market column is the reference
  (bookmakers' odds), not one of our models.`;
