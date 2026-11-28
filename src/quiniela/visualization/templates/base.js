// Shared helpers for the dashboards (inlined at build time).

/** Escape text before inserting it into HTML. */
function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** A number in Spanish format (decimal comma, thousands point), with the given decimals. */
function num0(value, digits = 0) {
  return Number(value).toLocaleString("es-ES", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

/** Format a 0-1 share as a percentage. */
function pct(value, digits = 0) {
  return value == null ? "–" : `${num0(value * 100, digits)} %`;
}

/** Format "YYYY-MM-DD HH:MM" as a short date (and time when requested). */
function fmtDate(value, withTime = false) {
  // A missing date must not break the table it is in
  if (!value) return "–";
  const d = new Date(value.replace(" ", "T"));
  const date = d.toLocaleDateString("es-ES", { weekday: "short", day: "2-digit", month: "short" });
  return withTime ? `${date} ${value.slice(11, 16)}` : date;
}

/**
 * Stacked 1X2 bar. Segments wide enough show their percentage; every
 * segment shows the exact value in the hover tooltip.
 */
function outcomeBar(home, draw, away, labels = ["Gana el local", "Empate", "Gana el visitante"]) {
  if (home == null) return '<span class="muted">Sin cuotas</span>';
  const parts = [["h", home, labels[0]], ["d", draw, labels[1]], ["a", away, labels[2]]];
  return `<div class="bar">${parts.map(([cls, v, label]) =>
    `<div class="${cls}" style="width:${(v * 100).toFixed(2)}%" data-tip="${esc(label)}: ${pct(v, 1)}">` +
    `${v >= 0.12 ? pct(v) : ""}</div>`).join("")}</div>`;
}

/** Legend for the 1X2 bars. */
function outcomeLegend(labels = ["Gana el local (1)", "Empate (X)", "Gana el visitante (2)"]) {
  return `<div class="legend">${[["--home", labels[0]], ["--draw", labels[1]], ["--away", labels[2]]]
    .map(([v, l]) => `<span><i style="background:var(${v})"></i>${esc(l)}</span>`).join("")}</div>`;
}

/** Last results as W/D/L chips (oldest first). */
function formChips(form) {
  const names = { W: "Victoria", D: "Empate", L: "Derrota" };
  return `<span class="form">${[...(form || "")].map((r) =>
    `<b class="${r}" data-tip="${names[r]}">${r}</b>`).join("")}</span>`;
}

/** League table (the form column is optional). */
function standingsTable(rows) {
  const hasForm = rows.length && rows[0].form !== undefined;
  return `<table><thead><tr>
      <th class="num">#</th><th>Equipo</th><th class="num" data-tip="Jugados">PJ</th><th class="num" data-tip="Ganados">G</th>
      <th class="num" data-tip="Empatados">E</th><th class="num" data-tip="Perdidos">P</th><th class="num" data-tip="Goles a favor">GF</th><th class="num" data-tip="Goles en contra">GC</th>
      <th class="num" data-tip="Diferencia de goles">DG</th><th class="num">Pts</th>${hasForm ? '<th data-tip="Resultados de los 5 últimos partidos de liga, del más antiguo al más reciente: W victoria, D empate, L derrota">Últimos 5 de liga</th>' : ""}
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
  return `<span class="badge q" data-tip="En el boleto oficial de la quiniela de la jornada ${j}">Q ${j}</span>`;
}

/** Legend of the categories and of the quiniela flag. */
function categoryLegend(legend = {}) {
  const items = ["M", "F", "O", "U"].filter((c) => legend[c])
    .map((c) => `${categoryChip(c, legend)} ${esc(legend[c])}`);
  return `<div class="legend" style="flex-wrap:wrap;gap:6px 16px">${items.map((i) => `<span>${i}</span>`).join("")}` +
    `<span>${quinielaCell(21).replace(/21/g, "N")} en el boleto oficial de la quiniela de la jornada N</span></div>`;
}

/** A model's mark on an official ticket match: sign(s), with ✓ / ✗ once played. */
function ticketPick(picks, result, reduced = false) {
  if (!picks) return '<span class="muted">–</span>';
  // Signs always in 1, X, 2 order, whatever order the planner marked them in
  const label = [...picks].sort((a, b) => "HDA".indexOf(a) - "HDA".indexOf(b)).map((p) => ({ H: "1", D: "X", A: "2" }[p])).join("") +
    (reduced ? '<sup class="muted" data-tip="En la reducción (no se juegan todas las combinaciones)">R</sup>' : "");
  if (!result) return `<span class="pick pending ${picks.length > 1 ? "multi" : ""}">${label}</span>`;
  const hit = picks.includes(result);
  if (hit && reduced) {
    // Covered by the reduction, but not every combination is played: the best column may miss it
    return `<span class="pick covered ${picks.length > 1 ? "multi" : ""}" data-tip="${COVERED_TIP}">${label} ✓</span>`;
  }
  return `<span class="pick ${hit ? "hit" : "miss"} ${picks.length > 1 ? "multi" : ""}">${label} ${hit ? "✓" : "✗"}</span>`;
}

/** Tooltip of a reduced match whose result was covered (grey, not green). */
const COVERED_TIP = "Cubierto por la reducción: no se juegan todas las combinaciones, así que la mejor columna aún puede fallar uno de los partidos R (el pie da los aciertos de la mejor columna)";

/**
 * An official quiniela ticket: its 15 matches in order with category, result
 * and, when ``models`` are given, what each one marked (✓ hit, ✗ miss) and
 * the Pleno al 15 pick; the footer gives each model's hits and ticket price.
 */
function officialTicketTable(ticket, legend = {}, models = null) {
  if (!ticket) return '<p class="empty">Sin boleto oficial.</p>';
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
  const footer = cols.length ? `<tfoot><tr><td colspan="6">Aciertos <span class="muted">(14 partidos) · precio</span></td>${cols.map((model) => {
    const first14 = ticket.matches.filter((m) => m.position <= 14 && m.sign);
    const marked = first14.filter((m) => m.picks?.[model.key]);
    const plan = ticket.plans?.[model.key];
    const outcome = ticket.outcomes?.[model.key];
    // A reduced ticket (v3+v3F·R) does not play every combination: hits of its best column
    const hits = outcome?.hits ?? plan?.hits ?? marked.filter((m) => m.picks[model.key].includes(m.sign)).length;
    const bets = plan ? plan.bets : 1;
    // Liga F models only mark the women's matches: hits, no ticket price
    const price = model.women ? "Partidos de Liga F" : `${eur(bets * 0.75)}${plan ? ` · ${plan.bets} apuestas` : ""}`;
    return `<td class="num"><span class="strong">${marked.length ? `${hits}/${marked.length}` : "–"}</span>` +
      `<br><span class="muted">${price}</span></td>`;
  }).join("")}</tr></tfoot>` : "";
  return `<table><thead><tr><th class="num">#</th><th>Cat.</th><th class="num">Local</th><th>Visitante</th>
      <th class="num">Marcador</th><th class="num">1X2</th>${cols.map((m) => `<th class="num">${modelLabel(m)}</th>`).join("")}</tr></thead>
    <tbody>${rows}</tbody>${footer}</table>`;
}

/** Played matches as a results table. */
function resultsTable(rows, legend = {}) {
  if (!rows.length) return '<p class="empty">Sin partidos.</p>';
  return `<table><thead><tr>
      <th>Fecha</th><th>Cat.</th><th>Quiniela</th><th class="num">Local</th><th class="num">Marcador</th><th>Visitante</th><th class="num">1X2</th>
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
  return `${esc(m.name)} <span class="muted">${m.reference ? "(referencia)" : esc(m.version)}</span>`;
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
  const tierName = { high: "alta", medium: "media", low: "baja" }[tier] ?? tier;
  const tip = `Confianza ${pct(value, 1)} (${tierName}): probabilidad del signo pronosticado · ` +
    `margen sobre el segundo signo ${pct(match[`${key}_margin`], 1)}`;
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
  if (!sign) return '<span class="pick none" data-tip="Sin pronóstico para este partido">–</span>';
  const prob = match[`${key}_prob`];
  const score = match[`${key}_score`];
  const extra = score ? ` · marcador ${score} · Pleno al 15 ${match[`${key}_pleno`]}` : "";
  const label = [...sign].sort((a, b) => "HDA".indexOf(a) - "HDA".indexOf(b)).map((s) => SIGN[s]).join("");
  const kind = { 2: "Doble", 3: "Triple" }[sign.length];
  const conf = match[`${key}_margin`] != null && !kind ? ` · margen ${pct(match[`${key}_margin`], 1)}` : "";
  const source = match[`${key}_source`] === "v3F" ? " · Liga F: v3F, pi-ratings de los goles y tiros a puerta de la Liga F"
    : match[`${key}_source`] === "v2F" ? " · Liga F: v2F, pi-ratings de los resultados de la Liga F"
    : match[`${key}_source`] === "v1F" ? " · Liga F: v1F, lo que marca el público"
    : match[`${key}_source`] === "v1N" ? " · selecciones: v1N, Elo de todos los partidos internacionales"
    : match[`${key}_source`] === "v1C" ? " · otras competiciones de clubes: v1C, Elo de todos los partidos de clubes de ESPN"
    : match[`${key}_source`] === "public" ? " · según lo que marca el público (sin modelo para este partido)"
    : match[`${key}_source`] === "reduced" ? " · en la reducción (no se juegan todas las combinaciones)" : "";
  const mark = match[`${key}_source`] === "reduced" ? '<sup class="muted">R</sup>' : "";
  const tip = kind ? `${kind} ${label}: cubre el ${pct(prob, 1)} de los resultados`
    : `Pronóstico ${label} (${pct(prob, 1)})${conf}${extra}${source}`;
  if (!match.FTR) return `<span class="pick pending ${kind ? "multi" : ""}" data-tip="${tip}">${label}${mark}</span>`;
  const hit = sign.includes(match.FTR);
  if (hit && match[`${key}_source`] === "reduced") {
    return `<span class="pick covered ${kind ? "multi" : ""}" data-tip="${tip} · ${COVERED_TIP}">${label}${mark} ✓</span>`;
  }
  return `<span class="pick ${hit ? "hit" : "miss"} ${kind ? "multi" : ""}" data-tip="${tip} · ${hit ? "acierto" : "fallo"}">` +
    `${label}${mark} ${hit ? "✓" : "✗"}</span>`;
}

/** Note for matchdays where some matches have no prediction (shown as an asterisk). */
function missingNote(summary) {
  if (!summary.missing) return "";
  const n = summary.missing;
  return `No están todos los pronósticos: ${n} partido${n > 1 ? "s" : ""} sin pronóstico` +
    (summary.missing_in_ticket ? ` (${summary.missing_in_ticket} del boleto, contados como fallos).` : ".");
}

/** Quiniela outcome of a model in one matchday, as a small badge. */
function prizeBadge(summary) {
  const ticket = summary.ticket;
  const label = {
    prize: esc(ticket.category),
    no_prize: "Sin premio",
    pending: "Pendiente",
    incomplete: "Sin boleto",
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
    ? `reducción n.º ${r.number} (${spanishLabel(r.label)}, ${r.bets} apuestas, 13 garantizados si aciertan todos los signos cubiertos` +
      ` · partidos marcados R)${r.direct_doubles || r.direct_triples ? ` × directos ${[r.direct_doubles ? plural(r.direct_doubles, "doble") : "",
        r.direct_triples ? plural(r.direct_triples, "triple") : ""].filter(Boolean).join(" + ")}` : ""}`
    : plan ? `${plural(plan.doubles, "doble")} + ${plural(plan.triples, "triple")}` : "";
  const price = plan ? `<br><strong style="display:inline">Precio del boleto: ${eur(plan.bets * 0.75)}</strong>` +
    `<span class="muted"> · ${mix} = ${plan.bets} apuestas × 0,75 €</span>` : "";
  const parts = ["10", "11", "12", "13"].map((k) => `${k}+ aciertos ${pct(odds[k], 1)}`);
  parts.push(`14 aciertos ${pct(odds["14"], 2)}`);
  // The traffic lights are in the recommendations table of current.html, not in the banners
  return `${price}<br><span class="muted">Probabilidad antes de la jornada: ${parts.join(" · ")}</span>`;
}

/** An English reduction label ("3 doubles + 3 triples") in Spanish. */
function spanishLabel(label) {
  return String(label ?? "").replace(/doubles?/g, (w) => (w.endsWith("s") ? "dobles" : "doble"));
}

/** Euros with two decimals. */
function eur(value) {
  return value == null ? "–" : `${num0(value, 2)} €`;
}

/** Chance as text, with "1 in N" for small chances. */
function chanceText(p) {
  if (p == null) return "–";
  if (p === 0) return "0%";
  if (p >= 0.01) return pct(p, 1);
  return `${num0(p * 100, p >= 0.001 ? 2 : 3)} % (1 de cada ${Math.round(1 / p).toLocaleString("es-ES")})`;
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
  const jackpot = m.jackpot ? ` · bote estimado del Pleno al 15 ${Math.round(m.jackpot).toLocaleString("es-ES")} €` : "";
  const every = m.matchdays_per_win ? ` (más o menos una vez cada ${Math.round(m.matchdays_per_win).toLocaleString("es-ES")} jornadas)` : "";
  // Where the other winners of the estimate come from: the public's real shares of this
  // ticket, or the market (the past tickets of the traffic lights always use the market)
  const winners = m.winners === "public"
    ? " Acertantes estimados con lo que marca de verdad el público en los 14 partidos (los semáforos comparan con boletos pasados estimados con el mercado)."
    : m.winners === "market" ? " Acertantes estimados con las cuotas del mercado." : "";
  // One line instead of the chart of payout ranges: the chance that matters and the money back
  return `<br><span class="muted">Premio de ${Math.round(m.threshold).toLocaleString("es-ES")} € o más: ${chanceText(m.chance)}${every} ·
      de media vuelven ${eur(m.expected_payout)} por cada ${eur(m.cost)} jugados (${pct(back, 0)})${jackpot}.${winners}</span>`;
}

/**
 * One traffic light: how the ticket's planned chance of something (a large
 * prize, many hits) ranks among the past real tickets planned the same way.
 */
function signalLine(s, measure) {
  if (!s) return "";
  const [icon, label, text] = {
    green: ["●", "Recomendado jugar", `alta, por encima del ${s.percentile} % de los ${s.draws} boletos pasados`],
    amber: ["◐", "Indeterminado", `normal, por encima del ${s.percentile} % de los ${s.draws} boletos pasados`],
    red: ["✕", "No recomendado", `baja, por debajo del ${100 - s.percentile} % de los ${s.draws} boletos pasados`],
  }[s.level];
  return `<div class="signal signal-${s.level}"><span class="signal-dot">${icon}</span>` +
    `<span><strong style="display:inline">${label}</strong> · ${esc(measure)}: ${text}.</span></div>`;
}

/** The rule of the traffic lights (small print) and the warning that none makes money (red). */
function signalRule(signals) {
  const shown = signals.filter(([s]) => s);
  if (!shown.length) return "";
  const limits = shown.map(([s, measure]) => `${measure}: rojo hasta ${chanceText(s.low)}, verde desde ${chanceText(s.high)}`).join("; ");
  return help(`Semáforo: esta probabilidad frente a la de los boletos oficiales pasados planificados igual (${esc(limits)}). Ningún semáforo hace rentable la quiniela.`, "margin:4px 0 0");
}

/** Euros spent and won by a ticket in a played matchday. */
function eurosLine(summary) {
  const e = summary.euros;
  if (!e) return "";
  const source = e.source === "draw" ? `premios reales del sorteo oficial, jornada ${e.jornada}`
    : "premios estimados con el modelo de premios";
  return `<br><span class="muted">Euros: jugados ${eur(e.cost)}, ganados ${eur(e.payout)} ` +
    `(neto ${e.payout - e.cost >= 0 ? "+" : ""}${eur(e.payout - e.cost)}; ${source}).</span>`;
}

/** One banner per model with the outcome of its 14-match quiniela ticket. */
function quinielaBanners(md, models) {
  // Liga F models (v1F) only predict the women's matches: no 14-match ticket of their own
  return models.filter((m) => !m.women).map((m) => {
    const s = md.summary[m.key];
    const t = s.ticket;
    const note = missingNote(s);
    const title = {
      prize: `Premio de quiniela: ${t.category}`,
      no_prize: "Quiniela: sin premio",
      pending: "Quiniela: jornada en juego",
      incomplete: "Quiniela: boleto incompleto",
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
    ? `<th class="num" data-tip="Probabilidad del signo pronosticado por ${esc(confModel.name)} ${esc(confModel.version)}: alta ≥ ${pct(tiers.high)}, media ≥ ${pct(tiers.medium)}, baja por debajo">Confianza</th>`
    : "";
  const result = (m) => m.FTR
    ? `<span class="strong">${m.FTHG} – ${m.FTAG}</span> <span class="muted">(${SIGN[m.FTR]})</span>`
    : '<span class="muted">Pendiente</span>';
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
    if (marked.length - played.length) details.push(`${marked.length - played.length} pendientes`);
    return `<td class="num"><span class="strong">${played.length ? pct(hits / played.length, 1) : "–"}</span>` +
      `<br><span class="muted">${details.join(" · ")}</span></td>`;
  };
  const footer = models.map((m) => {
    if (hasWomen) return groupRate(m, false);
    const s = md.summary[m.key];
    const played = s.hits + s.misses;
    const details = [`${s.hits}/${played} ✓`];
    if (s.missing) details.push(`${s.missing} –`);
    if (s.pending) details.push(`${s.pending} pendientes`);
    return `<td class="num"><span class="strong">${played ? pct(s.rate, 1) : "–"}</span>` +
      `<br><span class="muted">${details.join(" · ")}</span></td>`;
  }).join("");
  const womenFooter = hasWomen ? `<tr><td colspan="${confModel ? 9 : 8}">Aciertos · Liga F <span class="muted">(partidos F)</span></td>${models.map((m) => groupRate(m, true)).join("")}</tr>` : "";
  // Hits of the 14-match ticket
  const ticketFooter = models.map((m) => {
    const t = md.summary[m.key].ticket;
    if (t.status === "incomplete") return '<td class="num muted">–</td>';
    const pending = t.pending ? `<br><span class="muted">${t.pending} pendientes</span>` : "";
    return `<td class="num"><span class="strong">${t.hits}/${t.size}</span>${pending}</td>`;
  }).join("");
  return `<table><thead><tr>
      <th class="num" data-tip="Posición en el boleto">Boleto</th><th>Cat.</th><th>Quiniela</th><th>Liga</th><th>Inicio</th><th class="num">Local</th><th>Visitante</th>
      <th>Resultado</th>${confHeader}${models.map((m) => `<th class="num" data-tip="${esc(m.description)}">${modelLabel(m)}</th>`).join("")}
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
    <tfoot>${models.every((m) => m.ticket) ? "" : `<tr><td colspan="${confModel ? 9 : 8}">Aciertos${hasWomen ? " · masculino" : ""} <span class="muted">(${hasWomen ? "partidos M y O" : "partidos jugados"})</span></td>${footer}</tr>${womenFooter}`}
      <tr><td colspan="${confModel ? 9 : 8}">Aciertos <span class="muted">(de los partidos jugados; 14 en el boleto)</span></td>${ticketFooter}</tr></tfoot></table>`;
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
  nav.setAttribute("aria-label", "Índice");
  nav.innerHTML = `<span class="muted">Índice</span>${links.join("")}`;
  header.after(nav);
}

/** Liga F matches of the official tickets with the result and the v1F, v2F and v3F picks (✓ hit, ✗ miss). */
function ligaFMatchesTable(matches) {
  if (!matches?.length) return '<p class="empty">No hay partidos de Liga F en los boletos oficiales.</p>';
  return `<table><thead><tr><th>Temporada</th><th class="num">Jornada</th><th>Fecha</th><th class="num">Local</th><th>Visitante</th>
      <th class="num">Marcador</th><th class="num">1X2</th><th class="num">Público v1F</th><th class="num">Pi-ratings v2F</th><th class="num">Pi-ratings + tiros v3F</th></tr></thead>
    <tbody>${matches.map((m) => {
      const tip = m.p ? `Público: 1 ${pct(m.p[0], 0)} · X ${pct(m.p[1], 0)} · 2 ${pct(m.p[2], 0)}` : "Página de apuestas no guardada: sin pronóstico v1F";
      return `<tr><td>${esc(m.season)}</td><td class="num">${m.jornada}</td><td class="muted">${m.date ? fmtDate(m.date) : "–"}</td>
        <td class="num strong">${esc(m.home)}</td><td class="strong">${esc(m.away)}</td>
        <td class="num">${m.score ?? "–"}</td><td class="num">${{ H: "1", D: "X", A: "2" }[m.FTR] ?? "–"}</td>
        <td class="num" data-tip="${esc(tip)}">${ticketPick(m.public_picks, m.FTR)}</td>
        <td class="num" data-tip="${m.pi_ratings_f ? "Pi-ratings de los resultados de la Liga F (ESPN), antes del partido" : "No es un partido de Liga F en ESPN"}">${ticketPick(m.pi_ratings_f, m.FTR)}</td>
        <td class="num" data-tip="${m.pi_ratings_shots_f ? "Pi-ratings aprendidos de goles y tiros a puerta (ESPN), antes del partido" : "No es un partido de Liga F en ESPN"}">${ticketPick(m.pi_ratings_shots_f, m.FTR)}</td></tr>`;
    }).join("")}</tbody></table>`;
}

/** Per-season Liga F data: league matches (ESPN), on the tickets, with public picks, result shares, goals. */
function ligaFDataTable(data) {
  if (!data?.length) return '<p class="empty">Aún no hay datos de la Liga F. Ejecuta <code>invoke update</code>.</p>';
  return `<table><thead><tr><th>Temporada</th><th class="num">Partidos de Liga F (ESPN)</th><th class="num">Jugados</th>
      <th class="num">En boletos oficiales</th><th class="num">Con lo que marca el público (v1F)</th><th class="num">1</th><th class="num">X</th>
      <th class="num">2</th><th class="num">Goles por partido</th></tr></thead>
    <tbody>${data.map((d) => `<tr><td>${esc(d.season)}</td><td class="num">${d.league_matches ?? 0}</td><td class="num">${d.played ?? 0}</td>
      <td class="num">${d.tickets ?? 0}</td><td class="num">${d.with_picks ?? 0}</td><td class="num">${pct(d.H, 1)}</td>
      <td class="num">${pct(d.D, 1)}</td><td class="num">${pct(d.A, 1)}</td>
      <td class="num">${d.goals == null ? "–" : num0(d.goals, 2)}</td></tr>`).join("")}</tbody></table>`;
}


/**
 * Recommendations table: one traffic light per ticket and measure, with the chance of this
 * matchday, how it ranks among the past official tickets and a plain explanation.
 */
function recommendationsTable(rows, models) {
  if (!rows?.length) return '<p class="empty">Sin semáforos: hacen falta los boletos pasados (invoke backtest.run) y un boleto completo.</p>';
  const name = (key) => {
    const m = models.find((x) => x.key === key);
    return m ? modelLabel(m) : esc(key);
  };
  const words = {
    green: ["●", "Buena jornada", (s) => `más fácil que en ${Math.min(9, Math.floor(s.percentile / 10))} de cada 10 jornadas pasadas`],
    amber: ["◐", "Jornada normal", () => "como la mayoría de jornadas pasadas"],
    red: ["✕", "Mala jornada", (s) => `más difícil que en ${Math.min(9, Math.floor((100 - s.percentile) / 10))} de cada 10 jornadas pasadas`],
  };
  const body = rows.map((r) => {
    const [icon, label, text] = words[r.signal.level];
    return `<tr><td>${name(r.key)}</td><td>${esc(r.measure)}${r.primary ? "" : ' <span class="muted">(también)</span>'}</td>
      <td class="num strong">${chanceText(r.value)}</td>
      <td class="muted">normalmente de ${chanceText(r.signal.low)} a ${chanceText(r.signal.high)}</td>
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
      ? "Jornada de favoritos: es probable acertar mucho, pero los premios serán pequeños (muchos otros acertantes)."
      : "Jornada abierta: es difícil acertar mucho, pero un premio, si llega, puede ser mayor (pocos otros acertantes).";
  }
  // Only the table scrolls sideways: the notes below stay clear of its scroll bar
  return `<div class="scroll" style="padding-bottom:12px"><table><thead><tr><th>Boleto</th><th>Probabilidad de</th>
      <th class="num">Esta jornada</th><th>Rango normal</th><th>Semáforo</th><th>En pocas palabras</th></tr></thead>
      <tbody>${body}</tbody></table></div>
    ${summary ? `<p style="margin:12px 0 0"><strong>${esc(summary)}</strong></p>` : ""}
    <p class="help" style="margin:8px 0 8px">Rango normal: lo que tuvo la mitad central de los boletos oficiales pasados (desde
      2015-16), planificados igual. Verde: por encima (el 25 % más alto); rojo: por debajo (el 25 % más bajo). <b>Ningún semáforo
      hace rentable la quiniela.</b></p>`;
}

/**
 * Glossary of every model, ticket, metric and abbreviation of the dashboards: one short,
 * plain definition each. Every page lists the terms it shows (renderGlossary).
 */
const GLOSSARY = {
  // Modelos: Liga y Segunda masculinas
  v1: ["Modelos", "v1 · Enfrentamientos directos", "Resultados anteriores entre los mismos dos equipos en el mismo campo. Una referencia básica."],
  v2: ["Modelos", "v2 · Dixon-Coles", "Modelo de goles con el ataque y la defensa de cada equipo; los partidos recientes pesan más."],
  v3: ["Modelos", "v3 · DC + mercado", "El modelo principal: Dixon-Coles combinado con las cuotas de las casas de apuestas. Sin cuotas todavía, Dixon-Coles combinado con v4."],
  v4: ["Modelos", "v4 · Pi-ratings", "Valoraciones de cada equipo como local y como visitante, aprendidas de las diferencias de goles."],
  market: ["Modelos", "Mercado", "Probabilidades que dan las cuotas de las casas de apuestas (sin su margen). La referencia a batir."],
  // Otras competiciones
  v1F: ["Modelos", "v1F · Público (Liga F)", "Porcentaje de apostantes que marca 1, X y 2 en un partido de Liga F."],
  v2F: ["Modelos", "v2F · Pi-ratings (Liga F)", "v4 aplicado a los resultados de la Liga F."],
  v3F: ["Modelos", "v3F · Pi-ratings + tiros (Liga F)", "v2F aprendiendo también de los tiros a puerta. El modelo de los boletos para la Liga F."],
  v1N: ["Modelos", "v1N · Elo de selecciones", "Valoración Elo de cada selección desde 1872. El modelo de los boletos para las selecciones."],
  v1C: ["Modelos", "v1C · Elo de clubes", "Una valoración Elo de cada club de 38 ligas y copas (copas europeas, Copa del Rey, ligas extranjeras). El modelo de los boletos para esos partidos."],
  v1P: ["Modelos", "v1P · Pleno al 15 (Dixon-Coles)", "Categorías de goles (0, 1, 2, M) más probables del partido 15 según Dixon-Coles."],
  v2P: ["Modelos", "v2P · Pleno al 15 (mercado)", "Lo mismo a partir de las cuotas de 1X2 y de más/menos de 2,5 goles. Se usa cuando hay cuotas."],
  shares: ["Modelos", "Proporciones de resultados", "Siempre gana el local, con las proporciones de 1, X y 2 de los partidos anteriores. La referencia más sencilla."],
  public: ["Modelos", "Público", "Lo que marcan los apostantes de la quiniela: su porcentaje de 1, X y 2 en la página de apuestas."],
  // Boletos: formas de jugar las probabilidades
  single: ["Boletos", "Boleto sencillo", "Un signo por partido, el más probable. 0,75 € (una referencia: el boleto mínimo son 2 apuestas)."],
  D2: ["Boletos", "·D2 · Boleto pequeño", "Como ·D con solo 2 apuestas (1,50 €): un doble donde más ayuda."],
  D: ["Boletos", "·D · Boleto directo", "Dobles (1X) y triples (1X2) hasta 48 apuestas (36 €), colocados para la mayor probabilidad de 11 o más aciertos."],
  R: ["Boletos", "·R · Boleto reducido", "Como ·D, pero puede usar una reducción oficial: más partidos cubiertos por el mismo dinero, sin jugar todas las combinaciones."],
  B: ["Boletos", "·B · Boleto de 1.000 €", "Hasta 50 € colocados para la mayor probabilidad de un premio de 1.000 € o más (también puede usar una reducción)."],
  E: ["Boletos", "·E · Boleto de premio esperado", "Solo en el backtest: se juega solo cuando su premio esperado supera su coste (·E completo cuenta también los 14 aciertos y el Pleno)."],
  "v3+v3F": ["Boletos", "v3+v3F", "Los boletos usan v3 en Liga y Segunda, v3F en la Liga F, v1N en las selecciones y v1C en los demás clubes."],
  random: ["Boletos", "Al azar / a ojo / óptimo", "Formas de colocar los mismos dobles y triples para comparar con los planificadores: al azar, en los partidos menos claros o la mejor colocación."],
  bet: ["Boletos", "Apuesta (columna)", "Una combinación de 14 signos, 0,75 €. Un boleto con dobles y triples juega varias apuestas."],
  reduction: ["Boletos", "Reducción oficial", "Una tabla de LAE que juega solo algunas combinaciones de sus dobles y triples y garantiza 13 (o 12) aciertos si aciertan sus signos cubiertos."],
  // Lo bueno que es un modelo
  hit_rate: ["Métricas de los modelos", "Aciertos", "Porcentaje de partidos cuyo signo pronosticado fue el correcto. Más es mejor; fácil de leer, pero no tiene en cuenta lo seguro que estaba el modelo."],
  log_loss: ["Métricas de los modelos", "Log-loss", "La penalización −log(probabilidad dada a lo que pasó), de media sobre los partidos. 0 sería perfecto; 1,10 es dar 1/3 a cada signo; un pronóstico muy seguro que falla cuesta mucho. Menos es mejor: la métrica principal."],
  brier: ["Métricas de los modelos", "Brier", "Error al cuadrado entre las tres probabilidades y lo que pasó (1 para el resultado, 0 para los demás). Menos es mejor."],
  rps: ["Métricas de los modelos", "RPS", "Ranked probability score: como Brier, pero pronosticar empate cuando gana el local es menos grave que pronosticar que gana el visitante. Menos es mejor."],
  macro_f1: ["Métricas de los modelos", "Macro-F1", "Equilibrio de aciertos entre 1, X y 2 (cada uno cuenta igual). Un modelo que nunca dice X saca poca nota."],
  x_picks: ["Métricas de los modelos", "X pronosticadas / Empates reales", "Cuántas veces el modelo pronostica empate, frente a cuántos empates hay."],
  hit_by_result: ["Métricas de los modelos", "Aciertos por resultado", "De los partidos que acabaron 1 (o X, o 2), el porcentaje que acertó el modelo."],
  contingency: ["Métricas de los modelos", "Tabla de contingencia", "Filas: lo que pasó; columnas: lo que dijo el modelo. La diagonal son los aciertos."],
  calibration: ["Métricas de los modelos", "Calibración", "Si lo que se pronostica al 60 % pasa el 60 % de las veces (puntos sobre la diagonal)."],
  confidence: ["Métricas de los modelos", "Confianza", "Probabilidad del signo pronosticado: verde desde el 60 %, ámbar del 45 al 60 %, rojo por debajo del 45 %."],
  // Métricas de un partido, conocidas antes de jugarse
  fav_prob: ["Métricas de partido", "Favorito", "Probabilidad del signo más probable. Más alta: un partido más claro."],
  entropy: ["Métricas de partido", "Apertura (entropía)", "Lo repartidas que están las tres probabilidades: unos 0,80 con un favorito claro, 1,10 a cara o cruz. Más alta: más abierto."],
  strength: ["Métricas de partido", "Diferencia de fuerza", "Probabilidad de que gane el local menos la de que gane el visitante según v4: por encima de 0, el local es más fuerte."],
  edge_market: ["Métricas de partido", "v3 frente al mercado", "Cuánto más (por encima de 0) o menos probable ve v3 su signo que las casas de apuestas (logaritmo del cociente)."],
  edge_public: ["Métricas de partido", "v3 frente al público", "Cuánto más probable ve v3 su signo que el porcentaje de apostantes que lo marca (por encima de 0: el público lo infravalora)."],
  value: ["Métricas de partido", "Mejor valor", "De los tres signos, la mayor probabilidad de v3 dividida por el porcentaje de apostantes que lo marca (por encima de 1: vale más de lo que cree el público)."],
  spread: ["Métricas de partido", "Dispersión de cuotas", "Cuánto discrepan las casas de apuestas: la cuota más alta frente a la media. Solo informativa."],
  // Métricas de una jornada, conocidas antes del sorteo
  expected_hits: ["Métricas de jornada", "Aciertos esperados", "Aciertos que logra de media el boleto sencillo con estas probabilidades (la suma de sus 14 probabilidades)."],
  differentiation: ["Métricas de jornada", "Diferenciación", "Suma de las ventajas de v3 frente al público en el boleto sencillo: por encima de 0, sus signos están menos marcados de lo probables que son."],
  co_winners_14: ["Métricas de jornada", "Coacertantes si hay 14", "Apuestas que se espera que compartan el premio si el boleto sencillo hace 14. Menos: un premio mayor."],
  planned_chance: ["Métricas de jornada", "Probabilidad planificada", "Probabilidad de un resultado (12 o más aciertos, un premio de 1.000 €...) calculada antes del sorteo con las probabilidades. ·D 12+, ·D2 12+, sencillo 12+ y ·B 1.000 €+ son probabilidades planificadas."],
  expected_return: ["Métricas de jornada", "Retorno esperado", "Premio que espera ·E dividido por su coste, antes del sorteo (por encima de 1: espera ganar)."],
  jackpot: ["Métricas de jornada", "Bote", "Dinero acumulado de sorteos anteriores en la categoría de 14 + Pleno al 15."],
  typical_range: ["Métricas de jornada", "Rango normal", "Lo que tuvo la mitad central de los boletos oficiales pasados (percentiles 25 a 75) para la misma probabilidad."],
  // KPI: lo que pasó
  hits: ["KPI", "Aciertos", "Signos acertados de la mejor columna del boleto (un doble cuenta si el resultado es uno de sus signos). Con 10 o más hay premio."],
  hit: ["KPI", "Aciertos de v3 (por quintil)", "Porcentaje de partidos de un grupo cuyo signo de v3 fue el correcto."],
  return: ["KPI", "Devuelto por € (retorno)", "Euros ganados por cada euro jugado. Por debajo del 100 % se pierde dinero; la quiniela devuelve el 55 % de media."],
  net: ["KPI", "Neto", "Euros ganados menos euros jugados."],
  hit12: ["KPI", "·D 12+ / ·B 13+", "Porcentaje de boletos cuya mejor columna hizo 12 (o 13) aciertos o más."],
  big: ["KPI", "·B 1.000 €+", "Porcentaje (o número) de boletos ·B que cobraron 1.000 € o más."],
  prizes_won: ["KPI", "Premios", "Boletos con premio (10 aciertos o más)."],
  real_prize_rate: ["KPI", "Tasa real de premios", "Porcentaje de boletos que ganaron premio: debería coincidir con la probabilidad planificada."],
  winning_draws: ["KPI", "Sorteos con premio / Mejor premio", "Sorteos en los que el boleto ganó algo, y su mayor premio."],
  quintile: ["KPI", "Quintil", "Partidos o boletos divididos en cinco grupos iguales, del valor más bajo al más alto de una métrica. Una métrica importa cuando el KPI se mueve de forma constante entre grupos."],
  // Marcas y abreviaturas
  light: ["Marcas", "Semáforo", "La probabilidad de esta jornada frente a la de los boletos pasados: verde el 25 % más alto, ámbar lo normal, rojo el 25 % más bajo. Ninguno hace ganar dinero."],
  verdict: ["Marcas", "Veredicto", "Grande (jugar ·B), pequeño (sencillo o ·D) o no jugar, según los semáforos. Una primera regla, juzgada en metrics.html."],
  rec: ["Marcas", "Rec.", "Si el semáforo del boleto estaba en verde antes de la jornada."],
  M: ["Marcas", "M", "Partido de Liga o Segunda masculinas."],
  F: ["Marcas", "F", "Partido de Liga F (femenina)."],
  O: ["Marcas", "O", "Otra competición: selecciones, copas, ligas extranjeras."],
  U: ["Marcas", "U", "Competición desconocida."],
  Q: ["Marcas", "Q N", "El partido estaba en el boleto oficial de la quiniela de la jornada N."],
  Rmark: ["Marcas", "R", "Partido de una reducción oficial: no se juegan todas las combinaciones de sus signos."],
  signs: ["Marcas", "1 · X · 2", "Gana el local · empate · gana el visitante. 1X, X2 y 12 son dobles; 1X2 es un triple."],
  marks: ["Marcas", "✓ · ✗", "Acierto · fallo cuando el partido se ha jugado (✓ gris: cubierto por una reducción, la mejor columna aún puede fallarlo)."],
  one_in: ["Marcas", "1 de cada N", "Una probabilidad escrita como cuántas jornadas hacen falta, de media, para que pase una vez."],
  pleno: ["Marcas", "Pleno al 15", "Categoría exacta de goles de cada equipo (0, 1, 2, M = 3 o más) en el partido 15. Solo paga con 14 aciertos."],
  categories: ["Marcas", "Categorías de premio", "14 aciertos 1.ª, 13 2.ª, 12 3.ª, 11 4.ª, 10 5.ª; y el Pleno al 15 con 14 aciertos."],
  winners: ["Marcas", "Acertantes / premio por acertante", "Cuántas apuestas acertaron una categoría y los euros que cobró cada una (el bote repartido entre ellas)."],
  jornada: ["Marcas", "Jornada (J.)", "Número del sorteo oficial de la quiniela en su temporada."],
  walk_forward: ["Marcas", "Walk-forward", "Cada pronóstico se hace solo con lo que se sabía antes del partido: una prueba honesta."],
  standings: ["Marcas", "PJ G E P GF GC DG Pts", "Jugados, ganados, empatados, perdidos, goles a favor, goles en contra, diferencia de goles, puntos."],
  form: ["Marcas", "W · D · L", "Victoria, empate y derrota en los cinco últimos partidos de liga del equipo."],
};

/** The definition of a glossary term, for a data-tip attribute (escaped). */
function tip(key) {
  const entry = GLOSSARY[key];
  return entry ? esc(`${entry[1]}: ${entry[2]}`) : "";
}

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
