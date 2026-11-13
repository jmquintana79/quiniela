// Save the next quiniela ticket from the official betting page as upcoming.json.
//
// Open https://juegos.loteriasyapuestas.es/jugar/la-quiniela/apuesta, paste this
// script in the browser console (developer tools) and press Enter: the browser
// downloads upcoming.json. Move it to data/lae/upcoming.json and run
// `invoke dashboards.build` (or `invoke update`).
//
// It reads what the page shows: the matchday, the sales closing time, the 15
// matches and the public's share of each sign ("Pronósticos más apostados").
(() => {
  const text = document.body.innerText;
  const jornada = Number((text.match(/Jornada\s*(\d+)/) || [])[1]);
  const closes = (text.match(/Cierre de ventas:\s*([\d/]+\s*-\s*[\d:]+)/) || [])[1];
  const lines = [...new Set((text.match(/^\s*\d{1,2}\.\s.+\s-\s.+$/gm) || []).map((l) => l.trim()))];
  const matches = lines
    .map((l) => {
      const m = l.match(/^(\d{1,2})\.\s(.+?)\s-\s(.+)$/);
      return { position: Number(m[1]), home: m[2].trim(), away: m[3].trim() };
    })
    .filter((m, i, all) => all.findIndex((x) => x.position === m.position) === i)
    .sort((a, b) => a.position - b.position);
  const stats = [...document.querySelectorAll(".estadisticas-pronostico")]
    .slice(0, 14)
    .map((e) => e.textContent.trim().split(/\s+/).map(Number));
  matches.forEach((m, i) => {
    if (m.position <= 14 && stats[i] && stats[i].length === 3) {
      m.public = { 1: stats[i][0], X: stats[i][1], 2: stats[i][2] };
    }
  });
  // Pleno al 15: goals categories 0, 1, 2, M of each team (labels removed first)
  const pleno = document.querySelector(".pleno-15:not(.partidos)");
  const values = pleno ? (pleno.textContent.replace(/0\s*1\s*2\s*M/g, " ").match(/\d+/g) || []).map(Number) : [];
  const fifteen = matches.find((m) => m.position === 15);
  if (fifteen && values.length >= 8) {
    const [h0, h1, h2, hm, a0, a1, a2, am] = values.slice(0, 8);
    fifteen.public_home = { 0: h0, 1: h1, 2: h2, M: hm };
    fifteen.public_away = { 0: a0, 1: a1, 2: a2, M: am };
  }
  const data = { source: location.href.split("?")[0], saved_at: new Date().toISOString(), jornada, closes, matches };
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 1)], { type: "application/json" }));
  link.download = "upcoming.json";
  link.click();
  return data;
})();
