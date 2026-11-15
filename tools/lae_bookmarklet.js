// Quiniela bookmarklet: one click on a Loterías y Apuestas del Estado page saves its data.
//
// Install it once with tools/lae_bookmarklet.html (drag the link to the bookmarks bar).
// `invoke update` opens the pages it needs in your browser; click the bookmark on each:
//
// - On the betting page (https://juegos.loteriasyapuestas.es/jugar/la-quiniela/apuesta)
//   it downloads the next ticket: matchday, sales deadline, the 15 matches and the
//   public's share of each sign ("Pronósticos más apostados"), as
//   quiniela_upcoming_j<jornada>.json.
// - On the results service (www.loteriasyapuestas.es/servicios/buscadorSorteos...) it
//   downloads the season's draws (tickets, results, winners and prizes) as
//   quiniela_lae_<season>.json.
//
// It runs in your browser, on the page you opened, like any visitor: nothing is fetched
// from a script. `invoke update` picks the files up from the Downloads folder and moves
// them to data/lae/.
(() => {
  const save = (name, text) => {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([text], { type: "application/json" }));
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  if (location.href.includes("buscadorSorteos")) {
    // Results service: the raw JSON of the page (the browser may show it formatted)
    fetch(location.href)
      .then((response) => response.text())
      .then((text) => {
        const draws = JSON.parse(text);
        const season = String((draws[0] || {}).temporada || "").replace(/^(\d{4})-\d{2}(\d{2})$/, "$1-$2");
        if (!Array.isArray(draws) || !draws.length || !season) {
          alert("Quiniela: no draws on this page.");
          return;
        }
        save(`quiniela_lae_${season}.json`, text);
      })
      .catch((error) => alert(`Quiniela: could not read the draws (${error}).`));
    return;
  }

  // Betting page: what the page shows about the next ticket
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
  if (!jornada || !closes || matches.length < 14) {
    alert("Quiniela: open the La Quiniela betting page (apuesta) and wait until it loads.");
    return;
  }
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
  save(`quiniela_upcoming_j${jornada}.json`, JSON.stringify(data, null, 1));
})();
