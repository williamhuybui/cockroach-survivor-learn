const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (n) => (n > 0 ? "+" : "") + Number(n).toFixed(1).replace(/\.0$/, "");
let data = { teams: [] };
let scores = [];  // rows of scores.csv: { timestamp, date, team, mode, trial, questions, score }
let mode = "practice";

const runsOf = (team, m) => scores.filter((r) => r.team === team.id && r.mode === m);
const runOf = (team, m) => runsOf(team, m).reduce((best, r) => (!best || r.score > best.score ? r : best), null);  // a team's highest-scoring run
const pct = (r) => (Number(r.questions) > 0 ? ((100 * r.score) / Number(r.questions)).toFixed(1).replace(/\.0$/, "") + "%" : "");
const open = new Set();  // teams whose run history is expanded
const shown = {};        // last value each number was animated to, so only changed numbers count up again
let entered = false;     // rows slide in only the first time

function countUp(el, to) {
  const isPct = el.dataset.kind === "pct";
  const text = (v) => (isPct ? v.toFixed(1).replace(/\.0$/, "") + "%" : fmt(v));
  const t0 = performance.now(), dur = 1000;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce) { el.textContent = text(to); return; }
  (function tick(now) {
    const k = Math.min(1, (now - t0) / dur), ease = 1 - Math.pow(1 - k, 3);
    el.textContent = text(to * ease);
    if (k < 1) requestAnimationFrame(tick);
  })(t0);
}

function animateNumbers() {
  document.querySelectorAll("#board [data-count]").forEach((el) => {
    const to = Number(el.dataset.count), key = el.dataset.key;
    if (shown[key] === to) { el.textContent = el.dataset.final; return; }
    shown[key] = to;
    countUp(el, to);
  });
}

function parseCsv(text) {
  const [head, ...lines] = text.trim().split(/\r?\n/).map((l) => l.split(","));
  return lines.filter((l) => l.length === head.length).map((l) => Object.fromEntries(head.map((h, i) => [h, l[i]])));
}

function render() {
  const rows = data.teams
    .map((t) => ({ t, r: runOf(t, mode) }))
    .sort((a, b) => (b.r ? b.r.score : -Infinity) - (a.r ? a.r.score : -Infinity));

  $("#board tbody").innerHTML = rows.map(({ t, r }, i) => {
    if (!r) return `<tr><td>–</td><td>${esc(t.name)}</td><td colspan="2" class="muted">No ${mode} run yet</td></tr>`;
    const runs = runsOf(t, mode);
    const pv = Number(r.questions) > 0 ? (100 * r.score) / Number(r.questions) : 0;
    const main = `<tr class="${i === 0 ? "first" : ""} team-row ${entered ? "" : "enter"}" style="--i:${i}" data-team="${esc(t.id)}"><td>${i + 1}</td>
      <td><strong>${esc(t.name)}</strong> <span class="muted hist-toggle">${open.has(t.id) ? "▾" : "▸"} ${runs.length} run${runs.length > 1 ? "s" : ""}</span></td>
      <td class="score"><span data-count="${r.score}" data-kind="score" data-key="${esc(t.id)}-${mode}-s" data-final="${fmt(r.score)}">${fmt(r.score)}</span><span class="muted"> / ${esc(r.questions)}</span>
        <span class="pct" data-count="${pv}" data-kind="pct" data-key="${esc(t.id)}-${mode}-p" data-final="${pct(r)}">${pct(r)}</span>
        <div class="meter"><i style="--w:${Math.max(0, Math.min(100, pv)).toFixed(1)}%"></i></div></td>
      <td class="muted">Trial ${esc(r.trial)} · ${esc(r.date)}</td></tr>`;
    if (!open.has(t.id)) return main;
    const history = runs.slice().reverse().map((h) =>
      `<div class="hist-row"><span>Trial ${esc(h.trial)}</span><span class="muted">${esc(h.date)}</span><span>${fmt(h.score)} / ${esc(h.questions)}</span><b>${pct(h)}</b></div>`).join("");
    return main + `<tr class="history"><td></td><td colspan="3">${history}</td></tr>`;
  }).join("");
  entered = true;
  animateNumbers();
}

async function load() {
  try {
    data = await (await fetch("leaderboard.json", { cache: "no-store" })).json();
    scores = parseCsv(await (await fetch("scores.csv", { cache: "no-store" })).text()).map((r) => ({ ...r, score: Number(r.score) }));
    $("#updated").textContent = scores.length ? new Date(scores[scores.length - 1].timestamp).toLocaleString() : "never";
    $("#teams").innerHTML = data.teams.map((t) =>
      `<div class="card"><h4>${esc(t.name)}</h4><div>Captain: ${esc(t.captain)}</div><div class="muted">${t.members.map(esc).join(", ")}</div></div>`).join("");
  } catch (e) {
    $("#updated").textContent = "(couldn't load leaderboard.json / scores.csv, serve this folder over http)";
  }
  render();
}

$("#board").addEventListener("click", (e) => {
  const row = e.target.closest("tr.team-row");
  if (!row) return;
  const id = row.dataset.team;
  open.has(id) ? open.delete(id) : open.add(id);
  render();
});

$("#tabs").addEventListener("click", (e) => {
  if (!e.target.dataset.mode) return;
  mode = e.target.dataset.mode;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b === e.target));
  render();
});
load();
setInterval(load, 20000);  // pick up new scores without a page reload

// deadline bar: Oct 18 -> Nov 8 (2026)
(function () {
  const start = new Date(2026, 9, 18), end = new Date(2026, 10, 8, 23, 59, 59), now = new Date();
  const pct = Math.min(100, Math.max(0, (100 * (now - start)) / (end - start)));
  const days = Math.ceil((end - now) / 864e5);
  $("#dl-fill").style.width = pct.toFixed(1) + "%";
  $("#dl-left").textContent = days > 0 ? `${days} day${days > 1 ? "s" : ""} left` : "Deadline passed";
})();
