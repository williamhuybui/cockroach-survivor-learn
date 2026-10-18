const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (n) => (n > 0 ? "+" : "") + Number(n).toFixed(1).replace(/\.0$/, "");
let data = { teams: [] };
let mode = "practice";

const runOf = (team, m) => {
  const runs = team.runs.filter((r) => r.mode === m);
  return runs.length ? runs[runs.length - 1] : null;
};

function render() {
  const rows = data.teams
    .map((t) => ({ t, r: runOf(t, mode) }))
    .sort((a, b) => (b.r ? b.r.score : -Infinity) - (a.r ? a.r.score : -Infinity));

  $("#board tbody").innerHTML = rows.map(({ t, r }, i) => {
    if (!r) return `<tr><td>–</td><td>${esc(t.name)}</td><td colspan="2" class="muted">No ${mode} run yet</td></tr>`;
    return `<tr class="${i === 0 ? "first" : ""}"><td>${i + 1}</td><td><strong>${esc(t.name)}</strong></td>
      <td class="score">${fmt(r.score)}</td>
      <td class="muted">${esc(r.date)}</td></tr>`;
  }).join("");
}

async function load() {
  try {
    data = await (await fetch("leaderboard.json", { cache: "no-store" })).json();
    $("#updated").textContent = data.updated ? new Date(data.updated).toLocaleString() : "never";
    $("#teams").innerHTML = data.teams.map((t) =>
      `<div class="card"><h4>${esc(t.name)}</h4><div>Captain: ${esc(t.captain)}</div><div class="muted">${t.members.map(esc).join(", ")}</div></div>`).join("");
  } catch (e) {
    $("#updated").textContent = "(couldn't load leaderboard.json, serve this folder over http)";
  }
  render();
}

$("#tabs").addEventListener("click", (e) => {
  if (!e.target.dataset.mode) return;
  mode = e.target.dataset.mode;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b === e.target));
  render();
});
load();

// deadline bar: Oct 18 -> Nov 1 (2026)
(function () {
  const start = new Date(2026, 9, 18), end = new Date(2026, 10, 1, 23, 59, 59), now = new Date();
  const pct = Math.min(100, Math.max(0, (100 * (now - start)) / (end - start)));
  const days = Math.ceil((end - now) / 864e5);
  $("#dl-fill").style.width = pct.toFixed(1) + "%";
  $("#dl-left").textContent = days > 0 ? `${days} day${days > 1 ? "s" : ""} left` : "Deadline passed";
})();
