
const API_BASE = (() => {
  const isLocalStaticFrontend =
    (window.location.hostname === "localhost" ||
      window.location.hostname === "127.0.0.1" ||
      window.location.hostname === "::1") &&
    window.location.port === "8000";

  const apiOrigin = isLocalStaticFrontend
    ? window.location.origin.replace(":8000", ":5050")
    : window.location.origin;

  return isLocalStaticFrontend ? apiOrigin : `${apiOrigin}/api`;
})();

async function loadStats(){
let r = await fetch(`${API_BASE}/dashboard/stats`)
let data = await r.json()

document.getElementById("stats").innerHTML = `
Active Restaurants: ${data.restaurants}<br>
Total Calls: ${data.calls}<br>
Missed Calls: ${data.missed}<br>
Avg Duration: ${data.avg_duration}
`
}
loadStats()

