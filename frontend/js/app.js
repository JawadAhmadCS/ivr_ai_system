
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

const AUTH_TOKEN = localStorage.getItem("auth_token") || "";

async function apiFetch(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (AUTH_TOKEN) headers.Authorization = `Bearer ${AUTH_TOKEN}`;
  const res = await fetch(path, { ...options, headers });
  if (res.status === 401) {
    throw new Error("Unauthorized");
  }
  return res;
}

async function loadStats(){
let r = await apiFetch(`${API_BASE}/dashboard/stats`)
let data = await r.json()

document.getElementById("stats").innerHTML = `
Active Restaurants: ${data.restaurants}<br>
Total Calls: ${data.calls}<br>
Missed Calls: ${data.missed}<br>
Avg Duration: ${data.avg_duration}
`
}
loadStats()
