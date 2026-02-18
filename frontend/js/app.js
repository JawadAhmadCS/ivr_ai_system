
const API_BASE = `${window.location.origin}/api`;

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

