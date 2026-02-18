
const API_BASE = `${window.location.protocol}//${window.location.hostname}:5050`;

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
