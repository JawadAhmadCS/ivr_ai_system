
async function loadStats(){
let r = await fetch("http://localhost:8000/dashboard/stats")
let data = await r.json()

document.getElementById("stats").innerHTML = `
Active Restaurants: ${data.restaurants}<br>
Total Calls: ${data.calls}<br>
Missed Calls: ${data.missed}<br>
Avg Duration: ${data.avg_duration}
`
}
loadStats()
