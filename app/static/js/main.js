let tempChart, lightChart, motionChart, co2Chart;

async function fetchData() {
  // Pull the latest sensor snapshots from the backend API.
  const response = await fetch('/api/data');
  const data = await response.json();
  updateCards(data);
  updateCharts(data);
}

function updateCards(data) {
  // Render the hero cards with the newest readings.
  const latest = data[0];
  const container = document.getElementById('sensor-cards');
  container.innerHTML = `
    <div class="col-md-3">
      <div class="card bg-danger text-light p-3">
        <h5>Temperature</h5>
        <p class="fs-4">${latest.temperature} °C</p>
      </div>
    </div>
    <div class="col-md-3">
      <div class="card bg-warning text-dark p-3">
        <h5>Light</h5>
        <p class="fs-4">${latest.illumination} lx</p>
      </div>
    </div>
    <div class="col-md-3">
      <div class="card bg-success text-light p-3">
        <h5>Motion</h5>
        <p class="fs-4">${latest.motion ? "Detected" : "None"}</p>
      </div>
    </div>
    <div class="col-md-3">
      <div class="card bg-info text-dark p-3">
        <h5>CO₂</h5>
        <p class="fs-4">${latest.co2} ppm</p>
      </div>
    </div>
  `;
}

function updateCharts(data) {
  // Rebuild the time-series charts based on the retrieved dataset.
  const timestamps = data.map(d => d.timestamp).reverse();
  const temps = data.map(d => d.temperature).reverse();
  const lights = data.map(d => d.illumination).reverse();
  const motions = data.map(d => d.motion ? 1 : 0).reverse();
  const co2s = data.map(d => d.co2).reverse();

  const opts = {
    scales: { 
      x: { ticks: { color: '#fff' } },
      y: { ticks: { color: '#fff' } } 
    },
    plugins: { legend: { labels: { color: '#fff' } } }
  };

  const ctxT = document.getElementById('tempChart');
  const ctxL = document.getElementById('lightChart');
  const ctxM = document.getElementById('motionChart');
  const ctxC = document.getElementById('co2Chart');

  // Tear down any existing chart instances before drawing fresh ones.
  if (tempChart) tempChart.destroy();
  if (lightChart) lightChart.destroy();
  if (motionChart) motionChart.destroy();
  if (co2Chart) co2Chart.destroy();

  tempChart = new Chart(ctxT, {
    type: 'line',
    data: { labels: timestamps, datasets: [{ label: 'Temperature (°C)', data: temps, borderColor: 'rgb(255,99,132)' }] },
    options: opts
  });

  lightChart = new Chart(ctxL, {
    type: 'line',
    data: { labels: timestamps, datasets: [{ label: 'Illumination (lx)', data: lights, borderColor: 'rgb(255,206,86)' }] },
    options: opts
  });

  motionChart = new Chart(ctxM, {
    type: 'bar',
    data: { labels: timestamps, datasets: [{ label: 'Motion', data: motions, backgroundColor: 'rgb(75,192,192)' }] },
    options: opts
  });

  co2Chart = new Chart(ctxC, {
    type: 'line',
    data: { labels: timestamps, datasets: [{ label: 'CO₂ (ppm)', data: co2s, borderColor: 'rgb(54,162,235)' }] },
    options: opts
  });
}

fetchData();
setInterval(fetchData, 5000);

let autoScrollEnabled = true;

// Safely escape HTML special characters in logs
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}


// Function to fetch logs from the backend
console.log("📜 Dashboard script loaded");

async function fetchLogs() {
  console.log("Fetching logs...");
  try {
    const response = await fetch('/api/logs');
    console.log("Response:", response.status);
    const logs = await response.json();
    console.log("Logs received:", logs.length);
    const container = document.getElementById('log-container');
    // Abort gracefully if the dashboard lacks a log container.
    if (!container) {
      console.error("No log-container element found.");
      return;
    }

    let html = "";
    // Render each log file block with sanitized contents.
    logs.forEach(log => {
  html += `<div><strong>${escapeHtml(log.filename)}</strong></div>`;
  const lines = (log.content || "").split(/\r?\n/).slice(-100);

  // Walk each individual log line while trimming whitespace.
  lines.forEach(line => {
    if (!line.trim()) return; // Skip blank lines to avoid noisy output.

    let cssClass = "log-info";
    const upper = line.toUpperCase();

    // Adjust styling based on the detected log severity keyword.
    if (upper.includes("ERROR")) cssClass = "log-error";
    else if (upper.includes("WARN")) cssClass = "log-warning";
    else if (upper.includes("DEBUG")) cssClass = "log-debug";

    html += `<div class="${cssClass}">${escapeHtml(line)}</div>`;
  });
});


    container.innerHTML = html;
    console.log("Updated DOM with logs");

    // Keep scrolling pinned to the latest entries when auto-scroll is enabled.
    if (autoScrollEnabled) {
      container.scrollTop = container.scrollHeight;
    }
  } catch (err) {
    console.error("❌ Error fetching logs:", err);
  }
}

// Handle Pause/Resume Auto-Scroll
const toggleBtn = document.getElementById('toggle-scroll');

if (toggleBtn) {
  // Toggle auto-scroll on click so users can pause the output.
  toggleBtn.addEventListener('click', () => {
    autoScrollEnabled = !autoScrollEnabled;
    toggleBtn.textContent = autoScrollEnabled
      ? "⏸ Pause Auto-Scroll"
      : "▶ Resume Auto-Scroll";

    // If re-enabled, jump to the newest logs immediately
    const container = document.getElementById('log-container');
    if (autoScrollEnabled && container) {
      container.scrollTop = container.scrollHeight;
    }
  });
}

document.addEventListener('DOMContentLoaded', () => {
  // Start the log polling loop once the DOM is ready.
  console.log("DOM fully loaded, starting fetch interval");
  fetchLogs();
  setInterval(fetchLogs, 8000);
});
