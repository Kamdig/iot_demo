let tempChart, lightChart, motionChart, co2Chart;

async function fetchData() {
  const response = await fetch('/api/data');
  const data = await response.json();
  updateCards(data);
  updateCharts(data);
}

function updateCards(data) {
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
