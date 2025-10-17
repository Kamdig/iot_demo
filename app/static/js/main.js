let chart;

async function fetchData() {
  const response = await fetch('/api/data');
  const sensors = await response.json();
  updateCards(sensors);
  updateChart(sensors);
}

function updateCards(sensors) {
  const container = document.getElementById('sensor-cards');
  container.innerHTML = sensors.map(sensor => `
    <div class="col-md-4 mb-3">
      <div class="card bg-secondary shadow-sm p-3 h-100">
        <h5>${sensor.name || 'Unknown Sensor'}</h5>
        <p class="mb-1">Value: <strong>${sensor.value}</strong></p>
        <p class="text-muted small">${sensor.timestamp || ''}</p>
      </div>
    </div>
  `).join('');
}

function updateChart(sensors) {
  const ctx = document.getElementById('sensorChart').getContext('2d');
  const labels = sensors.map(s => s.name);
  const values = sensors.map(s => s.value);

  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Sensor Values',
        data: values,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.3,
      }]
    },
    options: {
      scales: { y: { beginAtZero: true } },
      plugins: { legend: { labels: { color: '#fff' } } }
    }
  });
}

fetchData();
setInterval(fetchData, 5000);
