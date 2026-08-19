function navigateTo(pageName) {
  const pages = document.querySelectorAll('.page');
  pages.forEach(p => p.classList.remove('active'));

  document.querySelectorAll('.nav-link').forEach(l => {
    l.classList.remove('active');
    if (l.getAttribute('data-page') === pageName) {
      l.classList.add('active');
    }
  });

  const target = document.getElementById('page-' + pageName);
  if (target) {
    target.classList.add('active');
  }

  // Trigger chart resize & initialization when entering Analysis
  if (pageName === 'analysis') {
    if (!window._chartInitialized) {
      initCandlestickChart();
      window._chartInitialized = true;
    } else if (window._chartInstance && window._chartContainer) {
      setTimeout(() => {
        window._chartInstance.applyOptions({
          width: window._chartContainer.clientWidth,
          height: window._chartContainer.clientHeight
        });
        window._chartInstance.timeScale().fitContent();
      }, 50);
    }
  }
}

function initThreeJS() {
  const container = document.getElementById('hero-canvas');
  if (!container || typeof THREE === 'undefined') return;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });

  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  // Outer wireframe icosahedron
  const geometry = new THREE.IcosahedronGeometry(2.6, 2);
  const material = new THREE.MeshBasicMaterial({
    color: 0xf7931a,
    wireframe: true,
    transparent: true,
    opacity: 0.14
  });
  const sphere = new THREE.Mesh(geometry, material);
  scene.add(sphere);

  // Inner core
  const innerGeo = new THREE.IcosahedronGeometry(1.6, 1);
  const innerMat = new THREE.MeshBasicMaterial({
    color: 0xf7931a,
    wireframe: true,
    transparent: true,
    opacity: 0.05
  });
  const innerSphere = new THREE.Mesh(innerGeo, innerMat);
  scene.add(innerSphere);

  // Particle cloud
  const count = 280;
  const pos = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const angle = (i / count) * Math.PI * 2;
    const r = 4 + Math.random() * 2;
    pos[i * 3] = Math.cos(angle) * r;
    pos[i * 3 + 1] = (Math.random() - 0.5) * 2;
    pos[i * 3 + 2] = Math.sin(angle) * r;
  }
  const partGeo = new THREE.BufferGeometry();
  partGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const partMat = new THREE.PointsMaterial({
    color: 0xf7931a,
    size: 0.035,
    transparent: true,
    opacity: 0.3
  });
  const particles = new THREE.Points(partGeo, partMat);
  scene.add(particles);

  camera.position.z = 6.5;

  function animate() {
    requestAnimationFrame(animate);
    sphere.rotation.x += 0.0015;
    sphere.rotation.y += 0.0025;
    innerSphere.rotation.x -= 0.002;
    particles.rotation.y += 0.0006;
    renderer.render(scene, camera);
  }
  animate();

  window.addEventListener('resize', () => {
    if (container.clientWidth === 0) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });
}

let candleSeries, volumeSeries, currentBar;

function generateBTCData() {
  const candles = [];
  const volume = [];
  let price = 64750.0;
  const now = Math.floor(Date.now() / 1000);
  const startTime = now - 120 * 60; // 120 1-minute candles

  for (let i = 0; i < 120; i++) {
    const time = startTime + i * 60;
    const volatility = 12.0 + Math.random() * 20.0;
    const change = (Math.random() - 0.49) * volatility;
    const open = price;
    const close = open + change;
    const high = Math.max(open, close) + Math.random() * 10.0;
    const low = Math.min(open, close) - Math.random() * 10.0;
    const vol = Math.floor(10 + Math.random() * 65);

    candles.push({
      time: time,
      open: parseFloat(open.toFixed(2)),
      high: parseFloat(high.toFixed(2)),
      low: parseFloat(low.toFixed(2)),
      close: parseFloat(close.toFixed(2))
    });

    volume.push({
      time: time,
      value: vol,
      color: close >= open ? 'rgba(34, 197, 94, 0.35)' : 'rgba(255, 71, 87, 0.35)'
    });

    price = close;
  }

  currentBar = { ...candles[candles.length - 1] };
  return { candles, volume };
}

function initCandlestickChart() {
  const container = document.getElementById('candlestick-chart');
  if (!container || typeof LightweightCharts === 'undefined') return;

  window._chartContainer = container;

  const chart = LightweightCharts.createChart(container, {
    layout: {
      background: { type: 'solid', color: '#080c14' },
      textColor: '#8a95a8',
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: 11
    },
    grid: {
      vertLines: { color: 'rgba(255, 255, 255, 0.03)' },
      horzLines: { color: 'rgba(255, 255, 255, 0.03)' }
    },
    crosshair: {
      mode: LightweightCharts.CrosshairMode.Normal,
      vertLine: {
        color: 'rgba(247, 147, 26, 0.4)',
        width: 1,
        style: LightweightCharts.LineStyle.Dashed,
        labelBackgroundColor: '#f7931a'
      },
      horzLine: {
        color: 'rgba(247, 147, 26, 0.4)',
        width: 1,
        style: LightweightCharts.LineStyle.Dashed,
        labelBackgroundColor: '#f7931a'
      }
    },
    rightPriceScale: {
      borderColor: 'rgba(255, 255, 255, 0.08)',
      autoScale: true,
      scaleMargins: { top: 0.1, bottom: 0.2 }
    },
    timeScale: {
      borderColor: 'rgba(255, 255, 255, 0.08)',
      timeVisible: true,
      secondsVisible: false,
      barSpacing: 9,
      minBarSpacing: 3,
      rightOffset: 6
    }
  });

  window._chartInstance = chart;

  candleSeries = chart.addCandlestickSeries({
    upColor: '#22c55e',
    downColor: '#ff4757',
    borderUpColor: '#22c55e',
    borderDownColor: '#ff4757',
    wickUpColor: '#22c55e',
    wickDownColor: '#ff4757'
  });

  volumeSeries = chart.addHistogramSeries({
    priceFormat: { type: 'volume' },
    priceScaleId: '',
    scaleMargins: { top: 0.82, bottom: 0 }
  });

  const data = generateBTCData();
  candleSeries.setData(data.candles);
  volumeSeries.setData(data.volume);
  chart.timeScale().fitContent();

  // Crosshair hover OHLC update
  chart.subscribeCrosshairMove((param) => {
    if (!param || !param.time || !param.seriesPrices) return;
    const priceObj = param.seriesPrices.get(candleSeries);
    const volObj = param.seriesPrices.get(volumeSeries);

    if (priceObj) {
      document.getElementById('ohlc-open').textContent = priceObj.open.toFixed(2);
      document.getElementById('ohlc-high').textContent = priceObj.high.toFixed(2);
      document.getElementById('ohlc-low').textContent = priceObj.low.toFixed(2);
      document.getElementById('ohlc-close').textContent = priceObj.close.toFixed(2);
      document.getElementById('live-price-val').textContent = `$${priceObj.close.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
    }
    if (volObj !== undefined) {
      document.getElementById('ohlc-vol').textContent = `${volObj} BTC`;
    }
  });

  // Start Live Ticking Simulation
  startLiveMarketSimulation();

  // Resize Observer
  const resizeObserver = new ResizeObserver(entries => {
    for (let entry of entries) {
      if (entry.contentRect.width > 0 && entry.contentRect.height > 0) {
        chart.applyOptions({
          width: entry.contentRect.width,
          height: entry.contentRect.height
        });
      }
    }
  });
  resizeObserver.observe(container);
}

// Live Ticking Engine: Updates Price & Telemetry Values in Real Time
function startLiveMarketSimulation() {

  // 1. Tick Candlestick Price every 1.0s
  setInterval(() => {
    if (!candleSeries || !currentBar) return;

    const tickDelta = (Math.random() - 0.49) * 4.2;
    currentBar.close = parseFloat((currentBar.close + tickDelta).toFixed(2));
    if (currentBar.close > currentBar.high) currentBar.high = currentBar.close;
    if (currentBar.close < currentBar.low) currentBar.low = currentBar.close;

    candleSeries.update(currentBar);

    // Update Header Toolbar Price & OHLC
    const livePriceEl = document.getElementById('live-price-val');
    const closeEl = document.getElementById('ohlc-close');
    const highEl = document.getElementById('ohlc-high');
    const lowEl = document.getElementById('ohlc-low');

    if (livePriceEl) {
      livePriceEl.textContent = `$${currentBar.close.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
      livePriceEl.style.color = tickDelta >= 0 ? '#22c55e' : '#ff4757';
    }
    if (closeEl) closeEl.textContent = currentBar.close.toFixed(2);
    if (highEl) highEl.textContent = currentBar.high.toFixed(2);
    if (lowEl) lowEl.textContent = currentBar.low.toFixed(2);

  }, 1000);

  // 2. Tick Telemetry Metrics every 1.8s (Moving averages, RSI, Volatility fluctuate like a live stream)
  setInterval(() => {
    const ma5 = (0.0018 + (Math.random() - 0.48) * 0.0012).toFixed(4);
    const ma15 = (0.0048 + (Math.random() - 0.48) * 0.0014).toFixed(4);
    const ma60 = (-0.0098 + (Math.random() - 0.5) * 0.001).toFixed(4);
    const rsi = (51.5 + (Math.random() - 0.48) * 7.0).toFixed(1);
    const vol = (0.0013 + (Math.random() - 0.48) * 0.0004).toFixed(4);
    const volRatio = (1.18 + (Math.random() - 0.48) * 0.12).toFixed(2);

    const d5 = document.getElementById('disp_ma_dist_5');
    const d15 = document.getElementById('disp_ma_dist_15');
    const d60 = document.getElementById('disp_ma_dist_60');
    const dRsi = document.getElementById('disp_rsi_14');
    const dVol = document.getElementById('disp_volatility_20');
    const dVRatio = document.getElementById('disp_vol_ratio_15m');

    if (d5) {
      d5.textContent = parseFloat(ma5) >= 0 ? `+${ma5}` : `${ma5}`;
      d5.style.color = parseFloat(ma5) >= 0 ? '#22c55e' : '#ff4757';
    }
    if (d15) {
      d15.textContent = parseFloat(ma15) >= 0 ? `+${ma15}` : `${ma15}`;
      d15.style.color = parseFloat(ma15) >= 0 ? '#22c55e' : '#ff4757';
    }
    if (d60) {
      d60.textContent = parseFloat(ma60) >= 0 ? `+${ma60}` : `${ma60}`;
      d60.style.color = parseFloat(ma60) >= 0 ? '#22c55e' : '#ff4757';
    }
    if (dRsi) dRsi.textContent = rsi;
    if (dVol) dVol.textContent = vol;
    if (dVRatio) dVRatio.textContent = `${volRatio}x`;

    // Update hidden input fields so sendPrediction() sends the exact live numbers to backend
    if (document.getElementById('ma_dist_5')) document.getElementById('ma_dist_5').value = ma5;
    if (document.getElementById('ma_dist_15')) document.getElementById('ma_dist_15').value = ma15;
    if (document.getElementById('ma_dist_60')) document.getElementById('ma_dist_60').value = ma60;
    if (document.getElementById('rsi_14')) document.getElementById('rsi_14').value = rsi;
    if (document.getElementById('volatility_20')) document.getElementById('volatility_20').value = vol;
    if (document.getElementById('vol_ratio_15m')) document.getElementById('vol_ratio_15m').value = volRatio;

  }, 1800);
}

function showToast(message, type) {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);

  const duration = type === 'error' ? 4500 : 3000;

  setTimeout(() => {
    toast.classList.add('out');
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

async function sendPrediction() {
  const btn = document.getElementById('btn-run');
  btn.classList.add('loading');

  const payload = {
    ma_dist_5: parseFloat(document.getElementById('ma_dist_5').value),
    ma_dist_15: parseFloat(document.getElementById('ma_dist_15').value),
    ma_dist_60: parseFloat(document.getElementById('ma_dist_60').value),
    volatility_20: parseFloat(document.getElementById('volatility_20').value),
    rsi_14: parseFloat(document.getElementById('rsi_14').value),
    vol_ratio_15m: parseFloat(document.getElementById('vol_ratio_15m').value)
  };

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const result = await response.json();

    const signalBox = document.getElementById('signal-display');
    const returnBox = document.getElementById('return-display');
    const mainViewport = document.getElementById('analysis-main');

    signalBox.innerText = result.signal;
    returnBox.innerText = result.predicted_return;

    mainViewport.classList.remove('sell-flash', 'buy-flash');

    if (result.signal === 'BUY') {
      signalBox.className = 'signal-badge buy';
      returnBox.className = 'return-positive';
      mainViewport.classList.add('buy-flash');
      showToast(`BUY SIGNAL CONFIRMED | Expected Return: ${result.predicted_return}`, 'success');
    } else {
      signalBox.className = 'signal-badge sell';
      returnBox.className = 'return-negative';
      mainViewport.classList.add('sell-flash');
      showToast(`SELL SIGNAL CONFIRMED | Expected Return: ${result.predicted_return}`, 'error');
    }

  } catch (err) {
    showToast(`Prediction Request Error: ${err.message}`, 'error');
  } finally {
    btn.classList.remove('loading');
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initThreeJS();

  // Real-time UTC clock
  const timeEl = document.getElementById('chart-time');
  function updateTime() {
    if (timeEl) {
      const now = new Date();
      timeEl.textContent = now.toLocaleTimeString('en-GB', { hour12: false });
    }
  }
  updateTime();
  setInterval(updateTime, 1000);
});
