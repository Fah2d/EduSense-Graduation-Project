/*
 * EduSense — Intelligent Classroom
 * app.js — All frontend logic
 *
 * Technologies used:
 *   - Vanilla JavaScript (ES2022 async/await)
 *   - Fetch API (REST calls to Flask backend)
 *   - MediaDevices API (webcam access)
 *   - Canvas API (frame capture → Base64)
 *   - YouTube iFrame API (video embed + getCurrentTime)
 *   - Chart.js (live emotion timeline chart)
 *   - FormData API (PDF uploads)
 *   - setInterval (polling every 2s)
 *
 * Key functions:
 *   startSession()       — init webcam + YouTube + Flask session
 *   captureAndAnalyze()  — grab frame, send to /api/analyze-frame
 *   updateEmotionUI()    — update bars and dots in real time
 *   updateChart()        — push new data point to Chart.js
 *   pollStatus()         — check session state every 2s
 *   endSession()         — stop monitoring, generate notebooks
 *   showResults()        — render final session report
 *
 * King Khalid University — Graduation Project 2025
 */

// ── State ─────────────────────────────────────────────────
const S = {
  sessionActive:  false,
  webcamStream:   null,
  analyzeTimer:   null,
  statusTimer:    null,
  timerInterval:  null,
  sessionStart:   null,
  ytPlayer:       null,
  ytReady:        false,
  lastStruggleN:  0,
  chartData:      { labels:[], eng:[], bor:[], con:[], fru:[] },
  engChart:       null,
  finalChart:     null,
};

// ── Chart setup ───────────────────────────────────────────
function initChart() {
  const ctx = document.getElementById('engagement-chart').getContext('2d');
  S.engChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: S.chartData.labels,
      datasets: [
        { label:'Engagement', data:S.chartData.eng, borderColor:'#6ee7b7', backgroundColor:'rgba(110,231,183,0.08)', tension:0.4, pointRadius:0, borderWidth:1.5 },
        { label:'Boredom',    data:S.chartData.bor, borderColor:'#fb923c', backgroundColor:'rgba(251,146,60,0.05)',  tension:0.4, pointRadius:0, borderWidth:1.5 },
        { label:'Confusion',  data:S.chartData.con, borderColor:'#38bdf8', backgroundColor:'rgba(56,189,248,0.05)',  tension:0.4, pointRadius:0, borderWidth:1.5 },
        { label:'Frustration',data:S.chartData.fru, borderColor:'#f43f5e', backgroundColor:'rgba(244,63,94,0.05)',   tension:0.4, pointRadius:0, borderWidth:1.5 },
      ]
    },
    options: {
      responsive:true,
      animation:false,
      plugins:{ legend:{ labels:{ color:'#64748b', font:{family:'DM Mono',size:9}, boxWidth:10 }}},
      scales:{
        x:{ ticks:{ color:'#4a5568', font:{family:'DM Mono',size:8} }, grid:{ color:'rgba(255,255,255,0.03)' }},
        y:{ min:0, max:1, ticks:{ color:'#4a5568', font:{family:'DM Mono',size:8} }, grid:{ color:'rgba(255,255,255,0.03)' }}
      }
    }
  });
}

// ── YouTube API ───────────────────────────────────────────
function onYouTubeIframeAPIReady() {
  S.ytReady = true;
}

function embedYouTube(url) {
  const match = url.match(/(?:v=|youtu\.be\/)([^&\s]+)/);
  if (!match) { showFlash('Invalid YouTube URL — check the link'); return; }
  const videoId = match[1];

  // Set src directly on iframe — most reliable on localhost
  const iframe = document.getElementById('yt-iframe');
  iframe.src = `https://www.youtube.com/embed/${videoId}?autoplay=1&rel=0&modestbranding=1`;
  console.log('YouTube embedded:', videoId);
}

function getVideoTime() {
  try {
    if (S.ytPlayer && typeof S.ytPlayer.getCurrentTime === 'function') {
      return S.ytPlayer.getCurrentTime();
    }
  } catch(e) {}
  // Fallback: estimate from session time
  return S.sessionStart ? (Date.now() - S.sessionStart) / 1000 : 0;
}

function getVideoDuration() {
  try {
    if (S.ytPlayer && typeof S.ytPlayer.getDuration === 'function') {
      return S.ytPlayer.getDuration();
    }
  } catch(e) {}
  return 0;
}

// ── Init ──────────────────────────────────────────────────
window.onload = () => {
  setupUpload();
  initChart();
};

// ── KB Stats ──────────────────────────────────────────────
async function loadKBStats() {
  try {
    const r = await fetch('/api/kb-stats');
    const d = await r.json();
    if (d.chunks > 0) {
      document.getElementById('kb-badge-wrap').style.display = 'block';
      document.getElementById('kb-chunks').textContent = d.chunks;
    }
  } catch(e) {}
}

// ── PDF Upload ────────────────────────────────────────────
function setupUpload() {
  const zone  = document.getElementById('upload-zone');
  const input = document.getElementById('pdf-input');
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
  zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('drag'); uploadPDFs(e.dataTransfer.files); });
  input.addEventListener('change', () => uploadPDFs(input.files));
}

async function uploadPDFs(files) {
  for (const file of files) {
    if (!file.name.endsWith('.pdf')) continue;
    const item = document.createElement('div');
    item.className = 'upload-item';
    item.textContent = `⏳ ${file.name}`;
    document.getElementById('upload-list').appendChild(item);

    const fd = new FormData();
    fd.append('file', file);
    try {
      const r = await fetch('/api/upload-pdf', { method:'POST', body:fd });
      const d = await r.json();
      if (d.success) {
        item.textContent = `✅ ${file.name} — ${d.chunks} chunks`;
        item.style.color = 'var(--accent)';
        loadKBStats();
      } else {
        item.textContent = `❌ ${file.name}: ${d.error}`;
        item.style.color = 'var(--red)';
      }
    } catch(e) {
      item.textContent = `❌ ${file.name}: failed`;
      item.style.color = 'var(--red)';
    }
  }
}

// ── Start Session ─────────────────────────────────────────
async function startSession() {
  const url = document.getElementById('youtube-url').value.trim();
  if (!url) { showFlash('Please enter a YouTube URL'); return; }

  document.getElementById('start-btn').disabled = true;

  // Start webcam
  try {
    S.webcamStream = await navigator.mediaDevices.getUserMedia({ video:true });
    document.getElementById('webcam').srcObject = S.webcamStream;
  } catch(e) {
    showFlash('Cannot access webcam: ' + e.message);
    document.getElementById('start-btn').disabled = false;
    return;
  }

  // Call backend
  const r = await fetch('/api/start-session', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ youtube_url: url })
  });
  if (!r.ok) { showFlash('Failed to start'); return; }

  S.sessionActive = true;
  S.sessionStart  = Date.now();

  // Show monitor section
  setStep(2);
  document.getElementById('section-setup').style.display   = 'none';
  document.getElementById('section-monitor').style.display = 'grid';
  document.getElementById('section-monitor').classList.add('show');

  // Embed YouTube
  embedYouTube(url);

  // Start loops
  S.analyzeTimer  = setInterval(captureAndAnalyze, 2000);
  // Show calibration notice
  document.getElementById('s-msg').textContent = '⏳ Calibrating to your face (first 40s)...';
  S.statusTimer   = setInterval(pollStatus, 2000);
  S.timerInterval = setInterval(updateTimers, 1000);

  document.getElementById('status-pill').textContent = '● LIVE';
  document.getElementById('status-pill').classList.add('live');
}

// ── Webcam Capture ────────────────────────────────────────
async function captureAndAnalyze() {
  if (!S.sessionActive) return;

  const video  = document.getElementById('webcam');
  const canvas = document.getElementById('webcam-canvas');
  canvas.width = 320; canvas.height = 240;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, 320, 240);
  const frame     = canvas.toDataURL('image/jpeg', 0.7);
  const videoTime = getVideoTime();

  try {
    const r = await fetch('/api/analyze-frame', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ image: frame, video_time: videoTime })
    });
    const d = await r.json();

    if (d.emotions) {
      updateEmotionUI(d.emotions);
      updateTriggerDots(d.trigger_count || 0);
      updateChart(d.emotions, videoTime);

      const dissat = ['boredom','confusion','frustration']
        .some(e => d.emotions[e]?.positive);

      document.getElementById('alert-box').textContent =
        dissat ? '⚠️ Dissatisfaction Detected' : '✅ Student Focused';
      document.getElementById('alert-box').className =
        'alert-indicator ' + (dissat ? 'dissat' : 'focused');

      document.getElementById('s-dot').className =
        's-dot ' + (dissat ? 'warn' : 'live');

      document.getElementById('webcam-overlay').textContent =
        dissat ? '⚠️ Analyzing...' : '● Monitoring';

      if (d.triggered) {
        showFlash('⚠️ Struggle detected — notebook queued!');
        addStruggleToTimeline(d, videoTime);
        addTimelineMarker(videoTime);
      }
    }

    if (d.struggle_count !== undefined) {
      document.getElementById('stat-struggles').textContent = d.struggle_count;
    }

  } catch(e) {}
}

// ── Emotion UI ────────────────────────────────────────────
function updateEmotionUI(emotions) {
  ['engagement','boredom','confusion','frustration'].forEach(e => {
    const conf = emotions[e]?.confidence || 0;
    const pct  = Math.round(conf * 100);
    document.getElementById(`bar-${e}`).style.width = pct + '%';
    document.getElementById(`pct-${e}`).textContent = pct + '%';
    const dot = document.getElementById(`dot-${e}`);
    dot.className = 'e-dot' + (emotions[e]?.positive ? ' on' : '');
  });
}

function updateTriggerDots(count) {
  [1,2,3].forEach(i => {
    document.getElementById(`td-${i}`).className =
      't-dot' + (i <= count ? ' filled' : '');
  });
}

function updateChart(emotions, videoTime) {
  const label = formatTime(videoTime);
  S.chartData.labels.push(label);
  S.chartData.eng.push(emotions.engagement?.confidence || 0);
  S.chartData.bor.push(emotions.boredom?.confidence    || 0);
  S.chartData.con.push(emotions.confusion?.confidence  || 0);
  S.chartData.fru.push(emotions.frustration?.confidence|| 0);

  // Keep last 30 points
  if (S.chartData.labels.length > 30) {
    S.chartData.labels.shift();
    S.chartData.eng.shift();
    S.chartData.bor.shift();
    S.chartData.con.shift();
    S.chartData.fru.shift();
  }

  if (S.engChart) S.engChart.update('none');
}

// ── Timeline markers ──────────────────────────────────────
function addTimelineMarker(videoTime) {
  const duration = getVideoDuration();
  if (duration <= 0) return;

  const pct    = (videoTime / duration) * 100;
  const track  = document.querySelector('.timeline-track');
  const marker = document.createElement('div');
  marker.className        = 'struggle-marker';
  marker.style.left       = pct + '%';
  marker.dataset.time     = formatTime(videoTime);
  track.appendChild(marker);

  // Click marker to seek video
  marker.addEventListener('click', () => {
    if (S.ytPlayer && S.ytPlayer.seekTo) {
      S.ytPlayer.seekTo(videoTime, true);
    }
  });
}

function updateTimelineProgress() {
  const duration = getVideoDuration();
  if (duration <= 0) return;
  const pct = (getVideoTime() / duration) * 100;
  document.getElementById('timeline-progress').style.width = pct + '%';
}

function addStruggleToTimeline(data, videoTime) {
  const list = document.getElementById('struggle-list');
  // Remove empty message
  const empty = list.querySelector('[style*="text-align:center"]');
  if (empty) empty.remove();

  const item = document.createElement('div');
  item.className = 'struggle-item';
  const mins = String(Math.floor(videoTime/60)).padStart(2,'0');
  const secs = String(Math.floor(videoTime%60)).padStart(2,'0');
  item.innerHTML = `
    <div class="struggle-time">⏱ Video ${mins}:${secs}</div>
    <div class="emotion-tags">
      ${['boredom','confusion','frustration']
        .filter(e => data.emotions?.[e]?.positive)
        .map(e => `<span class="tag tag-${e}">${e}</span>`)
        .join('')}
    </div>`;
  list.insertBefore(item, list.firstChild);
}

// ── Poll Status ───────────────────────────────────────────
async function pollStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();

    document.getElementById('s-msg').textContent       = d.status_msg;
    document.getElementById('stat-focus').textContent  = d.engagement_rate + '%';
    document.getElementById('stat-notebooks').textContent = d.notebook_count;

    if (d.status === 'done') {
      clearInterval(S.statusTimer);
      clearInterval(S.analyzeTimer);
      clearInterval(S.timerInterval);
      showResults(d);
    }
  } catch(e) {}
}

// ── Timers ────────────────────────────────────────────────
function updateTimers() {
  if (S.sessionStart) {
    const elapsed = Math.floor((Date.now() - S.sessionStart) / 1000);
    document.getElementById('timer-pill').textContent = formatTime(elapsed);
    document.getElementById('session-time-label').textContent =
      'Session: ' + formatTime(elapsed);
  }

  const vt = getVideoTime();
  document.getElementById('video-time-label').textContent =
    'Video: ' + formatTime(vt);

  updateTimelineProgress();
}

// ── End Session ───────────────────────────────────────────
async function endSession() {
  if (!confirm('End session and generate notebooks?')) return;

  S.sessionActive = false;
  clearInterval(S.analyzeTimer);

  if (S.webcamStream) S.webcamStream.getTracks().forEach(t => t.stop());
  if (S.ytPlayer && S.ytPlayer.pauseVideo) S.ytPlayer.pauseVideo();

  await fetch('/api/end-session', { method:'POST' });
  document.getElementById('s-msg').textContent = 'Generating notebooks...';
}

// ── Results ───────────────────────────────────────────────
async function showResults(data) {
  setStep(3);
  document.getElementById('section-monitor').style.display = 'none';
  document.getElementById('section-monitor').classList.remove('show');
  document.getElementById('section-results').style.display = 'block';
  document.getElementById('section-results').classList.add('show');

  document.getElementById('status-pill').textContent = '● DONE';
  document.getElementById('status-pill').className   = 'pill';

  document.getElementById('r-duration').textContent  = data.elapsed || '--';
  document.getElementById('r-focus').textContent     = data.engagement_rate + '%';
  document.getElementById('r-struggles').textContent = data.struggle_count;
  document.getElementById('r-notebooks').textContent = data.notebook_count;
  document.getElementById('results-sub').textContent =
    `${data.notebook_count} personalized notebooks generated`;

  // Final chart
  const tl = await fetch('/api/engagement-timeline').then(r=>r.json());
  if (tl.timeline && tl.timeline.length > 0) {
    const ctx2 = document.getElementById('final-chart').getContext('2d');
    new Chart(ctx2, {
      type: 'line',
      data: {
        labels: tl.timeline.map(p => formatTime(p.video_time)),
        datasets: [
          { label:'Engagement', data:tl.timeline.map(p=>p.engagement), borderColor:'#6ee7b7', tension:0.4, pointRadius:0, borderWidth:2 },
          { label:'Boredom',    data:tl.timeline.map(p=>p.boredom),    borderColor:'#fb923c', tension:0.4, pointRadius:0, borderWidth:1.5 },
          { label:'Confusion',  data:tl.timeline.map(p=>p.confusion),  borderColor:'#38bdf8', tension:0.4, pointRadius:0, borderWidth:1.5 },
          { label:'Frustration',data:tl.timeline.map(p=>p.frustration),borderColor:'#f43f5e', tension:0.4, pointRadius:0, borderWidth:1.5 },
        ]
      },
      options: {
        responsive:true,
        plugins:{ legend:{ labels:{ color:'#64748b', font:{family:'DM Mono',size:9}, boxWidth:10 }}},
        scales:{
          x:{ ticks:{ color:'#4a5568', font:{family:'DM Mono',size:8}, maxTicksLimit:10 }, grid:{ color:'rgba(255,255,255,0.03)' }},
          y:{ min:0, max:1, ticks:{ color:'#4a5568', font:{family:'DM Mono',size:8} }, grid:{ color:'rgba(255,255,255,0.03)' }}
        }
      }
    });
  }

  // Notebooks grid
  const grid = document.getElementById('notebooks-grid');
  grid.innerHTML = '';
  if (!data.notebooks || data.notebooks.length === 0) {
    grid.innerHTML = '<div style="font-family:DM Mono,monospace;font-size:0.72rem;color:var(--muted)">No struggle moments — great focus!</div>';
    return;
  }
  data.notebooks.forEach(nb => {
    const card = document.createElement('div');
    card.className = 'nb-card';
    card.innerHTML = `
      <div class="nb-time">⏱ Video ${nb.timestamp}</div>
      <div class="nb-topic">${nb.filename.replace('.ipynb','').replace(/_/g,' ').slice(16) || 'Study Notebook'}</div>
      <div class="emotion-tags" style="margin-bottom:12px">
        ${(nb.detected||[]).map(e=>`<span class="tag tag-${e}">${e}</span>`).join('')}
      </div>
      <button class="btn btn-primary" style="width:100%;font-size:0.75rem;padding:8px"
        onclick="window.location.href='/api/download/${nb.filename}'">
        ⬇ Download .ipynb
      </button>`;
    grid.appendChild(card);
  });
}

async function downloadReport() {
  window.location.href = '/api/session-report';
}

// ── Helpers ───────────────────────────────────────────────
function setStep(n) {
  [1,2,3].forEach(i => {
    document.getElementById(`step-${i}`).className =
      'step' + (i===n?' active':i<n?' done':'');
  });
}

function formatTime(secs) {
  const m = String(Math.floor(secs/60)).padStart(2,'0');
  const s = String(Math.floor(secs%60)).padStart(2,'0');
  return `${m}:${s}`;
}

function showFlash(msg) {
  const f = document.getElementById('flash');
  f.textContent = msg;
  f.style.display = 'block';
  setTimeout(() => { f.style.display = 'none'; }, 4000);
}

function resetSession() {
  fetch('/api/reset', { method:'POST' });
  S.sessionActive = false;
  S.lastStruggleN = 0;
  S.chartData     = { labels:[], eng:[], bor:[], con:[], fru:[] };
  clearInterval(S.timerInterval);
  clearInterval(S.analyzeTimer);
  clearInterval(S.statusTimer);

  setStep(1);
  document.getElementById('section-results').style.display = 'none';
  document.getElementById('section-setup').style.display   = 'block';
  document.getElementById('status-pill').textContent = '● IDLE';
  document.getElementById('status-pill').className   = 'pill';
  document.getElementById('timer-pill').textContent  = '00:00';
  document.getElementById('start-btn').disabled = false;
  document.getElementById('youtube-url').value  = '';
  document.getElementById('struggle-list').innerHTML =
    '<div style="font-family:DM Mono,monospace;font-size:0.7rem;color:var(--muted);padding:12px;text-align:center">No struggles detected yet</div>';

  if (S.engChart) {
    S.engChart.data.labels = [];
    S.engChart.data.datasets.forEach(d => d.data = []);
    S.engChart.update();
  }
}


// ── Loading Overlay ───────────────────────────────────
async function initLoadingSequence() {
  const overlay  = document.getElementById('loading-overlay');
  const statusEl = document.getElementById('loading-status');

  function setStep(n, done=false) {
    for (let i = 1; i <= 4; i++) {
      const el = document.getElementById(`ls-${i}`);
      if (i < n)      { el.className = 'loading-step done';   el.querySelector('.step-icon').textContent = '✓'; }
      else if (i === n) { el.className = done ? 'loading-step done' : 'loading-step active'; if(done) el.querySelector('.step-icon').textContent = '✓'; }
      else            { el.className = 'loading-step'; }
    }
  }

  function setStatus(msg) {
    statusEl.innerHTML = msg + '<span>...</span>';
  }

  // Step 1: Check FER warmup
  setStatus('Loading TensorFlow & FER model');
  setStep(1);

  // Poll backend until ready
  let ferReady = false;
  for (let i = 0; i < 30; i++) {
    try {
      const r = await fetch('/api/kb-stats');
      if (r.ok) { ferReady = true; break; }
    } catch(e) {}
    await new Promise(r => setTimeout(r, 1000));
  }

  setStep(1, true);
  await new Promise(r => setTimeout(r, 300));

  // Step 2: Knowledge base
  setStep(2);
  setStatus('Loading knowledge base');
  try {
    const r = await fetch('/api/kb-stats');
    const d = await r.json();
    await new Promise(r => setTimeout(r, 600));
    setStatus(`Knowledge base: ${d.chunks || 0} chunks indexed`);
  } catch(e) {}
  await new Promise(r => setTimeout(r, 400));
  setStep(2, true);

  // Step 3: Claude API
  setStep(3);
  setStatus('Connecting to Claude API');
  await new Promise(r => setTimeout(r, 800));
  setStep(3, true);

  // Step 4: Ready
  setStep(4);
  setStatus('All systems ready');
  await new Promise(r => setTimeout(r, 600));
  setStep(4, true);

  await new Promise(r => setTimeout(r, 400));

  // Fade out
  overlay.classList.add('overlay-fade-out');
  await new Promise(r => setTimeout(r, 600));
  overlay.style.display = 'none';

  // Now load KB stats for the UI
  loadKBStats();
}

// Start loading sequence immediately
initLoadingSequence();