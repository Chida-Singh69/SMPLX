let currentVideoId = null;
let abortController = null;
const API_BASE = 'http://127.0.0.1:5000';
let aslClips = [];
let extensionEnabled = true;

function getLocalSettings(keys) {
  return new Promise((resolve) => chrome.storage.local.get(keys, resolve));
}

function getVideoId() {
  const params = new URLSearchParams(window.location.search);
  return params.get('v');
}

function onUrlChange() {
  const newId = getVideoId();
  if (newId && newId !== currentVideoId) {
    currentVideoId = newId;
    if (extensionEnabled) {
      initOverlay(newId);
    } else {
      teardownOverlay({ keepVideoId: true });
    }
  }
}

function teardownOverlay({ keepVideoId = false } = {}) {
  if (abortController) abortController.abort();
  const old = document.getElementById('sv-asl-overlay');
  if (old) old.remove();
  aslClips = [];
  if (!keepVideoId) currentVideoId = null;
}

function cleanup() {
  teardownOverlay({ keepVideoId: false });
}

function showOrCreateOverlayForCurrentVideo() {
  const overlay = document.getElementById('sv-asl-overlay');
  if (overlay) {
    overlay.classList.remove('sv-hidden');
    return;
  }
  if (currentVideoId) {
    initOverlay(currentVideoId);
  }
}

function applyEnabledState(enabled) {
  extensionEnabled = enabled !== false;
  if (!extensionEnabled) {
    teardownOverlay({ keepVideoId: true });
    return;
  }
  showOrCreateOverlayForCurrentVideo();
}

chrome.runtime.onMessage.addListener((msg) => {
  if (!msg || typeof msg !== 'object') return;
  if (msg.type === 'sv-toggle-overlay') {
    applyEnabledState(msg.enabled);
  }
});

chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName !== 'local') return;
  if (changes.enabled) {
    applyEnabledState(changes.enabled.newValue);
  }
});

window.addEventListener('yt-navigate-start', cleanup);
const observer = new MutationObserver(() => onUrlChange());
observer.observe(document.querySelector('title'), { childList: true });
window.addEventListener('yt-navigate-finish', onUrlChange);

function getYTVideo() {
  return document.querySelector('video.html5-main-video') || document.querySelector('video');
}

function waitForYTPlayer() {
  return new Promise((resolve) => {
    const check = () => {
      const video = getYTVideo();
      if (video && video.readyState >= 1) {
        resolve(video);
      } else {
        setTimeout(check, 500);
      }
    };
    check();
  });
}

async function fetchTranscript(videoId) {
  if (!extensionEnabled) throw new Error('Overlay disabled');
  const url = `https://www.youtube.com/watch?v=${videoId}`;
  abortController = new AbortController();
  const res = await fetch(`${API_BASE}/extract_transcript`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
    signal: abortController.signal
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error || 'Failed to extract transcript');
  }
  return res.json();
}

async function streamASLChunks(chunks, gender = 'neutral', onClipReady, onAllClipsDone) {
  if (!extensionEnabled) throw new Error('Overlay disabled');
  abortController = new AbortController();
  const res = await fetch(`${API_BASE}/api/stream_youtube_chunks`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ chunks, gender }),
    signal: abortController.signal
  });

  if (!res.ok) throw new Error('Stream failed');
  if (!res.body) throw new Error('Streaming not supported');

  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const dataStr = line.replace('data: ', '').trim();
        if (!dataStr) continue;
        try {
          const data = JSON.parse(dataStr);
          if (data.status === 'chunk_ready') {
            const clip = {
              url: `${API_BASE}${data.url}`,
              text: data.text,
              start_time: data.start_time,
              end_time: data.end_time,
              index: data.chunk_index
            };
            onClipReady(clip);
          } else if (data.status === 'done') {
            if (onAllClipsDone) onAllClipsDone();
          }
        } catch (e) { console.error('SSE parse error:', e); }
      }
    }
  }
}

function initOverlay(videoId) {
  const old = document.getElementById('sv-asl-overlay');
  if (old) old.remove();

  const playerContainer = document.getElementById('movie_player') || document.querySelector('.html5-video-player');
  if (!playerContainer) return null;

  playerContainer.style.position = 'relative';

  const overlay = document.createElement('div');
  overlay.id = 'sv-asl-overlay';
  overlay.innerHTML = `
    <div class="sv-header">
      <span class="sv-title">🤟 Silentvoice</span>
      <div class="sv-controls">
        <button class="sv-btn sv-minimize" title="Minimize">−</button>
        <button class="sv-btn sv-maximize" title="Maximize">□</button>
        <button class="sv-btn sv-close" title="Hide">×</button>
      </div>
    </div>
    <div class="sv-video-container">
      <video class="sv-video" muted playsinline></video>
      <div class="sv-status">Ready</div>
      <button class="sv-generate-btn">Generate ASL</button>
    </div>
  `;

  playerContainer.appendChild(overlay);
  makeDraggable(overlay);
  
  // UI controls
  overlay.querySelector('.sv-minimize').onclick = () => overlay.classList.toggle('sv-minimized');
  overlay.querySelector('.sv-maximize').onclick = () => overlay.classList.toggle('sv-maximized');
  overlay.querySelector('.sv-close').onclick = () => overlay.classList.add('sv-hidden');
  
  // Generate button
  const genBtn = overlay.querySelector('.sv-generate-btn');
  genBtn.onclick = () => {
    if (!extensionEnabled) return;
    genBtn.style.display = 'none';
    startASLPipeline(videoId);
  };
  
  return overlay;
}

function updateStatus(msg) {
  const statusEl = document.querySelector('.sv-status');
  if (statusEl) {
    statusEl.textContent = msg;
    statusEl.style.display = msg ? 'block' : 'none';
  }
}

function setupSync(ytVideo) {
  const overlayVideo = document.querySelector('.sv-video');
  let currentClipIndex = -1;

  function findClipForTime(t) {
    for (let i = 0; i < aslClips.length; i++) {
      if (t >= aslClips[i].start_time && t < aslClips[i].end_time) {
        return i;
      }
    }
    return -1;
  }

  ytVideo.addEventListener('timeupdate', () => {
    const t = ytVideo.currentTime;
    const clipIdx = findClipForTime(t);

    if (clipIdx !== currentClipIndex) {
      currentClipIndex = clipIdx;
      if (clipIdx >= 0 && aslClips[clipIdx]) {
        overlayVideo.src = aslClips[clipIdx].url;
        overlayVideo.play().catch(e => console.log('Autoplay blocked:', e));
      } else {
        overlayVideo.pause();
      }
    }
  });

  ytVideo.addEventListener('pause', () => overlayVideo.pause());
  ytVideo.addEventListener('play', () => {
    if (currentClipIndex >= 0) overlayVideo.play().catch(e => console.log('Autoplay blocked:', e));
  });

  ytVideo.addEventListener('seeking', () => {
    const t = ytVideo.currentTime;
    const clipIdx = findClipForTime(t);
    currentClipIndex = clipIdx;
    if (clipIdx >= 0 && aslClips[clipIdx]) {
      overlayVideo.src = aslClips[clipIdx].url;
      overlayVideo.play().catch(e => console.log('Autoplay blocked:', e));
    }
  });
}

function makeDraggable(el) {
  const header = el.querySelector('.sv-header');
  let isDragging = false, offsetX, offsetY;

  header.addEventListener('mousedown', (e) => {
    isDragging = true;
    offsetX = e.clientX - el.getBoundingClientRect().left;
    offsetY = e.clientY - el.getBoundingClientRect().top;
    el.style.cursor = 'grabbing';
  });

  document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    el.style.left = (e.clientX - offsetX) + 'px';
    el.style.top = (e.clientY - offsetY) + 'px';
    el.style.right = 'auto';
    el.style.bottom = 'auto';
  });

  document.addEventListener('mouseup', () => {
    isDragging = false;
    el.style.cursor = '';
  });
}

async function startASLPipeline(videoId) {
  if (!extensionEnabled) return;
  const ytVideo = await waitForYTPlayer();
  const overlay = document.getElementById('sv-asl-overlay');
  if (!overlay) return;
  updateStatus('Extracting transcript...');
  aslClips = [];

  let transcriptData;
  try {
    transcriptData = await fetchTranscript(videoId);
  } catch (e) {
    updateStatus('❌ ' + e.message);
    return;
  }

  if (!transcriptData.chunks || transcriptData.chunks.length === 0) {
    updateStatus('No transcript available');
    return;
  }

  updateStatus(`Rendering ${transcriptData.chunks.length} chunks...`);

  chrome.storage.local.get(['gender'], async (result) => {
    if (!extensionEnabled) return;
    const gender = result.gender || 'neutral';
    
    try {
      await streamASLChunks(transcriptData.chunks, gender, (clip) => {
        if (!extensionEnabled) return;
        aslClips.push(clip);
        updateStatus('');
        if (aslClips.length === 1) {
          setupSync(ytVideo);
        }
      }, () => {
        // all done
      });
    } catch (e) {
      if (e.name !== 'AbortError') {
        updateStatus('❌ Rendering failed');
      }
    }
  });
}

(async () => {
  const settings = await getLocalSettings(['enabled']);
  applyEnabledState(settings.enabled);
  onUrlChange();
})();
