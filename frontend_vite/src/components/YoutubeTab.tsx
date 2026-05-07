import { useState } from 'react';

import { Play, Hand, Loader2, ChevronRight } from 'lucide-react';
import { API } from '@/lib/api';
import LiteYouTubeEmbed from 'react-lite-youtube-embed';
import 'react-lite-youtube-embed/dist/LiteYouTubeEmbed.css';
import type { Gender } from '../App';

export function YoutubeTab({ gender, onActivity }: { gender: Gender, onActivity: (a: any) => void }) {
  const [url, setUrl] = useState('https://www.youtube.com/watch?v=dQw4w9WgXcQ');
  const [status, setStatus] = useState<'idle' | 'analyzing' | 'analyzed' | 'generating' | 'done' | 'error'>('idle');
  const [errorMsg, setError] = useState('');
  const [transcriptData, setTranscriptData] = useState<any>(null);
  const [playlist, setPlaylist] = useState<{url: string, text: string}[]>([]);
  const [currentVideoIndex, setCurrentVideoIndex] = useState(0);

  // Extract YouTube video ID from URL
  const getYoutubeId = (u: string) => {
    try {
      const match = u.match(/(?:v=|youtu\.be\/)([A-Za-z0-9_-]{11})/);
      return match?.[1] ?? null;
    } catch { return null; }
  };
  const youtubeId = getYoutubeId(url);

  const handleAnalyze = async () => {
    if (!url) return;
    setStatus('analyzing'); setError('');
    try {
      const r = await fetch(`${API}/extract_transcript`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url }),
      });
      if (!r.ok) throw new Error((await r.json()).error || 'Failed to extract transcript');
      setTranscriptData(await r.json()); setStatus('analyzed');
    } catch (e: any) { setError(e.message); setStatus('error'); }
  };

  const handleGenerate = async () => {
    if (!transcriptData) return;
    setStatus('generating');
    setPlaylist([]);
    setCurrentVideoIndex(0);
    setError('');
    
    onActivity({
      action: 'YouTube Video',
      detail: url.replace('https://', '').replace('www.', '').slice(0, 25) + '...',
      icon: '🎬'
    });

    try {
      const r = await fetch(`${API}/api/stream_youtube_chunks`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chunks: transcriptData.chunks, gender }),
      });
      if (!r.ok) throw new Error('Generation failed');
      if (!r.body) throw new Error('Streaming not supported');

      const reader = r.body.getReader();
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
                const fullUrl = `${API}${data.url}`;
                setPlaylist(prev => [...prev, { url: fullUrl, text: data.text, start: data.start_time, end: data.end_time }]);
                // Update history with the first video preview if not already set
                if (playlist.length === 0) {
                  onActivity({
                    action: 'YouTube Video',
                    detail: transcriptData.title || url.slice(0, 20),
                    icon: '🎬',
                    vid: fullUrl
                  });
                }
              } else if (data.status === 'done') {
                setStatus('done');
              }
            } catch (e) { console.error('Error parsing SSE:', e); }
          }
        }
      }
    } catch (e: any) { setError(e.message); setStatus('error'); }
  };

  const card: React.CSSProperties = {
    background: 'rgba(255,255,255,0.03)',
    border: '1px solid rgba(255,255,255,0.06)',
    borderRadius: '1.25rem',
    boxShadow: '0 2px 24px rgba(0,0,0,0.4)',
    transition: 'all 0.3s ease',
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 22 }}>

      {/* URL bar */}
      <div style={{ ...card, padding: '20px 24px', display: 'flex', gap: 12, alignItems: 'center' }}>
        <div style={{
          width: 28, height: 28, borderRadius: '50%', flexShrink: 0,
          background: 'linear-gradient(135deg, #00d4aa, #00b894)', color: '#050505',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 12, fontWeight: 800,
        }}>1</div>
        <input
          type="text"
          placeholder="Paste a YouTube URL — e.g. https://youtube.com/watch?v=..."
          value={url}
          onChange={e => setUrl(e.target.value)}
          disabled={status === 'analyzing' || status === 'generating'}
          className="dark-input"
          style={{ flex: 1, fontSize: 13.5, padding: '8px 16px' }}
        />
        <button
          onClick={handleAnalyze}
          disabled={!url || status === 'analyzing' || status === 'generating'}
          className="btn-primary"
          style={{ padding: '8px 22px', fontSize: 13, fontWeight: 600 }}
        >
          {status === 'analyzing' ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Extract'}
        </button>
        {errorMsg && <p style={{ color: '#ef4444', fontSize: 12, background: 'rgba(239,68,68,0.10)', borderRadius: 8, padding: '4px 10px', border: '1px solid rgba(239,68,68,0.20)' }}>{errorMsg}</p>}
      </div>

      {/* Two-panel: LEFT = YouTube player, RIGHT = ASL output */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>

        {/* LEFT — YouTube player */}
        <div style={{ ...card, minHeight: 420, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div style={{ padding: '14px 20px', borderBottom: '1px solid rgba(255,255,255,0.06)', display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: 13.5, fontWeight: 700, color: 'rgba(255,255,255,0.90)' }}>YouTube Video</span>
            {youtubeId && <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 10, background: 'rgba(0,212,170,0.10)', color: '#00d4aa', fontWeight: 600, border: '1px solid rgba(0,212,170,0.15)' }}>● Live</span>}
          </div>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
            {youtubeId ? (
              <div style={{ width: '100%', aspectRatio: '16/9' }}>
                <LiteYouTubeEmbed
                  id={youtubeId}
                  title="YouTube video"
                  playerClass="lty-playbtn"
                  wrapperClass="yt-lite"
                />
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: 40 }}>
                <div style={{
                  width: 72, height: 72, borderRadius: 18, margin: '0 auto 18px',
                  background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.06)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 28, color: 'rgba(255,255,255,0.15)',
                }}>▶</div>
                <p style={{ fontWeight: 700, fontSize: 14, color: 'rgba(255,255,255,0.80)', marginBottom: 6 }}>YouTube Player</p>
                <p style={{ fontSize: 12.5, color: 'rgba(255,255,255,0.35)', maxWidth: 220, margin: '0 auto', lineHeight: 1.65 }}>
                  Paste a valid YouTube URL above — the video will embed here automatically.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* RIGHT — ASL sign language output */}
        <div style={{ ...card, minHeight: 420, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div style={{ padding: '14px 20px', borderBottom: '1px solid rgba(255,255,255,0.06)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: 13.5, fontWeight: 700, color: 'rgba(255,255,255,0.90)' }}>Sign Language Translation</span>

          </div>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: status === 'done' ? '20px' : '40px', textAlign: 'center' }}>
            {(status === 'done' || playlist.length > 0) ? (
              <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: 10, alignItems: 'center' }}>
                {playlist[currentVideoIndex] && (
                  <>
                    <video 
                      src={playlist[currentVideoIndex].url} 
                      controls 
                      autoPlay 
                      onEnded={() => {
                        if (currentVideoIndex < playlist.length - 1) {
                          setCurrentVideoIndex(prev => prev + 1);
                        }
                      }}
                      style={{ width: '100%', borderRadius: 12, border: '1px solid rgba(0,212,170,0.2)' }} 
                    />
                    <div style={{ fontSize: 13, color: '#00d4aa', fontWeight: 600 }}>
                      Playing {currentVideoIndex + 1} of {playlist.length}
                    </div>
                  </>
                )}
                {status === 'generating' && (
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 12, color: 'rgba(255,255,255,0.4)' }}>
                    <Loader2 className="w-3 h-3 animate-spin" /> Rendering next chunk...
                  </div>
                )}
              </div>
            ) : (
              <>
                <div style={{
                  width: 72, height: 72, borderRadius: 18, marginBottom: 18,
                  background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.06)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  {status === 'generating'
                    ? <Loader2 style={{ width: 32, height: 32, color: '#00d4aa' }} className="animate-spin" />
                    : <Hand style={{ width: 32, height: 32, color: 'rgba(255,255,255,0.15)' }} />}
                </div>
                <p style={{ fontWeight: 700, fontSize: 14, color: 'rgba(255,255,255,0.80)', marginBottom: 6 }}>
                  {status === 'generating' ? 'Rendering animation…' : 'Output Preview'}
                </p>
                <p style={{ fontSize: 12.5, color: 'rgba(255,255,255,0.35)', maxWidth: 240, lineHeight: 1.65 }}>
                  {status === 'generating'
                    ? 'This may take a moment depending on video length.'
                    : 'Your ASL animation will appear here after generation.'}
                </p>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Step 2: transcript + generate */}
      {transcriptData && (
        <div style={{ ...card, padding: '22px 24px', display: 'flex', gap: 20, alignItems: 'flex-start' }}>
          <div style={{
            width: 28, height: 28, borderRadius: '50%', flexShrink: 0, marginTop: 2,
            background: 'linear-gradient(135deg, #00d4aa, #00b894)', color: '#050505',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 12, fontWeight: 800,
          }}>2</div>
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
              <p style={{ fontWeight: 700, fontSize: 13.5, color: 'rgba(255,255,255,0.90)', margin: 0 }}>Transcript Extracted</p>
              <span className="dark-tag" style={{ fontSize: 11 }}>
                {transcriptData.sentences?.length || 0} sentences
              </span>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 14 }}>
              {(transcriptData.sentences || []).slice(0, 6).map((s: string, i: number) => (
                <span key={i} style={{ fontSize: 12, padding: '4px 10px', borderRadius: 8, background: 'rgba(255,255,255,0.04)', color: 'rgba(255,255,255,0.50)', border: '1px solid rgba(255,255,255,0.06)' }}>
                  {s.length > 55 ? s.slice(0, 55) + '…' : s}
                </span>
              ))}
              {(transcriptData.sentences?.length || 0) > 6 && (
                <span style={{ fontSize: 12, padding: '4px 10px', borderRadius: 8, background: 'rgba(255,255,255,0.04)', color: 'rgba(255,255,255,0.50)', border: '1px solid rgba(255,255,255,0.06)' }}>
                  +{transcriptData.sentences.length - 6} more…
                </span>
              )}
            </div>
            <button
              onClick={handleGenerate}
              disabled={status === 'generating'}
              className="btn-primary"
              style={{ padding: '10px 28px', fontSize: 13.5, display: 'flex', alignItems: 'center', gap: 8 }}
            >
              {status === 'generating'
                ? <><Loader2 className="w-4 h-4 animate-spin" />Generating…</>
                : <><Play className="w-4 h-4" />Generate ASL Animation</>}
            </button>
          </div>
        </div>
      )}

      {/* How it works hint */}
      {status === 'idle' && (
        <div style={{ padding: '18px 22px', borderRadius: 16, background: 'rgba(0,212,170,0.04)', border: '1px solid rgba(0,212,170,0.10)' }}>
          <p style={{ fontWeight: 700, fontSize: 13, color: 'rgba(255,255,255,0.85)', marginBottom: 10 }}>How it works</p>
          <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
            {['Extract speech transcript', 'Semantic sentence matching', 'SMPL-X pose generation', 'Render & export MP4'].map((s, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                <ChevronRight style={{ width: 14, height: 14, color: '#00d4aa', flexShrink: 0 }} />
                <span style={{ fontSize: 12.5, color: 'rgba(255,255,255,0.40)' }}>{s}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
