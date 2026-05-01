import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Play, Hand, Loader2, ChevronRight } from 'lucide-react';
import { API } from '@/lib/api';
import type { Gender } from '../App';

export function YoutubeTab({ gender }: { gender: Gender }) {
  const [url, setUrl] = useState('');
  const [status, setStatus] = useState<'idle' | 'analyzing' | 'analyzed' | 'generating' | 'done' | 'error'>('idle');
  const [errorMsg, setError] = useState('');
  const [transcriptData, setTranscriptData] = useState<any>(null);
  const [resultVideoUrl, setResultVideoUrl] = useState('');

  const handleAnalyze = async () => {
    if (!url) return;
    setStatus('analyzing'); setError('');
    try {
      const r = await fetch(`${API}/extract_transcript`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ youtube_url: url }),
      });
      if (!r.ok) throw new Error((await r.json()).error || 'Failed to extract transcript');
      setTranscriptData(await r.json()); setStatus('analyzed');
    } catch (e: any) { setError(e.message); setStatus('error'); }
  };

  const handleGenerate = async () => {
    if (!transcriptData) return;
    setStatus('generating');
    try {
      const r = await fetch(`${API}/asl_from_youtube_sentences`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sentences: transcriptData.sentences, youtube_video_id: transcriptData.video_id, gender }),
      });
      if (!r.ok) throw new Error((await r.json()).error || 'Generation failed');
      setResultVideoUrl(URL.createObjectURL(await r.blob())); setStatus('done');
    } catch (e: any) { setError(e.message); setStatus('error'); }
  };

  const card = { background: 'white', border: '1px solid #E2D5C2', borderRadius: '1rem' };

  return (
    <div className="grid grid-cols-2 gap-8 h-full">
      <div className="space-y-5">
        <div className="p-6 shadow-sm" style={card}>
          <div className="flex items-center gap-3 mb-5">
            <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold" style={{ background: '#F4A384', color: '#4A2C3F' }}>1</div>
            <h3 className="font-semibold text-sm" style={{ color: '#4A2C3F' }}>Paste a YouTube URL</h3>
          </div>
          <div className="flex gap-3">
            <input type="text" placeholder="https://youtube.com/watch?v=..." value={url}
              onChange={e => setUrl(e.target.value)}
              disabled={status === 'analyzing' || status === 'generating'}
              className="flex-1 text-sm px-4 py-2.5 rounded-xl outline-none transition-all"
              style={{ background: '#FAEFE9', border: '1px solid #E2D5C2', color: '#4A2C3F' }}
            />
            <Button onClick={handleAnalyze} disabled={!url || status === 'analyzing' || status === 'generating'}
              className="text-white text-sm px-5 rounded-xl border-0" style={{ background: '#7A5063' }}>
              {status === 'analyzing' ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Extract'}
            </Button>
          </div>
          {errorMsg && <p className="mt-3 text-xs text-red-700 bg-red-50 rounded-lg px-3 py-2">{errorMsg}</p>}
        </div>

        {transcriptData && (
          <div className="p-6 shadow-sm" style={card}>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold" style={{ background: '#F4A384', color: '#4A2C3F' }}>2</div>
              <h3 className="font-semibold text-sm" style={{ color: '#4A2C3F' }}>Transcript Extracted</h3>
              <span className="ml-auto text-xs px-2 py-0.5 rounded-full font-medium" style={{ background: 'rgba(72,109,131,0.12)', color: '#486D83' }}>
                {transcriptData.sentences?.length || 0} sentences
              </span>
            </div>
            <div className="max-h-40 overflow-y-auto space-y-1 mb-5">
              {(transcriptData.sentences || []).slice(0, 8).map((s: string, i: number) => (
                <div key={i} className="text-xs px-3 py-2 rounded-lg" style={{ background: '#FAEFE9', color: '#7A5063' }}>{s}</div>
              ))}
              {(transcriptData.sentences?.length || 0) > 8 && (
                <p className="text-xs text-center py-1" style={{ color: '#7A5063' }}>+{transcriptData.sentences.length - 8} more…</p>
              )}
            </div>
            <Button onClick={handleGenerate} disabled={status === 'generating'}
              className="w-full text-white text-sm py-5 rounded-xl border-0" style={{ background: '#4A2C3F' }}>
              {status === 'generating' ? <><Loader2 className="w-4 h-4 animate-spin mr-2" />Generating…</> : <><Play className="w-4 h-4 mr-2" />Generate ASL Animation</>}
            </Button>
          </div>
        )}

        {status === 'idle' && (
          <div className="p-6 rounded-2xl" style={{ background: 'rgba(244,163,132,0.12)', border: '1px solid rgba(244,163,132,0.3)' }}>
            <p className="font-semibold text-sm mb-4" style={{ color: '#4A2C3F' }}>How it works</p>
            {['Extract speech transcript', 'Semantic sentence matching', 'SMPL-X pose generation', 'Render & export MP4'].map((s, i) => (
              <div key={i} className="flex items-center gap-3 py-1.5">
                <ChevronRight className="w-3.5 h-3.5 shrink-0" style={{ color: '#F4A384' }} />
                <span className="text-xs" style={{ color: '#7A5063' }}>{s}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="shadow-sm flex flex-col overflow-hidden" style={{ ...card, minHeight: '420px' }}>
        {status === 'done' ? (
          <div className="flex flex-col h-full">
            <div className="px-6 py-4 border-b flex items-center justify-between" style={{ borderColor: '#E2D5C2' }}>
              <span className="text-sm font-semibold" style={{ color: '#4A2C3F' }}>Animation Output</span>
              <a href={resultVideoUrl} download="youtube_asl.mp4" className="text-xs px-3 py-1.5 rounded-lg font-medium" style={{ background: 'rgba(72,109,131,0.12)', color: '#486D83' }}>⬇ Download</a>
            </div>
            <div className="flex-1 flex items-center justify-center p-6">
              <video src={resultVideoUrl} controls autoPlay className="w-full rounded-xl" />
            </div>
          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center p-10 text-center">
            <div className="w-20 h-20 rounded-2xl mb-5 flex items-center justify-center" style={{ background: 'rgba(244,163,132,0.15)' }}>
              {status === 'analyzing' || status === 'generating'
                ? <Loader2 className="w-9 h-9 animate-spin" style={{ color: '#F4A384' }} />
                : <Hand className="w-9 h-9" style={{ color: '#E2D5C2' }} />}
            </div>
            <p className="font-semibold text-sm mb-1.5" style={{ color: '#4A2C3F' }}>
              {status === 'analyzing' ? 'Extracting transcript…' : status === 'generating' ? 'Rendering animation…' : 'Output Preview'}
            </p>
            <p className="text-xs max-w-xs leading-relaxed" style={{ color: '#7A5063' }}>
              {status === 'analyzing' || status === 'generating' ? 'This may take a moment depending on video length.' : 'Enter a YouTube URL on the left to get started.'}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
