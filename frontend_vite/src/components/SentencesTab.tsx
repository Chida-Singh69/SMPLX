import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Play, Loader2 } from 'lucide-react';
import { API, fetchSentences } from '@/lib/api';
import type { Gender } from '../App';

export function SentencesTab({ gender }: { gender: Gender }) {
  const [sentences, setSentences] = useState<any[]>([]);
  const [search, setSearch] = useState('');
  const [pkl, setPkl] = useState('');
  const [status, setStatus] = useState<'idle' | 'rendering' | 'done' | 'error'>('idle');
  const [videoUrl, setVideoUrl] = useState('');
  const [error, setError] = useState('');

  useEffect(() => { fetchSentences().then(setSentences).catch(() => setError('Could not load sentences from API')); }, []);

  const render = async () => {
    if (!pkl) return;
    setStatus('rendering');
    try {
      const r = await fetch(`${API}/api/render_sentence`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pkl, gender }),
      });
      if (!r.ok) throw new Error((await r.json()).error || 'Render failed');
      const data = await r.json();
      setVideoUrl(`${API}${data.url}`); setStatus('done');
    } catch (e: any) { setError(e.message); setStatus('error'); }
  };

  const filtered = search ? sentences.filter(s => s.text.toLowerCase().includes(search.toLowerCase())) : sentences;
  const selectedText = sentences.find(s => s.pkl === pkl)?.text || '';

  const card: React.CSSProperties = {
    background: 'rgba(255,255,255,0.03)',
    border: '1px solid rgba(255,255,255,0.06)',
    borderRadius: '1rem',
    boxShadow: '0 2px 24px rgba(0,0,0,0.4)',
  };

  return (
    <div className="grid grid-cols-2 gap-8">
      <div className="space-y-5">
        <div className="p-6 shadow-sm" style={card}>
          <h3 className="font-semibold text-sm mb-4" style={{ color: 'rgba(255,255,255,0.90)' }}>Choose a Sentence</h3>
          {error && <p className="text-xs rounded-lg px-3 py-2 mb-4" style={{ color: '#ef4444', background: 'rgba(239,68,68,0.10)', border: '1px solid rgba(239,68,68,0.15)' }}>{error}</p>}

          <input type="text" placeholder="Search sentences…" value={search}
            onChange={e => { setSearch(e.target.value); setPkl(''); }}
            className="dark-input w-full text-sm px-4 py-2.5 mb-3"
          />

          <select value={pkl} onChange={e => setPkl(e.target.value)}
            className="dark-select w-full text-sm px-4 py-2.5 mb-4">
            <option value="" disabled>— select a sentence —</option>
            {filtered.map(s => <option key={s.pkl} value={s.pkl}>{s.text}</option>)}
          </select>



          <div className="flex items-center justify-between text-xs" style={{ color: 'rgba(255,255,255,0.20)' }}>
            <span>How2Sign Dataset</span>
            <span>{filtered.length} / {sentences.length} sentences</span>
          </div>
        </div>

        <button disabled={!pkl || status === 'rendering'} onClick={render}
          className="btn-primary w-full py-4 rounded-xl text-sm font-semibold flex items-center justify-center gap-2">
          {status === 'rendering' ? <><Loader2 className="w-4 h-4 animate-spin" />Rendering…</> : <><Play className="w-4 h-4" />Render Animation</>}
        </button>
      </div>

      <div className="shadow-sm flex flex-col overflow-hidden" style={{ ...card, minHeight: '380px' }}>
        {status === 'done' ? (
          <div className="flex flex-col h-full">
            <div className="px-6 py-4 flex items-center justify-between" style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
              <span className="text-sm font-semibold" style={{ color: 'rgba(255,255,255,0.90)' }}>Result</span>
              <a href={videoUrl} download="sentence_asl.mp4" className="text-xs px-3 py-1.5 rounded-lg font-medium" style={{ background: 'rgba(0,212,170,0.10)', color: '#00d4aa', border: '1px solid rgba(0,212,170,0.15)' }}>⬇ Download</a>
            </div>
            <div className="flex-1 p-6 flex flex-col gap-4 items-center justify-center">
              <video src={videoUrl} controls autoPlay className="w-full rounded-xl" style={{ maxHeight: '400px', objectFit: 'contain' }} />
            </div>
          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center p-10 text-center">
            <div className="w-20 h-20 rounded-2xl mb-5 flex items-center justify-center" style={{ background: 'rgba(0,212,170,0.06)', border: '1px solid rgba(0,212,170,0.10)' }}>
              {status === 'rendering' ? <Loader2 className="w-9 h-9 animate-spin" style={{ color: '#00d4aa' }} /> : <span className="text-3xl" style={{ color: 'rgba(255,255,255,0.15)' }}>❝</span>}
            </div>
            <p className="font-semibold text-sm mb-1.5" style={{ color: 'rgba(255,255,255,0.80)' }}>{status === 'rendering' ? 'Rendering…' : 'Sentence Preview'}</p>
            <p className="text-xs" style={{ color: 'rgba(255,255,255,0.35)' }}>{status === 'rendering' ? 'Generating 3D poses from sentence data.' : 'Pick a sentence from the dropdown and click Render.'}</p>
          </div>
        )}
      </div>
    </div>
  );
}
