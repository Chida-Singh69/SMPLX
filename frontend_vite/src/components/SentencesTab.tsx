import { useState, useEffect } from 'react';
// import { Button } from '@/components/ui/button';
import { Play, Loader2 } from 'lucide-react';
import { API, fetchSentences } from '@/lib/api';
import type { Gender } from '../App';

export function SentencesTab({ gender, onActivity }: { gender: Gender, onActivity: (a: any) => void }) {
  const [sentences, setSentences] = useState<any[]>([]);
  const [search, setSearch] = useState('');
  const [pkl, setPkl] = useState('');
  const [status, setStatus] = useState<'idle' | 'rendering' | 'done' | 'error'>('idle');
  const [videoUrl, setVideoUrl] = useState('');
  const [error, setError] = useState('');
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 50;

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
      const fullUrl = `${API}${data.url}`;
      setVideoUrl(fullUrl); 
      setStatus('done');
      onActivity({
        action: 'Sentence Rendered',
        detail: selectedText,
        icon: '❝',
        vid: fullUrl
      });
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
            onChange={e => { setSearch(e.target.value); setPkl(''); setPage(1); }}
            className="dark-input w-full text-sm px-4 py-2.5 mb-3"
          />

          <div className="flex justify-between items-center mb-2 px-1">
            <button disabled={page === 1} onClick={() => setPage(p => Math.max(1, p - 1))} className="text-xs px-2 py-1 rounded bg-white/5 hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed">Prev</button>
            <span className="text-xs text-white/40">Page {page} of {Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))}</span>
            <button disabled={page >= Math.ceil(filtered.length / PAGE_SIZE)} onClick={() => setPage(p => p + 1)} className="text-xs px-2 py-1 rounded bg-white/5 hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed">Next</button>
          </div>

          <div className="flex flex-col gap-1.5 mb-4 max-h-[220px] overflow-y-auto pr-2 custom-scrollbar">
            {filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE).map(s => (
              <div 
                key={s.pkl} 
                onClick={() => setPkl(s.pkl)}
                className={`text-sm px-3 py-2 rounded-lg cursor-pointer transition-all ${pkl === s.pkl ? 'bg-[#00d4aa]/20 border border-[#00d4aa]/40 text-white' : 'bg-white/5 border border-transparent text-white/70 hover:bg-white/10 hover:text-white'}`}
                style={{ wordWrap: 'break-word', whiteSpace: 'normal', lineHeight: 1.4 }}
              >
                {s.text}
              </div>
            ))}
            {filtered.length === 0 && <div className="text-xs text-white/30 text-center py-4">No sentences found.</div>}
          </div>

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
