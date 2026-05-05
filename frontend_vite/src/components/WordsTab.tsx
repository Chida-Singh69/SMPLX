import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Play, Loader2, X } from 'lucide-react';
import { API, fetchWords } from '@/lib/api';
import type { Gender } from '../App';

export function WordsTab({ gender }: { gender: Gender }) {
  const [words, setWords] = useState<string[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [current, setCurrent] = useState('');
  const [status, setStatus] = useState<'idle' | 'generating' | 'done' | 'error'>('idle');
  const [videoUrl, setVideoUrl] = useState('');
  const [error, setError] = useState('');

  useEffect(() => { fetchWords().then(setWords).catch(() => setError('Could not load words from API')); }, []);
  const add = () => { if (current) { setSelected(p => [...p, current]); setCurrent(''); } };
  const generate = async () => {
    if (!selected.length) return;
    setStatus('generating');
    try {
      const r = await fetch(`${API}/asl_stream`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ words: selected, gender }),
      });
      if (!r.ok) throw new Error((await r.json()).error || 'Generation failed');
      setVideoUrl(URL.createObjectURL(await r.blob())); setStatus('done');
    } catch (e: any) { setError(e.message); setStatus('error'); }
  };

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
          <h3 className="font-semibold text-sm mb-5" style={{ color: 'rgba(255,255,255,0.90)' }}>Select Words</h3>
          {error && <p className="text-xs rounded-lg px-3 py-2 mb-4" style={{ color: '#ef4444', background: 'rgba(239,68,68,0.10)', border: '1px solid rgba(239,68,68,0.15)' }}>{error}</p>}
          <div className="flex gap-3 mb-5">
            <select value={current} onChange={e => setCurrent(e.target.value)}
              className="dark-select flex-1 text-sm px-4 py-2.5">
              <option value="" disabled>— choose a word —</option>
              {words.map(w => <option key={w} value={w}>{w}</option>)}
            </select>
            <button onClick={add} disabled={!current}
              className="btn-primary px-5 rounded-xl text-sm font-semibold">Add</button>
          </div>
          <div className="min-h-[80px] rounded-xl p-4 border border-dashed" style={{ borderColor: 'rgba(255,255,255,0.08)', background: 'rgba(255,255,255,0.02)' }}>
            {selected.length === 0
              ? <p className="text-xs text-center py-4" style={{ color: 'rgba(255,255,255,0.20)' }}>No words selected yet</p>
              : <div className="flex flex-wrap gap-2">
                  {selected.map((w, i) => (
                    <span key={i} className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-full"
                      style={{ background: 'rgba(0,212,170,0.10)', color: '#00d4aa', border: '1px solid rgba(0,212,170,0.20)' }}>
                      {w}
                      <X className="w-3 h-3 cursor-pointer opacity-60 hover:opacity-100" onClick={() => setSelected(p => p.filter((_, j) => j !== i))} />
                    </span>
                  ))}
                </div>
            }
          </div>
          {selected.length > 0 && <button className="mt-2 text-xs" style={{ color: 'rgba(255,255,255,0.20)' }} onClick={() => setSelected([])}>Clear all</button>}
        </div>
        <button disabled={!selected.length || status === 'generating'} onClick={generate}
          className="btn-primary w-full py-4 rounded-xl text-sm font-semibold flex items-center justify-center gap-2">
          {status === 'generating' ? <><Loader2 className="w-4 h-4 animate-spin" />Rendering…</> : <><Play className="w-4 h-4" />Generate Animation</>}
        </button>
      </div>

      <div className="shadow-sm flex flex-col overflow-hidden" style={{ ...card, minHeight: '380px' }}>
        {status === 'done' ? (
          <div className="flex flex-col h-full">
            <div className="px-6 py-4 flex items-center justify-between" style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
              <span className="text-sm font-semibold" style={{ color: 'rgba(255,255,255,0.90)' }}>Result</span>
              <a href={videoUrl} download="word_asl.mp4" className="text-xs px-3 py-1.5 rounded-lg font-medium" style={{ background: 'rgba(0,212,170,0.10)', color: '#00d4aa', border: '1px solid rgba(0,212,170,0.15)' }}>⬇ Download</a>
            </div>
            <div className="flex-1 p-6 flex flex-col gap-4">
              <p className="text-xs font-medium" style={{ color: 'rgba(255,255,255,0.50)' }}>{selected.join(' → ')}</p>
              <video src={videoUrl} controls autoPlay className="w-full rounded-xl" />
            </div>
          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center p-10 text-center">
            <div className="w-20 h-20 rounded-2xl mb-5 flex items-center justify-center" style={{ background: 'rgba(0,212,170,0.06)', border: '1px solid rgba(0,212,170,0.10)' }}>
              {status === 'generating' ? <Loader2 className="w-9 h-9 animate-spin" style={{ color: '#00d4aa' }} /> : <span className="text-3xl" style={{ color: 'rgba(255,255,255,0.15)' }}>✦</span>}
            </div>
            <p className="font-semibold text-sm mb-1.5" style={{ color: 'rgba(255,255,255,0.80)' }}>{status === 'generating' ? 'Generating…' : 'Word Animation'}</p>
            <p className="text-xs" style={{ color: 'rgba(255,255,255,0.35)' }}>{status === 'generating' ? 'Building poses and rendering your sequence.' : 'Select and queue words on the left, then hit Generate.'}</p>
          </div>
        )}
      </div>
    </div>
  );
}
