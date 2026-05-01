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
  const card = { background: 'white', border: '1px solid #E2D5C2', borderRadius: '1rem' };

  return (
    <div className="grid grid-cols-2 gap-8">
      <div className="space-y-5">
        <div className="p-6 shadow-sm" style={card}>
          <h3 className="font-semibold text-sm mb-4" style={{ color: '#4A2C3F' }}>Choose a Sentence</h3>
          {error && <p className="text-xs text-red-700 bg-red-50 rounded-lg px-3 py-2 mb-4">{error}</p>}

          <input type="text" placeholder="Search sentences…" value={search}
            onChange={e => { setSearch(e.target.value); setPkl(''); }}
            className="w-full text-sm px-4 py-2.5 rounded-xl outline-none mb-3"
            style={{ background: '#FAEFE9', border: '1px solid #E2D5C2', color: '#4A2C3F' }}
          />

          <select value={pkl} onChange={e => setPkl(e.target.value)}
            className="w-full text-sm px-4 py-2.5 rounded-xl outline-none mb-4"
            style={{ background: '#FAEFE9', border: '1px solid #E2D5C2', color: '#4A2C3F' }}>
            <option value="" disabled>— select a sentence —</option>
            {filtered.map(s => <option key={s.pkl} value={s.pkl}>{s.text}</option>)}
          </select>

          {pkl && (
            <div className="rounded-xl p-4 mb-4" style={{ background: 'rgba(244,163,132,0.12)', border: '1px solid rgba(244,163,132,0.3)' }}>
              <p className="text-xs font-medium mb-1" style={{ color: '#F4A384' }}>Selected</p>
              <p className="text-sm italic" style={{ color: '#4A2C3F' }}>"{selectedText}"</p>
            </div>
          )}

          <div className="flex items-center justify-between text-xs" style={{ color: '#E2D5C2' }}>
            <span>How2Sign Dataset</span>
            <span>{filtered.length} / {sentences.length} sentences</span>
          </div>
        </div>

        <Button disabled={!pkl || status === 'rendering'} onClick={render}
          className="w-full text-white py-6 rounded-xl border-0 text-sm font-semibold" style={{ background: '#4A2C3F' }}>
          {status === 'rendering' ? <><Loader2 className="w-4 h-4 animate-spin mr-2" />Rendering…</> : <><Play className="w-4 h-4 mr-2" />Render Animation</>}
        </Button>
      </div>

      <div className="shadow-sm flex flex-col overflow-hidden" style={{ ...card, minHeight: '380px' }}>
        {status === 'done' ? (
          <div className="flex flex-col h-full">
            <div className="px-6 py-4 border-b flex items-center justify-between" style={{ borderColor: '#E2D5C2' }}>
              <span className="text-sm font-semibold" style={{ color: '#4A2C3F' }}>Result</span>
              <a href={videoUrl} download="sentence_asl.mp4" className="text-xs px-3 py-1.5 rounded-lg font-medium" style={{ background: 'rgba(72,109,131,0.12)', color: '#486D83' }}>⬇ Download</a>
            </div>
            <div className="flex-1 p-6 flex flex-col gap-4">
              <p className="text-xs italic" style={{ color: '#7A5063' }}>"{selectedText}"</p>
              <video src={videoUrl} controls autoPlay className="w-full rounded-xl" />
            </div>
          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center p-10 text-center">
            <div className="w-20 h-20 rounded-2xl mb-5 flex items-center justify-center" style={{ background: 'rgba(244,163,132,0.15)' }}>
              {status === 'rendering' ? <Loader2 className="w-9 h-9 animate-spin" style={{ color: '#F4A384' }} /> : <span className="text-3xl" style={{ color: '#E2D5C2' }}>❝</span>}
            </div>
            <p className="font-semibold text-sm mb-1.5" style={{ color: '#4A2C3F' }}>{status === 'rendering' ? 'Rendering…' : 'Sentence Preview'}</p>
            <p className="text-xs" style={{ color: '#7A5063' }}>{status === 'rendering' ? 'Generating 3D poses from sentence data.' : 'Pick a sentence from the dropdown and click Render.'}</p>
          </div>
        )}
      </div>
    </div>
  );
}
