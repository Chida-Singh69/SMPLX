import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Play, Loader2 } from 'lucide-react';
import { API, fetchPoses } from '@/lib/api';
import type { Gender } from '../App';

export function PosesTab({ gender }: { gender: Gender }) {
  const [poses, setPoses] = useState<string[]>([]);
  const [selected, setSelected] = useState('');
  const [status, setStatus] = useState<'idle' | 'assembling' | 'done' | 'error'>('idle');
  const [videoUrl, setVideoUrl] = useState('');
  const [error, setError] = useState('');

  useEffect(() => { fetchPoses().then(setPoses).catch(() => setError('Could not load poses from API')); }, []);

  const assemble = async () => {
    if (!selected) return;
    setStatus('assembling');
    try {
      const r = await fetch(`${API}/api/render_poses`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder: selected, gender }),
      });
      if (!r.ok) throw new Error((await r.json()).error || 'Assembly failed');
      const data = await r.json();
      setVideoUrl(`${API}${data.url}`); setStatus('done');
    } catch (e: any) { setError(e.message); setStatus('error'); }
  };

  const card = { background: 'white', border: '1px solid #E2D5C2', borderRadius: '1rem' };

  return (
    <div className="grid grid-cols-2 gap-8">
      <div className="space-y-5">
        <div className="p-6 shadow-sm" style={card}>
          <h3 className="font-semibold text-sm mb-5" style={{ color: '#4A2C3F' }}>Select Pose Folder</h3>
          {error && <p className="text-xs text-red-700 bg-red-50 rounded-lg px-3 py-2 mb-4">{error}</p>}

          <select value={selected} onChange={e => setSelected(e.target.value)}
            className="w-full text-sm px-4 py-2.5 rounded-xl outline-none mb-6"
            style={{ background: '#FAEFE9', border: '1px solid #E2D5C2', color: '#4A2C3F' }}>
            <option value="" disabled>— select a pose folder —</option>
            {poses.map(p => <option key={p} value={p}>{p}</option>)}
          </select>

          {selected && (
            <div className="rounded-xl p-4 mb-5" style={{ background: 'rgba(244,163,132,0.12)', border: '1px solid rgba(244,163,132,0.3)' }}>
              <p className="text-xs font-medium mb-1" style={{ color: '#F4A384' }}>Selected folder</p>
              <p className="text-xs font-mono" style={{ color: '#4A2C3F' }}>{selected}</p>
            </div>
          )}

          <div className="flex items-center justify-between text-xs" style={{ color: '#E2D5C2' }}>
            <span>Frame-level PKL files</span>
            <span>{poses.length} folders</span>
          </div>
        </div>

        <Button disabled={!selected || status === 'assembling'} onClick={assemble}
          className="w-full text-white py-6 rounded-xl border-0 text-sm font-semibold" style={{ background: '#4A2C3F' }}>
          {status === 'assembling' ? <><Loader2 className="w-4 h-4 animate-spin mr-2" />Assembling…</> : <><Play className="w-4 h-4 mr-2" />Assemble Animation</>}
        </Button>
      </div>

      <div className="shadow-sm flex flex-col overflow-hidden" style={{ ...card, minHeight: '380px' }}>
        {status === 'done' ? (
          <div className="flex flex-col h-full">
            <div className="px-6 py-4 border-b flex items-center justify-between" style={{ borderColor: '#E2D5C2' }}>
              <span className="text-sm font-semibold" style={{ color: '#4A2C3F' }}>Result</span>
              <a href={videoUrl} download="pose_asl.mp4" className="text-xs px-3 py-1.5 rounded-lg font-medium" style={{ background: 'rgba(72,109,131,0.12)', color: '#486D83' }}>⬇ Download</a>
            </div>
            <div className="flex-1 p-6 flex flex-col gap-4">
              <p className="text-xs font-mono" style={{ color: '#7A5063' }}>{selected}</p>
              <video src={videoUrl} controls autoPlay className="w-full rounded-xl" />
            </div>
          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center p-10 text-center">
            <div className="w-20 h-20 rounded-2xl mb-5 flex items-center justify-center" style={{ background: 'rgba(244,163,132,0.15)' }}>
              {status === 'assembling' ? <Loader2 className="w-9 h-9 animate-spin" style={{ color: '#F4A384' }} /> : <span className="text-3xl" style={{ color: '#E2D5C2' }}>◈</span>}
            </div>
            <p className="font-semibold text-sm mb-1.5" style={{ color: '#4A2C3F' }}>{status === 'assembling' ? 'Assembling frames…' : 'Pose Preview'}</p>
            <p className="text-xs" style={{ color: '#7A5063' }}>{status === 'assembling' ? 'Combining frame-level PKL files into animation.' : 'Select a pose folder and click Assemble.'}</p>
          </div>
        )}
      </div>
    </div>
  );
}
