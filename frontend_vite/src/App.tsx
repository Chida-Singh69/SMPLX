import { useState } from 'react';
import { cn } from '@/lib/utils';
import { YoutubeTab } from './components/YoutubeTab';
import { WordsTab } from './components/WordsTab';
import { SentencesTab } from './components/SentencesTab';
import { PosesTab } from './components/PosesTab';

// ── Palette ──────────────────────────────────────
// #FAEFE9  Baby Blossom   → page bg, active nav bg
// #E2D5C2  Onion White    → card borders, subtle bg
// #7A5063  Grey Carmine   → secondary text, Add btn
// #F4A384  Creamy Peach   → badges, accents, highlights
// #4A2C3F  Obsidian Plum  → sidebar, primary btn, headings
// #486D83  Blue Loneliness→ info tags, download links
// ─────────────────────────────────────────────────

type Tab = 'youtube' | 'word' | 'sentences' | 'poses';
export type Gender = 'neutral' | 'male' | 'female';

const NAV_ITEMS: { id: Tab; label: string; icon: string; badge?: string }[] = [
  { id: 'youtube',   label: 'YouTube → ASL', icon: '▶', badge: 'LIVE' },
  { id: 'word',      label: 'Word Mode',      icon: '✦' },
  { id: 'sentences', label: 'Sentences',       icon: '❝' },
  { id: 'poses',     label: 'Poses',           icon: '◈' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('youtube');
  const [gender, setGender] = useState<Gender>('neutral');

  return (
    <div className="flex min-h-screen" style={{ fontFamily: '"Instrument Serif", serif', background: '#FAEFE9' }}>

      {/* ── Sidebar ── */}
      <aside className="w-64 flex-shrink-0 flex flex-col" style={{ background: '#4A2C3F' }}>
        {/* Logo */}
        <div className="px-6 py-7 border-b" style={{ borderColor: 'rgba(255,255,255,0.08)' }}>
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold" style={{ background: '#F4A384', color: '#4A2C3F' }}>
              S
            </div>
            <div>
              <p className="font-bold text-white text-sm tracking-wide">SMPL-X</p>
              <p className="text-xs" style={{ color: 'rgba(255,255,255,0.4)' }}>ASL Animation Suite</p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-5 space-y-1">
          <p className="px-3 pb-2 text-xs font-semibold uppercase tracking-widest" style={{ color: 'rgba(255,255,255,0.25)' }}>
            Modes
          </p>
          {NAV_ITEMS.map(item => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={cn(
                'w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium text-left transition-all duration-200',
                activeTab === item.id ? '' : 'text-white/50 hover:text-white hover:bg-white/5'
              )}
              style={activeTab === item.id ? { background: '#FAEFE9', color: '#4A2C3F' } : {}}
            >
              <span className="text-base w-5 text-center">{item.icon}</span>
              <span className="flex-1">{item.label}</span>
              {item.badge && (
                <span className="text-[10px] font-bold px-1.5 py-0.5 rounded" style={{ background: '#F4A384', color: '#4A2C3F' }}>
                  {item.badge}
                </span>
              )}
            </button>
          ))}
        </nav>

        {/* Gender selector */}
        <div className="px-5 py-5 border-t" style={{ borderColor: 'rgba(255,255,255,0.08)' }}>
          <p className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: 'rgba(255,255,255,0.25)' }}>
            Avatar Gender
          </p>
          <div className="flex gap-1.5">
            {(['neutral', 'male', 'female'] as Gender[]).map(g => (
              <button
                key={g}
                onClick={() => setGender(g)}
                className="flex-1 py-1.5 rounded-lg text-xs font-medium capitalize transition-all"
                style={gender === g
                  ? { background: '#F4A384', color: '#4A2C3F' }
                  : { background: 'rgba(255,255,255,0.07)', color: 'rgba(255,255,255,0.4)' }
                }
              >
                {g}
              </button>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="px-5 py-4 border-t" style={{ borderColor: 'rgba(255,255,255,0.08)' }}>
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold" style={{ background: '#7A5063', color: 'white' }}>
              {gender[0].toUpperCase()}
            </div>
            <div>
              <p className="text-white text-xs font-medium capitalize">Avatar: {gender}</p>
              <div className="flex items-center gap-1 mt-0.5">
                <div className="w-1.5 h-1.5 rounded-full bg-green-400" />
                <p className="text-[10px]" style={{ color: 'rgba(255,255,255,0.3)' }}>Model ready</p>
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* ── Main ── */}
      <div className="flex-1 flex flex-col overflow-hidden relative" style={{ background: '#FAEFE9' }}>

        {/* Subtle background — 3 soft orbs only, no pattern */}
        <div className="pointer-events-none absolute inset-0" aria-hidden="true" style={{ zIndex: 0 }}>
          <div style={{ position: 'absolute', top: '-120px', left: '-100px',  width: '500px', height: '500px', borderRadius: '50%', background: '#F4A384', filter: 'blur(120px)', opacity: 0.28 }} />
          <div style={{ position: 'absolute', bottom: '-100px', right: '-80px', width: '450px', height: '450px', borderRadius: '50%', background: '#486D83', filter: 'blur(130px)', opacity: 0.20 }} />
          <div style={{ position: 'absolute', top: '35%', left: '40%',        width: '380px', height: '380px', borderRadius: '50%', background: '#7A5063', filter: 'blur(110px)', opacity: 0.15 }} />
        </div>

        {/* Top bar */}
        <header className="flex items-center justify-between px-10 py-4 border-b bg-white/60 backdrop-blur-sm" style={{ borderColor: '#E2D5C2' }}>
          <div>
            <h1 className="text-lg font-bold" style={{ color: '#4A2C3F' }}>
              {NAV_ITEMS.find(n => n.id === activeTab)?.label}
            </h1>
            <p className="text-xs mt-0.5" style={{ color: '#7A5063' }}>
              {activeTab === 'youtube'   && 'Extract transcript and generate ASL animations from any YouTube video'}
              {activeTab === 'word'      && 'Select individual words and compose multi-word ASL sequences'}
              {activeTab === 'sentences' && 'Browse and render full ASL sentences from the How2Sign dataset'}
              {activeTab === 'poses'     && 'Assemble frame-level pose folders into smooth animations'}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs px-3 py-1.5 rounded-full font-medium" style={{ background: 'rgba(72,109,131,0.12)', color: '#486D83', border: '1px solid rgba(72,109,131,0.25)' }}>
              How2Sign Dataset
            </span>
            <span className="text-xs px-3 py-1.5 rounded-full font-medium capitalize" style={{ background: 'rgba(244,163,132,0.15)', color: '#7A5063', border: '1px solid rgba(244,163,132,0.3)' }}>
              Avatar: {gender}
            </span>
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-y-auto px-10 py-8" style={{ background: 'transparent', position: 'relative', zIndex: 1 }}>
          {activeTab === 'youtube'   && <YoutubeTab gender={gender} />}
          {activeTab === 'word'      && <WordsTab gender={gender} />}
          {activeTab === 'sentences' && <SentencesTab gender={gender} />}
          {activeTab === 'poses'     && <PosesTab gender={gender} />}
        </main>
      </div>
    </div>
  );
}
