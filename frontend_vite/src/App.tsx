import { useState, useEffect, useRef, useCallback } from 'react';
import { YoutubeTab } from './components/YoutubeTab';
import { WordsTab } from './components/WordsTab';
import { SentencesTab } from './components/SentencesTab';
import { PosesTab } from './components/PosesTab';

export type Gender = 'neutral' | 'male' | 'female';
type Tab = 'home' | 'sentences' | 'words' | 'about' | 'user';

const NAV = [
  { id: 'home' as Tab, icon: '▶', label: 'Video' },
  { id: 'sentences' as Tab, icon: '❝', label: 'Sentences' },
  { id: 'words' as Tab, icon: '✦', label: 'Words' },
  { id: 'about' as Tab, icon: 'ℹ', label: 'About' },
  { id: 'user' as Tab, icon: '◎', label: 'Profile' },
];

/* ═══ Particle Network Canvas — migam.ai style plexus ═══ */
function ParticleCanvas() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext('2d'); if (!ctx) return;
    let raf: number, w: number, h: number;

    const mouse = { x: -9999, y: -9999 };
    const MOUSE_RADIUS = 200;      // attraction zone around cursor
    const MOUSE_STRENGTH = 0.08;   // how strongly particles are pulled
    const N = 100;                  // particle count
    const DIST = 200;              // max connection distance
    const BASE_SPEED = 0.6;        // base drift speed
    const pts: { x: number; y: number; vx: number; vy: number; ox: number; oy: number }[] = [];

    const resize = () => { w = c.width = window.innerWidth; h = c.height = window.innerHeight; };
    resize(); window.addEventListener('resize', resize);

    const onMouse = (e: MouseEvent) => { mouse.x = e.clientX; mouse.y = e.clientY; };
    const onLeave = () => { mouse.x = -9999; mouse.y = -9999; };
    window.addEventListener('mousemove', onMouse);
    window.addEventListener('mouseleave', onLeave);

    for (let i = 0; i < N; i++) {
      const x = Math.random() * w, y = Math.random() * h;
      pts.push({
        x, y, ox: x, oy: y,
        vx: (Math.random() - 0.5) * BASE_SPEED,
        vy: (Math.random() - 0.5) * BASE_SPEED,
      });
    }

    const draw = () => {
      ctx.clearRect(0, 0, w, h);

      for (const p of pts) {
        // Mouse attraction
        const mdx = mouse.x - p.x, mdy = mouse.y - p.y;
        const md = Math.sqrt(mdx * mdx + mdy * mdy);
        if (md < MOUSE_RADIUS && md > 1) {
          const force = (1 - md / MOUSE_RADIUS) * MOUSE_STRENGTH;
          p.vx += (mdx / md) * force;
          p.vy += (mdy / md) * force;
        }

        // Apply velocity with friction
        p.vx *= 0.98; p.vy *= 0.98;
        p.x += p.vx; p.y += p.vy;

        // Gentle drift — keep particles moving even without mouse
        p.vx += (Math.random() - 0.5) * 0.01;
        p.vy += (Math.random() - 0.5) * 0.01;

        // Bounce off edges
        if (p.x < 0) { p.x = 0; p.vx *= -0.5; }
        if (p.x > w) { p.x = w; p.vx *= -0.5; }
        if (p.y < 0) { p.y = 0; p.vy *= -0.5; }
        if (p.y > h) { p.y = h; p.vy *= -0.5; }

        // Dot brightness increases near mouse
        const brightness = md < MOUSE_RADIUS ? 0.5 + 0.5 * (1 - md / MOUSE_RADIUS) : 0.35;
        const dotSize = md < MOUSE_RADIUS ? 1.8 + 1.2 * (1 - md / MOUSE_RADIUS) : 1.5;
        ctx.beginPath(); ctx.arc(p.x, p.y, dotSize, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(0,212,170,${brightness})`; ctx.fill();
      }

      // Draw connections
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
        const dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < DIST) {
          // Lines near mouse are brighter
          const mx = (pts[i].x + pts[j].x) / 2, my = (pts[i].y + pts[j].y) / 2;
          const mDist = Math.sqrt((mouse.x - mx) ** 2 + (mouse.y - my) ** 2);
          const mouseBoost = mDist < MOUSE_RADIUS ? 0.18 * (1 - mDist / MOUSE_RADIUS) : 0;
          const alpha = (0.06 + mouseBoost) * (1 - d / DIST);

          ctx.beginPath(); ctx.moveTo(pts[i].x, pts[i].y); ctx.lineTo(pts[j].x, pts[j].y);
          ctx.strokeStyle = `rgba(0,212,170,${alpha})`; ctx.lineWidth = 0.6; ctx.stroke();
        }
      }

      // Draw lines from mouse to nearby particles
      if (mouse.x > 0 && mouse.y > 0) {
        for (const p of pts) {
          const dx = mouse.x - p.x, dy = mouse.y - p.y;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d < MOUSE_RADIUS * 0.8) {
            const alpha = 0.15 * (1 - d / (MOUSE_RADIUS * 0.8));
            ctx.beginPath(); ctx.moveTo(mouse.x, mouse.y); ctx.lineTo(p.x, p.y);
            ctx.strokeStyle = `rgba(0,212,170,${alpha})`; ctx.lineWidth = 0.5; ctx.stroke();
          }
        }
      }

      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', onMouse);
      window.removeEventListener('mouseleave', onLeave);
    };
  }, []);
  return <canvas ref={ref} id="particle-canvas" />;
}

/* ═══ Loading Screen ═══ */
function LoadingScreen({ onDone }: { onDone: () => void }) {
  const [fade, setFade] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setFade(true), 2200);
    const t2 = setTimeout(onDone, 3000);
    return () => { clearTimeout(t); clearTimeout(t2); };
  }, [onDone]);

  return (
    <div className={`loading-screen ${fade ? 'fade-out' : ''}`}>
      <div className="loading-logo" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 }}>
        <img src="/logo.png" alt="Silentvoice Logo" style={{ width: 96, height: 'auto', maxHeight: 96, objectFit: 'contain', marginBottom: 4 }} />
        <div style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <p style={{ fontSize: 34, fontWeight: 800, color: '#ffffff', margin: '0 0 4px', letterSpacing: '-0.5px' }}>Silentvoice</p>
          <p style={{ fontSize: 14, color: 'rgba(255,255,255,0.35)', margin: 0, letterSpacing: '0.15em', textTransform: 'uppercase', fontFamily: "'Outfit', sans-serif" }}>ASL Translation</p>
        </div>
      </div>
      <div className="loading-bar-track"><div className="loading-bar-fill" /></div>
    </div>
  );
}

/* ═══ Scroll Observer Hook ═══ */
function useScrollReveal() {
  const observe = useCallback((el: HTMLElement | null) => {
    if (!el) return;
    const obs = new IntersectionObserver(([e]) => { if (e.isIntersecting) { e.target.classList.add('visible'); obs.unobserve(e.target); } }, { threshold: 0.15 });
    el.querySelectorAll('.animate-on-scroll').forEach(c => obs.observe(c));
    return () => obs.disconnect();
  }, []);
  return observe;
}

/* ═══ Welcome Avatar Component ═══ */
function WelcomeAvatar() {
  const [visible, setVisible] = useState(true);
  const [showBubble, setShowBubble] = useState(true);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!visible) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    let raf: number;
    const render = () => {
      if (video.paused || video.ended) {
        raf = requestAnimationFrame(render);
        return;
      }

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const l = frame.data.length;

      for (let i = 0; i < l; i += 4) {
        const r = frame.data[i];
        const g = frame.data[i + 1];
        const b = frame.data[i + 2];
        
        // Improved Luma key with slight feathering
        const brightness = (r + g + b) / 3;
        if (brightness < 30) {
          // Smoothly fade out near-black pixels
          frame.data[i + 3] = brightness < 10 ? 0 : (brightness - 10) * 12.75;
        }
      }
      ctx.putImageData(frame, 0, 0);
      raf = requestAnimationFrame(render);
    };

    const handlePlay = () => {
      raf = requestAnimationFrame(render);
    };

    video.addEventListener('play', handlePlay);
    // Force play if needed
    video.play().catch(() => {
      // Autoplay might be blocked, wait for any click
      const resume = () => { video.play(); window.removeEventListener('click', resume); };
      window.addEventListener('click', resume);
    });

    return () => {
      cancelAnimationFrame(raf);
      video.removeEventListener('play', handlePlay);
    };
  }, [visible]);

  if (!visible) return null;

  return (
    <div className="welcome-avatar-container">
      <div className="welcome-avatar-video-wrap" style={{ position: 'relative', width: '100%', height: '100%' }}>
        <video 
          ref={videoRef}
          src="/assets/welcome_avatar.mp4"
          autoPlay 
          loop 
          muted 
          playsInline
          style={{ display: 'none' }}
        />
        <canvas 
          ref={canvasRef}
          width={640} 
          height={480}
          className="welcome-avatar-video"
        />
        
        {showBubble && (
          <div className="welcome-avatar-bubble" onClick={() => setShowBubble(false)} style={{ cursor: 'pointer' }}>
            Welcome to Silentvoice! 👋
          </div>
        )}
        
        <button 
          className="welcome-avatar-close" 
          style={{ top: 0, right: 0, opacity: 0.6, zIndex: 10 }} 
          onClick={() => setVisible(false)}
          title="Dismiss avatar"
        >
          ×
        </button>
      </div>
    </div>
  );
}

export interface Activity {
  action: string;
  detail: string;
  time: string;
  icon: string;
  vid?: string;
}

/* ═══ MAIN APP ═══ */
export default function App() {
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<Tab>('about');
  const [gender, setGender] = useState<Gender>('neutral');
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const [history, setHistory] = useState<Activity[]>(() => {
    const saved = localStorage.getItem('smplx_history');
    return saved ? JSON.parse(saved) : [];
  });
  const scrollRef = useScrollReveal();

  useEffect(() => {
    localStorage.setItem('smplx_history', JSON.stringify(history));
  }, [history]);

  const addActivity = (item: Omit<Activity, 'time'>) => {
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const newActivity = { ...item, time: timeStr };
    setHistory(prev => [newActivity, ...prev].slice(0, 8));
  };

  useEffect(() => {
    if (theme === 'light') {
      document.documentElement.classList.add('light-theme');
    } else {
      document.documentElement.classList.remove('light-theme');
    }
  }, [theme]);

  return (
    <>
      {loading && <LoadingScreen onDone={() => setLoading(false)} />}
      <WelcomeAvatar />

      <div style={{
        display: 'flex', flexDirection: 'column', minHeight: '100vh',
        fontFamily: "'Outfit', sans-serif",
        background: '#050505', color: 'rgba(255,255,255,0.92)',
        opacity: loading ? 0 : 1,
        transition: 'opacity 0.6s ease',
      }}>

        {/* Background layers */}
        <ParticleCanvas />
        <div className="gradient-mesh">
          <div className="orb orb-1" />
          <div className="orb orb-2" />
          <div className="orb orb-3" />
        </div>
        <div className="grid-overlay" />
        <div className="noise-overlay" />
        <div className="vignette-overlay" />

        {/* Floating geometric decorations */}
        <div aria-hidden style={{ position: 'fixed', inset: 0, zIndex: 0, pointerEvents: 'none' }}>
          <div className="anim-spin-slow" style={{ position: 'absolute', top: '8%', right: '6%', width: 200, height: 200, borderRadius: '50%', border: '1px solid rgba(0,212,170,0.08)' }} />
          <div className="anim-spin-rev" style={{ position: 'absolute', top: 'calc(8% + 25px)', right: 'calc(6% + 25px)', width: 150, height: 150, borderRadius: '50%', border: '1px solid rgba(0,212,170,0.05)' }} />
          <div className="anim-drift-x" style={{ position: 'absolute', bottom: '15%', left: '3%', width: 28, height: 28, border: '1px solid rgba(0,212,170,0.12)', transform: 'rotate(45deg)' }} />
        </div>

        {/* ── Header ── */}
        <header style={{
          position: 'sticky', top: 0, zIndex: 20,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '12px 32px',
          background: 'rgba(5,5,5,0.75)', backdropFilter: 'blur(20px)',
          borderBottom: '1px solid rgba(255,255,255,0.06)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <img src="/logo.png" alt="Silentvoice Logo" style={{ width: 52, height: 'auto', maxHeight: 52, objectFit: 'contain' }} />
            <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
              <p style={{ fontWeight: 800, fontSize: 22, color: '#ffffff', margin: '0 0 1px', letterSpacing: '-0.5px' }}>Silentvoice</p>
              <p style={{ fontSize: 12, color: 'rgba(255,255,255,0.35)', margin: 0 }}>ASL Animation Suite</p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: 5, alignItems: 'center' }}>
            <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.35)', marginRight: 4 }}>Avatar:</span>
            {(['neutral', 'male', 'female'] as Gender[]).map(g => (
              <button key={g} onClick={() => setGender(g)} style={{
                padding: '4px 13px', borderRadius: 20, fontSize: 11, fontWeight: 600,
                border: 'none', cursor: 'pointer', textTransform: 'capitalize', transition: 'all 0.18s',
                background: gender === g ? 'linear-gradient(135deg, #00d4aa, #00b894)' : 'rgba(255,255,255,0.04)',
                color: gender === g ? '#050505' : 'rgba(255,255,255,0.50)',
              }}>{g}</button>
            ))}

            <div style={{ width: 1, height: 16, background: 'rgba(255,255,255,0.1)', margin: '0 8px' }} />

            <button onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')} style={{
              background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.1)',
              color: 'rgba(255,255,255,0.8)', padding: '4px 12px', borderRadius: 20,
              cursor: 'pointer', fontSize: 11, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 6,
              transition: 'all 0.2s', fontFamily: "'Outfit', sans-serif"
            }}>
              {theme === 'dark' ? '☀️ Light' : '🌙 Dark'}
            </button>
          </div>

        </header>

        {/* ── Content ── */}
        <main ref={scrollRef} style={{ flex: 1, position: 'relative', zIndex: 1, padding: '24px 20px 110px' }}>
          {tab === 'home' && <YoutubeTab gender={gender} onActivity={addActivity} />}
          {tab === 'sentences' && <SentencesTab gender={gender} onActivity={addActivity} />}
          {tab === 'words' && <WordsTab gender={gender} onActivity={addActivity} />}
          {tab === 'about' && <AboutSection setTab={setTab} />}
          {tab === 'user' && <UserSection history={history} />}
        </main>

        {/* ── Bottom Nav ── */}
        <nav style={{
          position: 'fixed', bottom: 0, left: 0, right: 0, zIndex: 30,
          display: 'flex', justifyContent: 'center', padding: '10px 0 14px',
          background: 'rgba(5,5,5,0.85)', backdropFilter: 'blur(20px)',
          borderTop: '1px solid rgba(255,255,255,0.06)',
        }}>
          <div style={{
            display: 'flex', gap: 3, padding: '5px 7px', borderRadius: 30,
            background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)',
            boxShadow: '0 4px 30px rgba(0,0,0,0.5)',
          }}>
            {NAV.map(n => {
              const active = tab === n.id;
              return (
                <button key={n.id} onClick={() => setTab(n.id)}
                  className={!active ? 'nav-tooltip' : ''} data-tooltip={n.label}
                  style={{
                    display: 'flex', alignItems: 'center', gap: active ? 7 : 0,
                    padding: active ? '8px 20px' : '8px 15px',
                    borderRadius: 24, border: 'none', cursor: 'pointer',
                    background: active ? 'linear-gradient(135deg, #00d4aa, #00b894)' : 'transparent',
                    color: active ? '#050505' : 'rgba(255,255,255,0.40)',
                    fontWeight: 600, fontSize: 13,
                    transition: 'all 0.22s cubic-bezier(.4,0,.2,1)',
                    whiteSpace: 'nowrap', position: 'relative',
                    fontFamily: "'Outfit', sans-serif",
                  }}>
                  <span style={{ fontSize: 14, lineHeight: 1 }}>{n.icon}</span>
                  {active && <span style={{ fontSize: 12.5 }}>{n.label}</span>}
                </button>
              );
            })}
          </div>
        </nav>
      </div>
    </>
  );
}

/* ═══ ABOUT SECTION — migam.ai/product inspired ═══ */
function AboutSection({ setTab }: { setTab: (t: Tab) => void }) {
  const sectionRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = sectionRef.current; if (!el) return;
    const obs = new IntersectionObserver((entries) => {
      entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('visible'); obs.unobserve(e.target); } });
    }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });
    el.querySelectorAll('.animate-on-scroll').forEach(c => obs.observe(c));
    return () => obs.disconnect();
  }, []);

  const card: React.CSSProperties = {
    background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)',
    borderRadius: '1.25rem', padding: '32px',
    transition: 'all 0.35s cubic-bezier(0.4,0,0.2,1)',
  };

  const FEATURES = [
    { icon: '▶', title: 'YouTube → ASL', desc: 'Paste any YouTube URL. We extract the transcript and convert it into sign language animations automatically.' },
    { icon: '❝', title: 'Sentence Translation', desc: 'Pick from 10,000+ pre-mapped sentences in the How2Sign dataset for instant, accurate ASL rendering.' },
    { icon: '✦', title: 'Word Composition', desc: 'Build custom sign sequences word-by-word. Queue multiple words and generate a combined animation.' },
    { icon: '◈', title: 'Pose Assembly', desc: 'Load raw SMPL-X pose parameters from pickle files and assemble them into smooth MP4 animations.' },
    { icon: '⚡', title: 'Real-time Rendering', desc: 'GPU-accelerated SMPL-X body model rendering with studio lighting, skin textures, and hair.' },
    { icon: '⬇', title: 'Export & Download', desc: 'Every animation is rendered as a high-quality MP4 video, ready to download and share instantly.' },
  ];

  const STEPS = [
    { n: '01', title: 'Input', desc: 'Paste a YouTube URL, select a sentence from the dataset, or compose words manually.', accent: '#00d4aa' },
    { n: '02', title: 'AI Processing', desc: 'Transcript extraction, semantic matching with SentenceTransformers, and ASL gloss lookup from How2Sign.', accent: '#00e5b8' },
    { n: '03', title: 'Pose Generation', desc: 'Matched glosses are mapped to SMPL-X body parameters — 55 joints, 10 finger joints per hand.', accent: '#00f5c8' },
    { n: '04', title: 'Render & Export', desc: 'Frames are rendered with studio lighting and realistic skin, then encoded into a downloadable MP4.', accent: '#33ffd6' },
  ];

  return (
    <div ref={sectionRef} style={{ maxWidth: 1100, margin: '0 auto' }}>

      {/* ═══ HERO — 2-Column with Image ═══ */}
      <div className="animate-on-scroll" style={{ display: 'flex', alignItems: 'center', gap: 64, padding: '60px 0 100px', position: 'relative' }}>
        {/* glow orbs */}
        <div className="anim-pulse-glow" style={{ position: 'absolute', top: -40, left: 0, width: 400, height: 400, borderRadius: '50%', background: 'rgba(0,212,170,0.06)', filter: 'blur(100px)', pointerEvents: 'none' }} />

        <div style={{ position: 'relative', zIndex: 1, flex: 1 }}>
          <span style={{
            display: 'inline-block', fontSize: 12, letterSpacing: '0.25em', color: '#00d4aa',
            textTransform: 'uppercase', fontWeight: 700, marginBottom: 24,
            padding: '6px 16px', borderRadius: 20,
            background: 'rgba(0,212,170,0.10)', border: '1px solid rgba(0,212,170,0.20)',
          }}>Product Overview</span>

          <h1 style={{
            fontSize: 'clamp(40px, 5vw, 64px)', fontWeight: 700,
            color: '#ffffff', margin: '0 0 24px',
            letterSpacing: '-1.5px', lineHeight: 1.15, textShadow: '0 4px 20px rgba(0,0,0,0.5)',
          }}>
            English to ASL,<br />
            <span style={{ background: 'linear-gradient(135deg, #00d4aa, #00f5c8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Powered by AI</span>
          </h1>

          <p style={{
            fontSize: 18, color: 'rgba(255,255,255,0.7)', margin: '0 0 32px',
            maxWidth: 560, lineHeight: 1.6, fontWeight: 400,
          }}>
            Transform spoken English into photorealistic 3D sign language animations using SMPL-X body models and the How2Sign dataset — directly in your browser.
          </p>
        </div>

        <div style={{ flex: 1, position: 'relative' }}>
          <div style={{
            position: 'absolute', inset: -15, background: 'linear-gradient(135deg, rgba(0,212,170,0.2), transparent)',
            borderRadius: '2rem', filter: 'blur(20px)', zIndex: 0
          }} />
          <img
            src="/assets/hero_asl_avatar.png"
            alt="3D ASL Avatar"
            style={{
              width: '100%', height: 'auto', borderRadius: '1.5rem', position: 'relative', zIndex: 1,
              border: '1px solid rgba(255,255,255,0.1)', boxShadow: '0 20px 40px rgba(0,0,0,0.5)',
              objectFit: 'cover', aspectRatio: '4/3'
            }}
          />
        </div>
      </div>

      {/* ═══ FEATURE BENTO GRID ═══ */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 96 }}>
        {FEATURES.map((f, i) => (
          <div key={f.title} className="animate-on-scroll dark-card" style={{
            ...card, cursor: 'default',
            transitionDelay: `${i * 0.07}s`,
          }}>
            <div style={{
              width: 44, height: 44, borderRadius: 12, marginBottom: 20,
              background: 'rgba(0,212,170,0.08)', border: '1px solid rgba(0,212,170,0.12)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 18, color: '#00d4aa',
            }}>{f.icon}</div>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: 'rgba(255,255,255,0.92)', margin: '0 0 10px' }}>{f.title}</h3>
            <p style={{ fontSize: 13.5, color: 'rgba(255,255,255,0.40)', margin: 0, lineHeight: 1.7 }}>{f.desc}</p>
          </div>
        ))}
      </div>

      {/* ═══ HOW IT WORKS — PIPELINE ═══ */}
      <div style={{ marginBottom: 96 }}>
        <div className="animate-on-scroll" style={{ textAlign: 'center', marginBottom: 56 }}>
          <span style={{
            display: 'inline-block', fontSize: 11, letterSpacing: '0.25em', color: '#00d4aa',
            textTransform: 'uppercase', fontWeight: 700, marginBottom: 20,
            padding: '6px 16px', borderRadius: 20,
            background: 'rgba(0,212,170,0.08)', border: '1px solid rgba(0,212,170,0.15)',
          }}>How It Works</span>
          <h2 style={{ fontSize: 'clamp(28px, 3.5vw, 42px)', fontWeight: 700, color: 'rgba(255,255,255,0.92)', margin: 0, letterSpacing: '-1px' }}>
            From text to sign in four steps
          </h2>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 0, position: 'relative' }}>
          {/* connector line */}
          <div style={{ position: 'absolute', top: 28, left: '12.5%', right: '12.5%', height: 2, background: 'linear-gradient(90deg, rgba(0,212,170,0.15), rgba(0,212,170,0.30), rgba(0,212,170,0.15))', zIndex: 0 }} />

          {STEPS.map((s, i) => (
            <div key={s.n} className="animate-on-scroll" style={{ textAlign: 'center', position: 'relative', zIndex: 1, padding: '0 16px', transitionDelay: `${i * 0.12}s` }}>
              <div style={{
                width: 56, height: 56, borderRadius: '50%', margin: '0 auto 20px',
                background: '#0a0a0a', border: `2px solid ${s.accent}`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 16, fontWeight: 800, color: s.accent,
                boxShadow: `0 0 20px ${s.accent}22`,
              }}>{s.n}</div>
              <h4 style={{ fontSize: 15, fontWeight: 700, color: 'rgba(255,255,255,0.90)', margin: '0 0 8px' }}>{s.title}</h4>
              <p style={{ fontSize: 12.5, color: 'rgba(255,255,255,0.38)', margin: 0, lineHeight: 1.65 }}>{s.desc}</p>
            </div>
          ))}
        </div>

        {/* Panoramic Process Image */}
        <div className="animate-on-scroll" style={{ marginTop: 48, borderRadius: '1.5rem', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.08)', boxShadow: '0 10px 40px rgba(0,0,0,0.4)', position: 'relative', height: 280 }}>
          <div style={{ position: 'absolute', inset: 0, zIndex: 1, background: 'linear-gradient(to top, rgba(5,5,5,0.8), transparent)' }} />
          <img
            src="/assets/rendering_engine_mesh.png"
            alt="AI Rendering Mesh"
            style={{ width: '100%', height: '100%', objectFit: 'cover', opacity: 0.85 }}
          />
          <div style={{ position: 'absolute', bottom: 24, left: 32, zIndex: 2 }}>
            <p style={{ margin: 0, color: '#00d4aa', fontSize: 13, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase' }}>Rendering Engine</p>
            <p style={{ margin: '4px 0 0', color: 'rgba(255,255,255,0.9)', fontSize: 20, fontWeight: 600 }}>Real-time SMPL-X generation</p>
          </div>
        </div>
      </div>



      {/* ═══ TECH STACK ═══ */}
      <div className="animate-on-scroll" style={{ marginBottom: 96 }}>
        <div style={{ textAlign: 'center', marginBottom: 40 }}>
          <span style={{
            display: 'inline-block', fontSize: 11, letterSpacing: '0.25em', color: '#00d4aa',
            textTransform: 'uppercase', fontWeight: 700, marginBottom: 20,
            padding: '6px 16px', borderRadius: 20,
            background: 'rgba(0,212,170,0.08)', border: '1px solid rgba(0,212,170,0.15)',
          }}>Tech Stack</span>
          <h2 style={{ fontSize: 'clamp(24px, 3vw, 36px)', fontWeight: 700, color: 'rgba(255,255,255,0.92)', margin: 0, letterSpacing: '-0.5px' }}>
            Built with industry-leading tools
          </h2>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, justifyContent: 'center' }}>
          {[
            { name: 'SMPL-X', cat: 'core' },
            { name: 'How2Sign', cat: 'data' }, { name: 'Flask', cat: 'api' },
            { name: 'React', cat: 'ui' }, { name: 'Vite', cat: 'ui' }
            , { name: 'SentenceTransformers', cat: 'ai' },
            { name: 'FAISS', cat: 'api' }, { name: 'Trimesh', cat: 'core' },
            { name: 'Pyrender', cat: 'core' }, { name: 'PyTorch', cat: 'core' },
          ].map(t => (
            <span key={t.name} style={{
              fontSize: 13, fontWeight: 600, padding: '10px 20px', borderRadius: 12,
              background: t.cat === 'core' ? 'rgba(0,212,170,0.08)' : 'rgba(255,255,255,0.03)',
              color: t.cat === 'core' ? '#00d4aa' : 'rgba(255,255,255,0.60)',
              border: `1px solid ${t.cat === 'core' ? 'rgba(0,212,170,0.15)' : 'rgba(255,255,255,0.06)'}`,
              transition: 'all 0.2s',
              cursor: 'default',
            }}>{t.name}</span>
          ))}
        </div>
      </div>

      {/* ═══ CTA BANNER ═══ */}
      <div className="animate-on-scroll" style={{
        borderRadius: '1.5rem', overflow: 'hidden', position: 'relative',
        padding: '70px 48px', textAlign: 'center',
        border: '1px solid rgba(0,212,170,0.15)',
        boxShadow: '0 10px 40px rgba(0,0,0,0.5)',
      }}>
        <div style={{
          position: 'absolute', inset: 0, zIndex: 0,
          backgroundImage: 'url("/assets/asl_cta_background.png")',
          backgroundSize: 'cover', backgroundPosition: 'center',
          opacity: 0.35, filter: 'grayscale(50%) contrast(1.2)'
        }} />
        <div style={{ position: 'absolute', inset: 0, zIndex: 0, background: 'linear-gradient(180deg, rgba(5,5,5,0.5) 0%, rgba(5,5,5,0.9) 100%)' }} />

        <div className="anim-pulse-glow" style={{ position: 'absolute', top: -60, right: -60, width: 300, height: 300, borderRadius: '50%', background: '#00d4aa', filter: 'blur(120px)', opacity: 0.15, pointerEvents: 'none' }} />

        <div style={{ position: 'relative', zIndex: 1 }}>
          <h2 style={{ fontSize: 'clamp(28px, 3.5vw, 42px)', fontWeight: 800, color: '#ffffff', margin: '0 0 16px', letterSpacing: '-0.5px' }}>
            Ready to translate?
          </h2>
          <p style={{ fontSize: 16, color: 'rgba(255,255,255,0.7)', margin: '0 auto 32px', maxWidth: 440, lineHeight: 1.65 }}>
            Switch to the Video tab to get started — paste a YouTube URL and watch your first ASL animation come to life.
          </p>
          <div style={{ display: 'flex', gap: 12, justifyContent: 'center' }}>
            <button 
              className="btn-primary" 
              onClick={() => setTab('home')}
              style={{ padding: '14px 36px', fontSize: 15, borderRadius: 12, display: 'inline-flex', alignItems: 'center', gap: 8, fontWeight: 700, border: 'none' }}
            >
              ▶ Try It Now
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══ USER / PROFILE SECTION ═══ */
function UserSection({ history }: { history: Activity[] }) {
  const card: React.CSSProperties = {
    background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)',
    borderRadius: '1.25rem', padding: '28px',
    boxShadow: '0 2px 24px rgba(0,0,0,0.4)',
  };
  
  const stats = [
    { label: 'Total Jobs', value: history.length.toString() },
    { label: 'Tokens Processed', value: (history.length * 48).toString() },
  ];



  return (
    <div style={{ maxWidth: 720, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 24 }}>
      
      {/* Profile Header */}
      <div style={{ ...card, display: 'flex', alignItems: 'center', gap: 24, position: 'relative', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', top: -20, right: -20, width: 120, height: 120, borderRadius: '50%', background: 'rgba(0,212,170,0.05)', filter: 'blur(30px)' }} />
        
        <div style={{
          width: 90, height: 90, borderRadius: 24, flexShrink: 0,
          background: 'linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05))',
          border: '1px solid rgba(255,255,255,0.1)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 36,
        }}>
          👤
        </div>
        
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 4 }}>
            <h2 style={{ fontSize: 24, fontWeight: 800, color: '#ffffff', margin: 0 }}>Guest User</h2>
            <span style={{ fontSize: 10, background: 'rgba(0,212,170,0.15)', color: '#00d4aa', padding: '2px 8px', borderRadius: 12, fontWeight: 700, textTransform: 'uppercase' }}>Pro Plan</span>
          </div>
          <p style={{ fontSize: 14, color: 'rgba(255,255,255,0.45)', margin: '0 0 16px' }}>user_82931 · sign-language-pro@smplx.ai</p>
          
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn-ghost" style={{ padding: '6px 16px', fontSize: 12, borderRadius: 20 }}>Edit Profile</button>
            <button className="btn-ghost" style={{ padding: '6px 16px', fontSize: 12, borderRadius: 20 }}>Settings</button>
          </div>
        </div>
        

      </div>



      <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
        {/* Recent Activity */}
        <div style={card}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: '#ffffff', margin: 0 }}>Recent Activity</h3>
            <span style={{ fontSize: 12, color: '#00d4aa', cursor: 'pointer' }}>View All</span>
          </div>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {history.length === 0 ? (
              <div style={{ padding: '30px', textAlign: 'center', opacity: 0.3, border: '1px dashed rgba(255,255,255,0.1)', borderRadius: 16 }}>
                <p style={{ fontSize: 13 }}>No recent activity yet.<br />Try generating an animation!</p>
              </div>
            ) : (
              history.map((a, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 14, padding: '12px 14px', borderRadius: 16, background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)', transition: 'all 0.2s' }}>
                  <div style={{ 
                    width: 32, height: 32, borderRadius: 10, background: 'rgba(255,255,255,0.04)', 
                    display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 16 
                  }}>
                    {a.icon}
                  </div>
                  <div style={{ flex: 1 }}>
                    <p style={{ fontSize: 13.5, fontWeight: 600, color: 'rgba(255,255,255,0.9)', margin: 0 }}>{a.action}</p>
                    <p style={{ fontSize: 11.5, color: 'rgba(255,255,255,0.35)', margin: 0 }}>{a.detail}</p>
                  </div>
                  
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 6 }}>
                    {a.vid && (
                      <div style={{ 
                        width: 64, height: 40, borderRadius: 8, overflow: 'hidden', 
                        border: '1px solid rgba(0,212,170,0.2)', background: '#000',
                        boxShadow: '0 4px 12px rgba(0,0,0,0.3)', position: 'relative'
                      }}>
                        <video 
                          src={a.vid} 
                          autoPlay 
                          loop 
                          muted 
                          playsInline 
                          style={{ width: '100%', height: '100%', objectFit: 'cover', opacity: 0.8 }} 
                        />
                        <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(rgba(0,212,170,0.1), transparent)', pointerEvents: 'none' }} />
                      </div>
                    )}
                    <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.20)', fontWeight: 500 }}>{a.time}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Account Management */}
        <div style={{ ...card, border: '1px solid rgba(239,68,68,0.2)', background: 'rgba(239,68,68,0.02)' }}>
          <h3 style={{ fontSize: 16, fontWeight: 700, color: '#ef4444', marginBottom: 8 }}>Danger Zone</h3>
          <p style={{ fontSize: 13, color: 'rgba(255,255,255,0.4)', marginBottom: 20 }}>Manage your account session and local data.</p>
          
          <div style={{ display: 'flex', gap: 12 }}>
            <button 
              className="btn-ghost" 
              style={{ color: '#ef4444', borderColor: 'rgba(239,68,68,0.3)', padding: '10px 24px', borderRadius: 12 }}
              onClick={() => {
                localStorage.removeItem('smplx_history');
                window.location.reload();
              }}
            >
              Sign Out
            </button>
            <button 
              className="btn-ghost" 
              style={{ color: 'rgba(255,255,255,0.3)', padding: '10px 24px', borderRadius: 12 }}
              onClick={() => {
                if (confirm('Clear all translation history?')) {
                  localStorage.removeItem('smplx_history');
                  window.location.reload();
                }
              }}
            >
              Clear Cache
            </button>
          </div>
        </div>
      </div>

    </div>
  );
}
