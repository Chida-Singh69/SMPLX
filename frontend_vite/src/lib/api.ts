export const API = '';

export async function fetchWords() {
  const r = await fetch(`${API}/api/available_words`);
  if (!r.ok) throw new Error('Failed to fetch words');
  return r.json();
}

export async function fetchSentences() {
  const r = await fetch(`${API}/api/list_sentences`);
  if (!r.ok) throw new Error('Failed to fetch sentences');
  return r.json();
}

export async function fetchPoses() {
  const r = await fetch(`${API}/api/list_poses`);
  if (!r.ok) throw new Error('Failed to fetch poses');
  return r.json();
}
