/**
 * Type definitions for the structured daily paper digest.
 * Loaded from src/articles/papers/<slug>/papers.json (not the content collection).
 */

export interface PaperItem {
  index: number;
  title: string;
  score: number;
  channel: string;
  category: string;
  date: string;
  authors: string;
  institutions: string;
  abstract: string;
  contributions: string[];
  conclusions: string[];
  experiments: string[];
  significance: string;
  takeaway: string;
  keyExperiment: string | null;
  arxivUrl: string;
  pdfUrl: string;
}

export interface ChannelGroup {
  channel: string;
  countText: string;
  papers: PaperItem[];
}

export interface ChannelMeta {
  name: string;
  countText: string;
}

export interface PaperDigest {
  date: string;
  title: string;
  lede: string;
  stats: { selected: number; channels: number; candidates: number };
  channels: ChannelMeta[];
  channelGroups: ChannelGroup[];
}

/** Load the structured papers.json for a given digest slug. */
export async function loadDigest(slug: string): Promise<PaperDigest | null> {
  const { readFileSync } = await import('node:fs');
  const { resolve } = await import('node:path');
  const path = resolve(process.cwd(), `src/articles/papers/${slug}/papers.json`);
  try {
    const raw = readFileSync(path, 'utf-8');
    return JSON.parse(raw) as PaperDigest;
  } catch {
    return null;
  }
}
