/**
 * content.ts — unified read API over all Content Collections.
 *
 * Replaces the old `src/data/blogPosts.ts` and the inline arrays that were
 * duplicated across index.astro / blog/index.astro / papers/index.astro.
 *
 * All homepage / archive / topic / RSS / sitemap reads go through here.
 * The actual article HTML still lives in public/blog/<slug>/index.html and
 * is served as-is — the collection stores metadata only.
 */
import { getCollection, type CollectionEntry } from 'astro:content';

export type Post = CollectionEntry<'posts'>;
export type Paper = CollectionEntry<'papers'>;
export type Interview = CollectionEntry<'interviews'>;
export type Travel = CollectionEntry<'travel'>;
export type Book = CollectionEntry<'books'>;

/** Resolve a post's on-site URL. Preserves existing /blog/<slug>/ shape. */
export function postHref(post: Post): string {
  return post.data.externalHref ?? `/blog/${post.data.slug}/`;
}

/** Resolve a paper's URL. */
export function paperHref(p: Paper): string {
  return `/papers/${p.data.slug}/`;
}

/** Resolve an interview's URL. */
export function interviewHref(i: Interview): string {
  return `/blog/${i.data.slug}/`;
}

/** Resolve a travel guide's URL. */
export function travelHref(t: Travel): string {
  return `/travel/${t.data.slug}.html`;
}

/** All published posts (drafts excluded), newest first. */
export async function getPublishedPosts(): Promise<Post[]> {
  const all = await getCollection('posts', (p) => !p.data.draft);
  return all.sort((a, b) => (a.data.publishedDate < b.data.publishedDate ? 1 : -1));
}

/** Featured posts only. */
export async function getFeaturedPosts(): Promise<Post[]> {
  const all = await getPublishedPosts();
  return all.filter((p) => p.data.featured);
}

/** Latest N posts, excluding featured (which appear in their own section). */
export async function getLatestPosts(n = 6, excludeFeatured = true): Promise<Post[]> {
  const all = await getPublishedPosts();
  return (excludeFeatured ? all.filter((p) => !p.data.featured) : all).slice(0, n);
}

/** All papers, newest first. */
export async function getPapers(): Promise<Paper[]> {
  const all = await getCollection('papers', (p) => !p.data.draft);
  return all.sort((a, b) => (a.data.date < b.data.date ? 1 : -1));
}

/** Latest paper issue. */
export async function getLatestPaper(): Promise<Paper | undefined> {
  return (await getPapers())[0];
}

/** All interviews. */
export async function getInterviews(): Promise<Interview[]> {
  const all = await getCollection('interviews', (i) => !i.data.draft);
  return all.sort((a, b) => (a.data.publishedDate < b.data.publishedDate ? 1 : -1));
}

/** All travel guides. */
export async function getTravelGuides(): Promise<Travel[]> {
  const all = await getCollection('travel', (t) => !t.data.draft);
  return all.sort((a, b) => (a.data.publishedDate < b.data.publishedDate ? 1 : -1));
}

/** All books. */
export async function getBooks(): Promise<Book[]> {
  return getCollection('books');
}

/** Aggregate all tags across posts with counts, sorted by count desc. */
export async function getAllTags(): Promise<{ tag: string; count: number }[]> {
  const posts = await getPublishedPosts();
  const counts = new Map<string, number>();
  for (const p of posts) {
    for (const t of p.data.tags) counts.set(t, (counts.get(t) ?? 0) + 1);
  }
  return [...counts.entries()].map(([tag, count]) => ({ tag, count })).sort((a, b) => b.count - a.count);
}

/** Posts grouped by year, newest year first. */
export async function getPostsByYear(): Promise<{ year: string; posts: Post[] }[]> {
  const posts = await getPublishedPosts();
  const byYear = new Map<string, Post[]>();
  for (const p of posts) {
    const y = p.data.publishedDate.slice(0, 4);
    if (!byYear.has(y)) byYear.set(y, []);
    byYear.get(y)!.push(p);
  }
  return [...byYear.entries()].map(([year, ps]) => ({ year, posts: ps })).sort((a, b) => (a.year < b.year ? 1 : -1));
}

/**
 * Topic registry — drives /topics/ pages.
 * primaryTopic classification: posts appear in at most one topic's main list.
 * paperTags: filter related papers. bookSlug: auto-link Starlight chapters.
 */
export interface Topic {
  slug: string;
  label: string;
  description: string;
  matchTags: string[];
  matchCategory?: string;
  paperTags?: string[];
  bookSlug?: string;
}

export const TOPICS: Topic[] = [
  { slug: 'llm-serving', label: 'LLM Serving', description: '推理引擎架构、调度、KV cache 与生产部署。', matchTags: ['LLM Serving', 'Scheduling'], matchCategory: 'llm-serving', paperTags: ['llm-serving', 'scheduling', 'inference', 'serving'] },
  { slug: 'speculative-decoding', label: 'Speculative Decoding', description: '投机解码：树起草、扩散 draft、验证与吞吐。', matchTags: ['Speculative Decoding'], matchCategory: 'speculative-decoding', paperTags: ['speculative-decoding', 'speculative', 'draft', 'verification'] },
  { slug: 'kv-cache', label: 'KV Cache', description: 'KV cache 管理、offload、分页与压缩池化。', matchTags: ['KV Cache'], matchCategory: 'kv-cache', paperTags: ['kv-cache', 'kv', 'cache', 'memory-management', 'pagedattention'] },
  { slug: 'gpu', label: 'GPU', description: 'GPU 执行模型、Tensor Core、TMA、kernel 优化。', matchTags: ['GPU', 'Megakernel'], matchCategory: 'gpu', paperTags: ['gpu', 'cuda', 'tensor-core', 'tma', 'kernel', 'flash-attention'], bookSlug: 'modern-gpu-programming-for-mlsys' },
  { slug: 'moe', label: 'MoE', description: '混合专家：路由、负载均衡、可实现性。', matchTags: ['MoE'], matchCategory: 'moe', paperTags: ['moe', 'mixture-of-experts', 'expert', 'routing'] },
  { slug: 'inference-engine', label: 'Inference Engine', description: '端到端推理引擎源码与架构对比。', matchTags: ['Inference Engine', 'RTP-LLM', 'vLLM'], matchCategory: 'inference-engine', paperTags: ['inference-engine', 'vllm', 'rtp-llm', 'engine'] },
];

/** Posts for a topic — classified by primaryTopic (falls back to category). */
export async function getPostsForTopic(topic: Topic): Promise<Post[]> {
  const posts = await getPublishedPosts();
  return posts.filter((p) =>
    (p.data.primaryTopic && p.data.primaryTopic === topic.slug) ||
    (!p.data.primaryTopic && p.data.category === topic.matchCategory)
  );
}

/** Papers related to a topic — filtered by paperTags against description/channels/highlights. */
export async function getPapersForTopic(topic: Topic, limit = 3): Promise<Paper[]> {
  if (!topic.paperTags || topic.paperTags.length === 0) return [];
  const papers = await getPapers();
  const tags = topic.paperTags.map((t) => t.toLowerCase());
  const scored = papers.map((p) => {
    const hay = (
      p.data.description + ' ' +
      p.data.channels.join(' ') + ' ' +
      p.data.highlights.map((h) => h.name + ' ' + (h.note || '')).join(' ')
    ).toLowerCase();
    const score = tags.reduce((s, t) => s + (hay.includes(t) ? 1 : 0), 0);
    return { p, score };
  }).filter((x) => x.score > 0).sort((a, b) => b.score - a.score).slice(0, limit);
  return scored.map((x) => x.p);
}

/** Starlight book chapters for a topic — reads .md files from src/content/docs/books/<bookSlug>/. */
export async function getBookChaptersForTopic(topic: Topic): Promise<{ title: string; href: string; book: string }[]> {
  if (!topic.bookSlug) return [];
  const { readdirSync, readFileSync } = await import('node:fs');
  const { resolve } = await import('node:path');
  const dir = resolve(process.cwd(), `src/content/docs/books/${topic.bookSlug}`);
  let files: string[] = [];
  try { files = readdirSync(dir).filter((f: string) => f.endsWith('.md') && f !== 'index.md'); } catch { return []; }
  const chapters = files.map((f) => {
    const txt = readFileSync(`${dir}/${f}`, 'utf-8');
    let title = f.replace(/\.md$/, '');
    const fm = txt.match(/^[﻿\s]*---[\s\S]*?title:\s*(.+?)\s*$/m);
    if (fm) title = fm[1].replace(/['"]/g, '');
    else {
      const h1 = txt.match(/^#\s+(.+)$/m);
      if (h1) title = h1[1];
    }
    const slug = f.replace(/\.md$/, '');
    return { title, href: `/books/${topic.bookSlug}/${slug}/`, book: topic.bookSlug! };
  }).sort((a, b) => a.title.localeCompare(b.title));
  return chapters;
}

/** Related posts — share at least one tag, excluding self, top N. */
export async function getRelatedPosts(current: Post, n = 3): Promise<Post[]> {
  const posts = await getPublishedPosts();
  const scored = posts
    .filter((p) => p.data.slug !== current.data.slug)
    .map((p) => ({
      p,
      score: p.data.tags.filter((t) => current.data.tags.includes(t)).length +
        (p.data.category && current.data.category && p.data.category === current.data.category ? 1 : 0),
    }))
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, n);
  return scored.map((x) => x.p);
}
