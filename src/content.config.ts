/**
 * Astro Content Collections config (Astro 7 — uses the loader API).
 *
 * Strategy: the actual article HTML lives in public/blog/<slug>/index.html and
 * is served as-is. These collections store ONLY metadata (JSON files) so that
 * the homepage, blog archive, topic pages, search, RSS, and sitemap all read
 * from a single source. The collection entries point to the existing HTML via
 * `slug` / `externalHref`, preserving all existing URLs.
 */
import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

/** Docs collection — Starlight's content (uses Starlight's own schema via docsLoader). */
const docs = defineCollection({ loader: docsLoader(), schema: docsSchema() });

/** Posts collection — deep technical articles (metadata only, HTML in public/). */
const posts = defineCollection({
  loader: glob({ pattern: '**/*.json', base: './src/content/posts' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    publishedDate: z.string(),
    updatedDate: z.string().optional(),
    category: z.enum(['llm-serving', 'speculative-decoding', 'kv-cache', 'gpu', 'moe', 'inference-engine', 'safety', 'survey']).optional(),
    primaryTopic: z.enum(['llm-serving', 'speculative-decoding', 'kv-cache', 'gpu', 'moe', 'inference-engine']).optional(),
    secondaryTopics: z.array(z.string()).default([]),
    tags: z.array(z.string()).default([]),
    featured: z.boolean().default(false),
    draft: z.boolean().default(false),
    readingTime: z.number().optional(),
    slug: z.string(),
    externalHref: z.string().optional(),
    hasMath: z.boolean().default(false),
    hasAnimation: z.boolean().default(false),
  }),
});

/** Paper digest issues — one entry per daily issue. */
const papers = defineCollection({
  loader: glob({ pattern: '**/*.json', base: './src/content/papers' }),
  schema: z.object({
    date: z.string(),
    title: z.string(),
    description: z.string(),
    paperCount: z.number().optional(),
    selectedCount: z.number().optional(),
    channels: z.array(z.string()).default(['AI Infra', 'Algorithms', 'Architecture']),
    highlights: z.array(z.object({
      name: z.string(),
      org: z.string().optional(),
      note: z.string().optional(),
    })).default([]),
    draft: z.boolean().default(false),
    slug: z.string(),
  }),
});

/** Interviews collection. */
const interviews = defineCollection({
  loader: glob({ pattern: '**/*.json', base: './src/content/interviews' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    publishedDate: z.string(),
    tags: z.array(z.string()).default([]),
    slug: z.string(),
    draft: z.boolean().default(false),
  }),
});

/** Travel guides collection. */
const travel = defineCollection({
  loader: glob({ pattern: '**/*.json', base: './src/content/travel' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    publishedDate: z.string(),
    tags: z.array(z.string()).default([]),
    slug: z.string(),
    draft: z.boolean().default(false),
  }),
});

/** Books — metadata for ebook entries (Starlight manages the actual content). */
const books = defineCollection({
  loader: glob({ pattern: '**/*.json', base: './src/content/books' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    href: z.string(),
    tag: z.string(),
    chapters: z.number().optional(),
    status: z.enum(['ongoing', 'complete', 'stub']).default('ongoing'),
    external: z.boolean().default(false),
  }),
});

export const collections = { docs, posts, papers, interviews, travel, books };
