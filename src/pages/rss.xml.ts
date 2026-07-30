import rss from '@astrojs/rss';
import type { APIContext } from 'astro';
import { getPublishedPosts, postHref, getLatestPaper, paperHref } from '../data/content.ts';

export async function GET(context: APIContext) {
  const posts = await getPublishedPosts();
  const latestPaper = await getLatestPaper();
  const items = posts.map((post) => ({
    title: post.data.title,
    description: post.data.description,
    pubDate: new Date(post.data.publishedDate),
    link: postHref(post),
    categories: post.data.tags,
  }));
  // Include latest paper digest
  if (latestPaper) {
    items.unshift({
      title: latestPaper.data.title,
      description: latestPaper.data.description,
      pubDate: new Date(latestPaper.data.date),
      link: paperHref(latestPaper),
      categories: ['论文日报'],
    });
  }
  return rss({
    title: 'yuancaoyaohw — AI Infra, Explained Deeply',
    description: '深入理解 LLM 推理、Serving、GPU Kernel 与系统优化。深度技术文章与每日论文精选。',
    site: context.site!,
    items,
    customData: '<language>zh-cn</language>',
  });
}
