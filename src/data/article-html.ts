/**
 * 从 src/articles/**​/index.html 里取出可以嵌进站点外壳的 head / body 片段。
 *
 * 这些源文件本来是能独立打开的整页，自带一套导航、页脚、主题脚本和 tokens.css
 * 引用。嵌进 BaseLayout 时那几样都会和站点外壳打架（两条导航、两个主题开关、
 * 文章的 tokens.css 覆盖站点的），所以在这里统一摘掉。
 *
 * 抽成公共函数是因为 /blog/[slug] 和 /blog/interview-qa 都要做同一件事——
 * 两处各写一份的话，以后改剥离规则必然漏掉一处。
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

export interface ArticleHtml {
  /** 文章 <head> 里的样式与 meta，已去掉 tokens.css 与主题引导脚本 */
  headHtml: string;
  /** 文章 <body> 内容，已去掉自带导航 / 页脚 / 主题脚本 */
  bodyHtml: string;
  /** 源文件是否存在 */
  found: boolean;
}

export function extractArticleHtml(relPath: string): ArticleHtml {
  let raw = '';
  try {
    raw = readFileSync(resolve(process.cwd(), relPath), 'utf-8');
  } catch {
    return { headHtml: '', bodyHtml: '', found: false };
  }

  let headHtml = '';
  const headMatch = raw.match(/<head[^>]*>([\s\S]*?)<\/head>/i);
  if (headMatch) {
    headHtml = headMatch[1]
      // 站点令牌由 BaseLayout 统一引入；文章再引一次会以相同特异性覆盖后者
      .replace(/<link[^>]*href="[^"]*\/styles\/tokens\.css"[^>]*>/gi, '')
      // 主题引导交给站点，文章自带的那段会和站点脚本抢 data-theme
      .replace(/<script[^>]*>[^<]*?localStorage[^<]*?data-theme[^<]*?<\/script>/gis, '');
  }

  let bodyHtml = '';
  const bodyMatch = raw.match(/<body[^>]*>([\s\S]*?)<\/body>/i);
  if (bodyMatch) {
    bodyHtml = bodyMatch[1]
      .replace(/<style\s+data-sn-hide-old[^>]*>[\s\S]*?<\/style>/gi, '')
      .replace(/<style\s+data-sn-nav[^>]*>[\s\S]*?<\/style>/gi, '')
      .replace(/<nav\s+class="sn-nav"[^>]*>[\s\S]*?<\/nav>/gi, '')
      .replace(/<script\s+data-sn-pad[^>]*>[\s\S]*?<\/script>/gi, '')
      // 只删站点自己的页脚，文章内部的 <footer> 语义块要留
      .replace(/<footer[^>]*>[\s\S]*?<\/footer>/gi, (m) =>
        /Semantic Paper Radar|hermes-agent|yuancaoyaohw|© 2026/i.test(m) ? '' : m)
      .replace(/<script[^>]*>[^<]*?function\s+toggleTheme[\s\S]*?<\/script>/gi, '')
      .replace(/<script[^>]*>[^<]*?kv-radar-theme[\s\S]*?<\/script>/gi, '')
      .trim();
  }

  return { headHtml, bodyHtml, found: true };
}
