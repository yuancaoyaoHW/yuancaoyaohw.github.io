#!/usr/bin/env node
/**
 * 断言：文章内联样式不得劫持站点设计令牌。
 *
 * 背景：src/articles/**\/index.html 的 <head> 内联 <style> 会经
 * [slug].astro 的 slot="head" 注入到 tokens.css 之后。:root 与
 * [data-theme="..."] 特异性相同（均为 0,1,0），后来者胜——文章一旦
 * 定义 --orange / --border 之类的站点令牌名，就会连带改掉导航、CTA、
 * focus ring 和 footer 的配色。
 *
 * 文章自有变量请统一加 --a- 前缀；共享语义色见 public/styles/article-tokens.css。
 *
 * 用法：node scripts/check-article-tokens.mjs   （npm run check:tokens）
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

/** 站点令牌名单 — 取自 public/styles/tokens.css 的 :root / [data-theme] 块 */
const SITE_TOKENS = [
  'black', 'bg-1', 'bg-2', 'panel', 'panel-hover', 'white',
  'orange', 'orange-soft', 'text', 'text-muted',
  'border', 'border-hover', 'nav-bg', 'overlay',
  'grid-line', 'glow-1', 'glow-2', 'footer-link',
  'radius', 'radius-sm', 'radius-md', 'radius-lg', 'radius-pill',
  'page-max', 'prose-max', 'wide-content-max',
  'content-w', 'content-w-narrow', 'content-w-wide',
  'font-display', 'font-body', 'font-mono',
  'nav-h', 'focus-ring', 'shadow-sm', 'shadow-md', 'shadow-lg',
];

/**
 * 只检查 src/articles/blog：这些 index.html 会被 src/pages/blog/[slug].astro
 * 读取并注入。src/articles/papers/* 是遗留的独立页面，papers/[slug].astro 改从
 * papers.json 渲染、不再注入其 <head>，因此不在检查范围内。
 */
const ROOT = 'src/articles/blog';
const files = [];
(function walk(dir) {
  for (const e of readdirSync(dir)) {
    const p = join(dir, e);
    if (statSync(p).isDirectory()) walk(p);
    else if (e === 'index.html') files.push(p);
  }
})(ROOT);

const violations = [];

for (const file of files) {
  const src = readFileSync(file, 'utf-8');

  // 只检查 <head>：body 内的 data-sn-* 样式块会被 [slug].astro 剥离，不会生效。
  const head = /<head[^>]*>([\s\S]*?)<\/head>/i.exec(src)?.[1] ?? '';
  const lineOf = (idx) => src.slice(0, idx).split('\n').length;

  for (const token of SITE_TOKENS) {
    const re = new RegExp(`(?:^|[;{\\s])--${token}(?![\\w-])\\s*:`, 'gm');
    let m;
    while ((m = re.exec(head)) !== null) {
      violations.push({ file, line: lineOf(m.index), token });
    }
  }
}

if (violations.length > 0) {
  console.error(`\n✗ 发现 ${violations.length} 处站点令牌劫持：\n`);
  const byFile = new Map();
  for (const v of violations) {
    if (!byFile.has(v.file)) byFile.set(v.file, []);
    byFile.get(v.file).push(v);
  }
  for (const [file, vs] of byFile) {
    console.error(`  ${file}`);
    for (const v of vs) console.error(`    L${v.line}  --${v.token}  →  改为 --a-${v.token}`);
  }
  console.error(`\n文章内联 <style> 注入在 tokens.css 之后，同名变量会覆盖站点令牌，`);
  console.error(`导致导航、CTA、footer 的配色被文章接管。给变量加 --a- 前缀即可。`);
  console.error(`共享语义色见 public/styles/article-tokens.css。\n`);
  process.exit(1);
}

console.log(`✓ ${files.length} 篇文章未劫持站点令牌`);
