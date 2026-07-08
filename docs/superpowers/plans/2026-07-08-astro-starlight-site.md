# Astro Starlight Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `yuancaoyaohw.github.io` into an Astro + Starlight personal site for books and blog posts.

**Architecture:** Astro owns the personal homepage and future blog routes. Starlight owns documentation-style book pages under `src/content/docs/books/...`. Static book images and demos live in `public/books/...`, and GitHub Actions builds `dist/` for GitHub Pages.

**Tech Stack:** Astro, `@astrojs/starlight`, Markdown/MDX content, GitHub Actions Pages deployment.

---

### Task 1: Scaffold Astro + Starlight Project

**Files:**
- Create: `package.json`
- Create: `astro.config.mjs`
- Create: `tsconfig.json`
- Create: `src/content.config.ts`
- Create: `src/pages/index.astro`
- Create: `src/pages/blog/index.astro`
- Create: `.github/workflows/deploy.yml`
- Keep: `.nojekyll`

- [ ] **Step 1: Create package metadata**

Write `package.json`:

```json
{
  "name": "yuancaoyaohw.github.io",
  "type": "module",
  "private": true,
  "scripts": {
    "dev": "astro dev",
    "build": "astro check && astro build",
    "preview": "astro preview"
  },
  "dependencies": {
    "@astrojs/check": "latest",
    "@astrojs/starlight": "latest",
    "astro": "latest",
    "typescript": "latest"
  },
  "devDependencies": {}
}
```

- [ ] **Step 2: Configure Astro and Starlight**

Write `astro.config.mjs` with `site: 'https://yuancaoyaohw.github.io'`, Starlight title, Chinese locale labels, and sidebar entries for the migrated GPU book.

- [ ] **Step 3: Configure content collections**

Write `src/content.config.ts` using Starlight's `docsLoader()` and `docsSchema()`.

- [ ] **Step 4: Add personal homepage and blog placeholder**

Write `src/pages/index.astro` as the site entrance linking to `/books/modern-gpu-programming-for-mlsys/` and `/blog/`. Write `src/pages/blog/index.astro` as a simple future blog index.

- [ ] **Step 5: Add GitHub Pages workflow**

Write `.github/workflows/deploy.yml` using the official Astro GitHub Pages workflow pattern: checkout, setup Node, install, build, upload Pages artifact, deploy Pages.

- [ ] **Step 6: Verify build tooling**

Run:

```powershell
npm install
npm run build
```

Expected: dependencies install and Astro builds `dist/`.

- [ ] **Step 7: Commit**

```powershell
git add package.json package-lock.json astro.config.mjs tsconfig.json src .github .nojekyll
git commit -m "setup astro starlight site"
```

### Task 2: Migrate Current Book Content

**Files:**
- Create: `src/content/docs/books/modern-gpu-programming-for-mlsys/index.md`
- Create: `src/content/docs/books/modern-gpu-programming-for-mlsys/*.md`
- Create: `public/books/modern-gpu-programming-for-mlsys/img/*`
- Create: `public/books/modern-gpu-programming-for-mlsys/demo/*`
- Remove: old root static directory `modern-gpu-programming-for-mlsys/`

- [ ] **Step 1: Copy translated Markdown chapters**

Copy Markdown and RST source files from `C:\Users\hw\Documents\modern-gpu-programming-for-mlsys\zh\` into `src/content/docs/books/modern-gpu-programming-for-mlsys/`, flattening each chapter directory to a `.md` page where practical.

- [ ] **Step 2: Copy static assets**

Copy images from `C:\Users\hw\Documents\modern-gpu-programming-for-mlsys\_images\` into `public/books/modern-gpu-programming-for-mlsys/img/`.

- [ ] **Step 3: Copy demos**

Copy HTML demos from `C:\Users\hw\Documents\modern-gpu-programming-for-mlsys\demo\` into `public/books/modern-gpu-programming-for-mlsys/demo/`.

- [ ] **Step 4: Update links**

Update image links to `/books/modern-gpu-programming-for-mlsys/img/...` and iframe/demo links to `/books/modern-gpu-programming-for-mlsys/demo/...`.

- [ ] **Step 5: Remove old generated static site**

Remove `modern-gpu-programming-for-mlsys/` after migrated Starlight pages build.

- [ ] **Step 6: Verify migrated build**

Run:

```powershell
npm run build
```

Expected: Astro build completes and routes include `/books/modern-gpu-programming-for-mlsys/`.

- [ ] **Step 7: Commit**

```powershell
git add src/content public/books astro.config.mjs
git rm -r modern-gpu-programming-for-mlsys
git commit -m "migrate gpu book to starlight"
```

### Task 3: Final Local Verification

**Files:**
- Read: `dist/index.html`
- Read: `dist/books/modern-gpu-programming-for-mlsys/index.html`

- [ ] **Step 1: Build from clean state**

Run:

```powershell
npm run build
```

Expected: command exits 0.

- [ ] **Step 2: Inspect output files**

Run:

```powershell
Test-Path dist/index.html
Test-Path dist/books/modern-gpu-programming-for-mlsys/index.html
```

Expected: both return `True`.

- [ ] **Step 3: Commit any final fixes**

If verification requires fixes, commit them with:

```powershell
git add .
git commit -m "fix astro starlight site build"
```

### Self-Review

- Spec coverage: covers personal homepage, book section, future blog entry point, Starlight docs, GitHub Pages workflow, and migration of current book assets.
- Placeholder scan: no unresolved placeholders remain.
- Type consistency: file paths and route paths consistently use `books/modern-gpu-programming-for-mlsys`.
