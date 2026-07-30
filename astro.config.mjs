import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import sitemap from '@astrojs/sitemap';

export default defineConfig({
  site: 'https://yuancaoyaohw.github.io',
  integrations: [
    sitemap({
      filter: (page) => {
        // Exclude Starlight's internal search/asset pages from sitemap
        return !page.includes('/pagefind/') && !page.includes('/_astro/');
      },
    }),
    starlight({
      title: 'yuancaoyaohw',
      defaultLocale: 'root',
      locales: {
        root: {
          label: '简体中文',
          lang: 'zh-CN',
        },
      },
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/yuancaoyaoHW',
        },
      ],
      customCss: ['./src/styles/collapsible-sidebars.css'],
      components: {
        Head: './src/components/overrides/Head.astro',
      },
      sidebar: [
        {
          label: '电子书',
          items: [
            {
              label: 'Modern GPU Programming For MLSys',
              items: [
                {
                  autogenerate: {
                    directory: 'books/modern-gpu-programming-for-mlsys',
                  },
                },
              ],
            },
          ],
        },
      ],
    }),
  ],
});
