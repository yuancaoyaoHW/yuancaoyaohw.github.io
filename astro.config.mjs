import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://yuancaoyaohw.github.io',
  integrations: [
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
