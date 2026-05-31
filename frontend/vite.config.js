import { resolve } from 'path';
import { defineConfig } from 'vite';

// Multi-page: landing (index.html) + chat app (app.html).
export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        landing: resolve(__dirname, 'index.html'),
        app: resolve(__dirname, 'app.html'),
      },
    },
  },
});
