import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: '127.0.0.1',
    hmr: {
      host: 'localhost',
    },
    proxy: {
      '/api':                    { target: 'http://127.0.0.1:5000', changeOrigin: true },
      '/asl_stream':             { target: 'http://127.0.0.1:5000', changeOrigin: true },
      '/asl_from_youtube_sentences': { target: 'http://127.0.0.1:5000', changeOrigin: true },
      '/extract_transcript':     { target: 'http://127.0.0.1:5000', changeOrigin: true },
      '/output':                 { target: 'http://127.0.0.1:5000', changeOrigin: true },
    },
  },
})
