// vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: './' // هذا السطر يحل مشكلة الـ 404 لأنه يجعل المسارات نسبية
})