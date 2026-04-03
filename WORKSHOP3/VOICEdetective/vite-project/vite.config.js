// vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/OOE-Virtual-Lab/WORKSHOP3/VOICEdetective/vite-project/dist/' 
})
