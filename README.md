# OpenJournal

A lightweight AI voice journaling application. Speak naturally to an AI assistant that guides self-reflection through probing questions. Uses ElevenLabs for voice and OpenRouter for LLM responses.

**By:** John Stewart, Sherelle McDaniel, Aniyah Tucker, Dominique Sanchez, Andy Coto, Jackeline Garcia Ulloa

## Tech Stack

- **Frontend:** React, Vite, Tailwind CSS
- **Voice:** ElevenLabs TTS + Realtime Speech-to-Text (Scribe v2)
- **LLM:** OpenRouter (Gemini 3.1 Pro for interviewer and reformat)

## Quick Start

1. **Install dependencies**
   ```bash
   npm install
   ```

2. **Configure environment**
   - Copy `.env.example` to `.env`
   - Add `OPENROUTER_API_KEY` (from [OpenRouter](https://openrouter.ai/keys)) and `ELEVENLABS_API_KEY` (from [ElevenLabs](https://elevenlabs.io/app/settings/api-keys))

3. **Run locally**
   ```bash
   npm run dev
   ```
   This starts the API server (port 3001 or 3002) and Vite dev server (port 5173).

4. **Build for production**
   ```bash
   npm run build
   ```

## Deployment (Vercel)

- Root directory: `.` (project root)
- Add `OPENROUTER_API_KEY` and `ELEVENLABS_API_KEY` to Vercel Environment Variables
- Deploy. The API routes are served as serverless functions.

## Project Structure

```
├── api/                 # Serverless API routes (interviewer, voice, transcribe, etc.)
├── scripts/             # Local API server for dev
├── src/
│   ├── pages/
│   │   └── Personaplex/ # Main journaling UI
│   └── ...
├── public/
└── ...
```
