# SSDataAgent Web Console

React + Vite single-page application that provides a browser UI for the SSDataAgent backend.

## Prerequisites

- Node.js v20+ / npm v10+
- The Python backend running (`python -m ssdataagent.console` or uvicorn on port 8000)

## Development

```bash
# Install dependencies (run once, or after package.json changes)
npm install

# Start dev server with hot-reload (proxies /api → http://127.0.0.1:8000)
npm run dev
```

Visit <http://localhost:5173> in your browser. API requests are proxied to the FastAPI backend.

## Production build

```bash
npm run build
```

Writes a self-contained bundle to `dist/`. The Python backend mounts this directory at `/`
via FastAPI's `StaticFiles` mount, so `python -m ssdataagent.console` serves the full console.

## Tests

```bash
npm run test
```

Runs two vitest component tests (Leaderboard table rendering + Compare heatmap container).

## Views

| Route | View | Backend endpoint |
|-------|------|-----------------|
| `/` | Leaderboard | `GET /api/leaderboard` |
| `/runs` | Launcher | `GET /api/runs`, `POST /api/runs`, `GET /api/runs/:name/log` |
| `/runs/:name` | Run Detail | `GET /api/runs/:name/detail` |
| `/compare` | Compare (Plotly heatmap) | `POST /api/compare` |
| `/reports` | Report export | `POST /api/reports` |
| `/notebook` | Lab Notebook | `GET /api/notebook`, `POST /api/notebook` |
