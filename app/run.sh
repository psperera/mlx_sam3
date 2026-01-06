#!/bin/bash

# SAM3 Segmentation Studio - Development Server Launcher
# Starts both backend (FastAPI) and frontend (Next.js) servers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   SAM3 Segmentation Studio Launcher    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down servers...${NC}"
    kill $(jobs -p) 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Ensure python env + dependencies (uv) and then install backend deps into that same venv.
# Note: `uv run` may recreate `.venv` to match the project, so we sync first, then install
# backend requirements afterwards to avoid them being wiped.
echo -e "${YELLOW}Ensuring backend dependencies...${NC}"
cd "$PROJECT_ROOT"
uv sync --quiet
uv pip install -r "$BACKEND_DIR/requirements.txt" --quiet

# Check if frontend dependencies are installed
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    cd "$FRONTEND_DIR" && npm install
    cd "$SCRIPT_DIR"
fi

echo -e "${GREEN}Starting Backend (FastAPI) on http://localhost:8090${NC}"
cd "$PROJECT_ROOT"
"$PROJECT_ROOT/.venv/bin/python" "$BACKEND_DIR/main.py" &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

echo -e "${GREEN}Starting Frontend (Next.js) on http://localhost:3000${NC}"
cd "$FRONTEND_DIR"
npm run dev &
FRONTEND_PID=$!

echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}  Servers are running!${NC}"
echo -e "${GREEN}  Frontend: http://localhost:3000${NC}"
echo -e "${GREEN}  Backend:  http://localhost:8090${NC}"
echo -e "${GREEN}  API Docs: http://localhost:8090/docs${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"
echo ""

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID

