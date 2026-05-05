#!/bin/bash
echo "============================================"
echo " GNSS AI Dashboard — Startup"
echo " GLA University B.Tech 2025-26"
echo "============================================"

echo ""
echo "[1/2] Installing & starting FastAPI backend..."
cd backend
pip install -r requirements.txt --quiet
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
cd ..

sleep 2

echo "[2/2] Installing & starting React frontend..."
cd frontend
npm install --silent
npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "============================================"
echo " Backend:  http://localhost:8000"
echo " Frontend: http://localhost:3000"
echo " API Docs: http://localhost:8000/docs"
echo "============================================"
echo ""
echo "Press Ctrl+C to stop both servers."

wait $BACKEND_PID $FRONTEND_PID
