@echo off
echo ============================================
echo  GNSS AI Dashboard — Startup
echo  GLA University B.Tech 2025-26
echo ============================================

echo.
echo [1/2] Starting FastAPI backend...
cd backend
start cmd /k "pip install -r requirements.txt && uvicorn main:app --reload --port 8000"
cd ..

timeout /t 3 >nul

echo [2/2] Starting React frontend...
cd frontend
start cmd /k "npm install && npm start"
cd ..

echo.
echo ============================================
echo  Backend:  http://localhost:8000
echo  Frontend: http://localhost:3000
echo  API Docs: http://localhost:8000/docs
echo ============================================
