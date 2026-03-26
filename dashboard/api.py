"""
dashboard/api.py
FastAPI application exposing the ATM agent system via REST and WebSocket.
Powers the real-time dashboard.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import data.store as store
from data.store import AIRPORTS, RUNWAYS, seed_data

app = FastAPI(
    title="Indian Airport ATM System",
    description="Agentic AI-powered Air Traffic Management for Indian Airports",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Seed on startup
@app.on_event("startup")
async def startup():
    seed_data(num_flights=30)
    print("[ATM] Data seeded. System ready.")


# ─── WebSocket Connection Manager ─────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)


manager = ConnectionManager()


# ─── REST Endpoints ──────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"service": "Indian Airport ATM System", "status": "operational",
            "airports": list(AIRPORTS.keys())}


@app.get("/flights")
async def list_flights(airport: Optional[str] = None, status: Optional[str] = None):
    flights = list(store.flights.values())
    if airport:
        flights = [f for f in flights
                   if f.origin_icao == airport.upper() or f.destination_icao == airport.upper()]
    if status:
        flights = [f for f in flights if f.status.value == status.upper()]
    return {
        "count": len(flights),
        "flights": [f.model_dump(mode="json") for f in flights],
    }


@app.get("/flights/{flight_id}")
async def get_flight(flight_id: str):
    f = store.get_flight(flight_id.upper())
    if not f:
        raise HTTPException(404, f"Flight {flight_id} not found")
    return f.model_dump(mode="json")


@app.get("/weather")
async def all_weather():
    return {icao: w.model_dump(mode="json") for icao, w in store.weather.items()}


@app.get("/weather/{icao}")
async def airport_weather(icao: str):
    w = store.get_weather(icao.upper())
    if not w:
        raise HTTPException(404, f"Weather data for {icao} not found")
    return w.model_dump(mode="json")


@app.get("/airports")
async def list_airports():
    return {k: v.model_dump() for k, v in AIRPORTS.items()}


@app.get("/alerts")
async def list_alerts(resolved: bool = False):
    alerts = [a for a in store.alerts if a.resolved == resolved]
    return {"count": len(alerts), "alerts": [a.model_dump(mode="json") for a in alerts]}


@app.get("/runways")
async def list_runways():
    return {k: v.model_dump(mode="json") for k, v in RUNWAYS.items()}


@app.get("/dashboard/summary")
async def dashboard_summary():
    """High-level summary for the dashboard header cards."""
    total     = len(store.flights)
    active    = store.get_active_flights()
    delayed   = store.get_delayed_flights()
    critical  = [a for a in store.alerts if a.level.value in ("CRITICAL", "EMERGENCY") and not a.resolved]

    weather_summary = {}
    for icao, w in store.weather.items():
        weather_summary[icao] = {
            "severity":    w.severity.value,
            "is_ifr":      w.is_ifr,
            "visibility":  w.visibility_km,
            "city":        AIRPORTS[icao].city,
        }

    return {
        "total_flights":      total,
        "active_flights":     len(active),
        "delayed_flights":    len(delayed),
        "critical_alerts":    len(critical),
        "airports_weather":   weather_summary,
        "decisions_today":    len(store.decisions_log),
        "timestamp":          datetime.utcnow().isoformat(),
    }


# ─── Agent Query Endpoint ─────────────────────────────────────────────────────

class AgentQueryRequest(BaseModel):
    query: str
    agent: Optional[str] = None  # runway | weather | scheduling | supervisor


@app.post("/agent/query")
async def agent_query(req: AgentQueryRequest):
    """
    Route a natural language query to the appropriate ATM agent.
    NOTE: Requires OPENAI_API_KEY set in environment.
    """
    if not os.getenv("OPENAI_API_KEY"):
        # Return a mock response when no API key is set (demo mode)
        return {
            "agent":  "DemoMode",
            "output": (
                f"[DEMO] Query received: '{req.query}'. "
                "Set OPENAI_API_KEY in .env to activate full ReAct agent reasoning. "
                "The system is seeded with live simulated data — all REST endpoints are fully functional."
            ),
            "steps": [],
        }

    # Lazy import to avoid loading LangChain if not needed
    from agents.atm_agents import ATMAgentSystem
    atm = ATMAgentSystem(verbose=False)
    result = await atm.arun(req.query, force_agent=req.agent)
    await manager.broadcast({"type": "agent_decision", "data": result})
    return result


# ─── WebSocket ───────────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """Push real-time ATM state updates every 5 seconds."""
    await manager.connect(websocket)
    try:
        while True:
            active   = store.get_active_flights()
            delayed  = store.get_delayed_flights()
            alerts   = [a for a in store.alerts if not a.resolved]

            snapshot = {
                "type": "state_update",
                "timestamp": datetime.utcnow().isoformat(),
                "active_flights": len(active),
                "delayed_flights": len(delayed),
                "active_alerts": len(alerts),
                "weather": {
                    icao: {"severity": w.severity.value, "visibility": w.visibility_km}
                    for icao, w in store.weather.items()
                },
            }
            await websocket.send_json(snapshot)
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ─── Embedded Dashboard HTML ──────────────────────────────────────────────────
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the embedded real-time ATM dashboard."""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, encoding="utf-8", errors="ignore") as f:
        return f.read()
