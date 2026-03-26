# ✈ Agentic AI System for Indian Airport Traffic Management
### LangChain · ReAct Agents · FastAPI · Real-Time Dashboard · ML Delay Prediction

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│               DATA INGESTION LAYER                          │
│  ATC Radar · ACARS · IMD Weather · NOTAM · DATIS · BRS/DCS  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│           LANGCHAIN ORCHESTRATOR (ReAct Executor)           │
│                 Multi-Agent Dispatcher                       │
└────┬──────────┬────────────┬─────────────────┬─────────────┘
     │          │            │                 │
┌────▼────┐ ┌──▼──────┐ ┌───▼──────────┐ ┌───▼──────────────┐
│ Runway  │ │ Weather │ │  Scheduling  │ │   Supervisor     │
│  Agent  │ │  Agent  │ │    Agent     │ │     Agent        │
│         │ │         │ │              │ │  (Full Access)   │
└────┬────┘ └──┬──────┘ └───┬──────────┘ └───┬──────────────┘
     │         │            │                 │
┌────▼─────────▼────────────▼─────────────────▼──────────────┐
│                   14 LANGCHAIN TOOLS                        │
│  flight status · runway assign · weather assess             │
│  delay predict · alert raise · capacity status · ...        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│           IN-MEMORY DATA STORE (swap → Redis / DB)          │
│  5 Indian Hub Airports · 30 Simulated Flights · Live WX     │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
airport_atm/
├── agents/
│   └── atm_agents.py        # 4 ReAct agents + ATMAgentSystem dispatcher
├── tools/
│   └── atm_tools.py         # 14 LangChain @tool functions
├── models/
│   ├── schemas.py           # Pydantic v2 data models
│   └── delay_predictor.py   # GradientBoosting delay ML model
├── data/
│   └── store.py             # In-memory store + seed data (5 Indian airports)
├── dashboard/
│   ├── api.py               # FastAPI REST + WebSocket server
│   └── terminal_dashboard.py# Rich terminal live dashboard
├── tests/
│   └── test_system.py       # 25+ pytest unit & integration tests
├── main.py                  # CLI entry point (demo/api/agent/train)
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Install dependencies
```bash
cd airport_atm
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — add OPENAI_API_KEY for full agent mode
```

### 3. Run modes

#### Terminal Dashboard (no API key needed)
```bash
python main.py demo
```
Live Rich terminal showing flights, weather, capacity, alerts — auto-refreshes every 2s.

#### FastAPI Server
```bash
python main.py api
# → http://localhost:8000/docs          (Swagger UI)
# → http://localhost:8000/dashboard      (Web dashboard)
# → ws://localhost:8000/ws/live          (WebSocket feed)
```

#### Interactive ReAct Agent REPL
```bash
python main.py agent
```
Try these queries:
```
ATM> Check weather at all airports and raise alerts for any severe conditions
ATM> What is the arrival sequence for VIDP right now?
ATM> List all delayed flights and predict how much worse the delays will get
ATM> Assign the best available runway at VABB to the highest priority inbound flight
ATM> Generate a full situational awareness report for the network
```

#### Train ML Model
```bash
python main.py train
```

### 4. Run tests
```bash
pytest tests/ -v
```

---

## The Four ReAct Agents

| Agent | Responsibility | Tools Available |
|---|---|---|
| **RunwayAgent** | Runway allocation, arrival/departure sequencing, wake turbulence separation | 8 tools |
| **WeatherAgent** | Meteorological monitoring, LVP advisories, METAR assessment, diversion recommendations | 7 tools |
| **SchedulingAgent** | Delay prediction, ATFM ground delay programmes, gate coordination, slot management | 9 tools |
| **SupervisorAgent** | System-wide oversight, emergency protocols, multi-airport flow management | All 14 tools |

---

## The 14 LangChain Tools

| Tool | Description |
|---|---|
| `get_flight_status` | Current status, delay, altitude, speed for any flight |
| `list_active_flights` | All active flights, optionally filtered by airport |
| `get_delayed_flights_report` | All delayed flights sorted by severity |
| `assign_runway` | Assign a runway with conflict/weather checks |
| `update_flight_status` | Update flight status and propagate delay |
| `get_weather_conditions` | METAR-style weather for any Indian airport |
| `get_all_airports_weather` | Network-wide weather snapshot |
| `assess_weather_impact` | Capacity impact, operational constraints, ILS requirements |
| `get_arrival_sequence` | Priority-ordered inbound sequence with wake turbulence spacing |
| `check_gate_availability` | Gate count and utilisation by airport |
| `predict_delay` | ML-powered delay forecast with contributing factors |
| `raise_alert` | Create INFO/WARNING/CRITICAL/EMERGENCY alerts |
| `get_active_alerts` | All unresolved system alerts sorted by severity |
| `get_airport_capacity_status` | Utilisation, saturation detection, ATFM recommendations |

---

## Indian Airports Modelled

| ICAO | IATA | City | Capacity/hr | Runways |
|---|---|---|---|---|
| VIDP | DEL | Delhi | 46 | 28, 29, 09/27 (CAT II/III ILS) |
| VABB | BOM | Mumbai | 48 | 09, 14/32 (CAT I ILS) |
| VOMM | MAA | Chennai | 30 | 07, 12/30 (CAT I ILS) |
| VOBL | BLR | Bengaluru | 28 | 09/27 (CAT II ILS) |
| VECO | CCU | Kolkata | 24 | 01R/19L, 01L/19R |

---

## ML Delay Prediction Model

**Algorithm:** Gradient Boosting Regressor (sklearn)  
**Features:**
- Aircraft wake turbulence category
- Departure & arrival weather severity (CLEAR → EXTREME)
- Wind speed (kts)
- Visibility (km)
- Hour of day (IST peak detection)
- Airport congestion score
- Historical on-time performance rate

**Typical performance:**
- MAE: ~8–12 minutes
- Within 15 min accuracy: ~70–75%

Replace `_generate_training_data()` with real DGCA historical data for production accuracy.

---

## Production Extensions

1. **Replace in-memory store** → PostgreSQL + Redis pub/sub for multi-instance state
2. **Real data feeds** → DGCA ATC radar API, IMD weather API, ACARS integration
3. **LLM caching** → LangChain cache layer (Redis) to reduce API costs
4. **Agent memory** → Add `ConversationBufferWindowMemory` for multi-turn ATC conversations
5. **ATFM integration** → Connect to Eurocontrol CFMU / DGCA flow management APIs
6. **Auth** → JWT authentication on FastAPI endpoints for ATC operator access
7. **Observability** → LangSmith tracing for agent reasoning audit trails

---

## DGCA Regulations Implemented

- Wake turbulence separation minima (ICAO Doc 4444)
- ILS approach categories (CAT I/II/III) linked to weather minima
- Low Visibility Procedure (LVP) triggers at <1500m visibility
- Priority sequencing: Emergency > Fuel Critical > Medical > Normal
- Mumbai runway curfew awareness (14/32 after 23:00 IST)
