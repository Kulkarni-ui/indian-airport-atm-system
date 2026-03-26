"""
data/store.py
In-memory data store seeded with real Indian airport & flight data.
Acts as the shared state layer across all agents.
"""
from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from typing import Optional

from models.schemas import (
    AlertLevel, Flight, FlightStatus, IndianAirport,
    Runway, RunwayStatus, SystemAlert, WeatherCondition, WeatherSeverity,
)


# ─── Indian Airport Registry ──────────────────────────────────────────────────

AIRPORTS: dict[str, IndianAirport] = {
    "VIDP": IndianAirport(icao_code="VIDP", iata_code="DEL", name="Indira Gandhi International",
                          city="Delhi", elevation_ft=777, latitude=28.5562, longitude=77.1000,
                          runways=["10/28", "11/29", "09/27"], capacity_per_hour=46),
    "VABB": IndianAirport(icao_code="VABB", iata_code="BOM", name="Chhatrapati Shivaji Maharaj International",
                          city="Mumbai", elevation_ft=37, latitude=19.0896, longitude=72.8656,
                          runways=["09/27", "14/32"], capacity_per_hour=48),
    "VOMM": IndianAirport(icao_code="VOMM", iata_code="MAA", name="Chennai International",
                          city="Chennai", elevation_ft=52, latitude=12.9900, longitude=80.1693,
                          runways=["07/25", "12/30"], capacity_per_hour=30),
    "VOBL": IndianAirport(icao_code="VOBL", iata_code="BLR", name="Kempegowda International",
                          city="Bengaluru", elevation_ft=3000, latitude=13.1986, longitude=77.7066,
                          runways=["09/27"], capacity_per_hour=28),
    "VECO": IndianAirport(icao_code="VECO", iata_code="CCU", name="Netaji Subhas Chandra Bose International",
                          city="Kolkata", elevation_ft=19, latitude=22.6547, longitude=88.4467,
                          runways=["01R/19L", "01L/19R"], capacity_per_hour=24),
}

RUNWAYS: dict[str, Runway] = {
    "VIDP-28":  Runway(runway_id="28",  airport_icao="VIDP", length_m=4430, width_m=60, ils_category="CAT III"),
    "VIDP-29":  Runway(runway_id="29",  airport_icao="VIDP", length_m=3810, width_m=60, ils_category="CAT II"),
    "VABB-09":  Runway(runway_id="09",  airport_icao="VABB", length_m=3445, width_m=45, ils_category="CAT I"),
    "VOMM-07":  Runway(runway_id="07",  airport_icao="VOMM", length_m=3658, width_m=45, ils_category="CAT I"),
    "VOBL-09":  Runway(runway_id="09",  airport_icao="VOBL", length_m=4000, width_m=60, ils_category="CAT II"),
    "VECO-19L": Runway(runway_id="19L", airport_icao="VECO", length_m=3627, width_m=45, ils_category="CAT I"),
}

# ─── In-Memory State ──────────────────────────────────────────────────────────

flights:       dict[str, Flight]           = {}
weather:       dict[str, WeatherCondition] = {}
alerts:        list[SystemAlert]           = []
decisions_log: list[dict]                  = []


# ─── Seed Data Generators ─────────────────────────────────────────────────────

_AIRLINES = [
    ("AIC", "AI"),   # Air India
    ("IGO", "6E"),   # IndiGo
    ("SEJ", "SG"),   # SpiceJet
    ("GOW", "G8"),   # Go First
    ("VTI", "UK"),   # Vistara
    ("BLU", "IX"),   # Air India Express
]
_AIRCRAFT = ["A320", "B738", "A321", "B77W", "A319", "ATR72", "B787"]
_AIRPORT_CODES = list(AIRPORTS.keys())


def _random_flight(idx: int) -> Flight:
    airline_icao, iata = random.choice(_AIRLINES)
    origin      = random.choice(_AIRPORT_CODES)
    destination = random.choice([a for a in _AIRPORT_CODES if a != origin])
    dep = datetime.utcnow() - timedelta(minutes=random.randint(-60, 120))
    arr = dep + timedelta(hours=random.uniform(1, 4))
    delay = random.choices([0, random.randint(15, 180)], weights=[0.6, 0.4])[0]
    status = random.choice(list(FlightStatus))
    return Flight(
        flight_id=f"{iata}{random.randint(100, 999)}",
        callsign=f"{airline_icao}{random.randint(100,999)}",
        aircraft_type=random.choice(_AIRCRAFT),
        airline_icao=airline_icao,
        origin_icao=origin,
        destination_icao=destination,
        scheduled_departure=dep,
        scheduled_arrival=arr,
        status=status,
        delay_minutes=delay,
        altitude_ft=random.randint(5000, 39000) if status == FlightStatus.EN_ROUTE else None,
        speed_kts=random.randint(380, 480) if status == FlightStatus.EN_ROUTE else None,
        priority_level=random.choices([0, random.randint(1, 10)], weights=[0.9, 0.1])[0],
    )


def _random_weather(icao: str) -> WeatherCondition:
    vis  = random.uniform(1.0, 15.0)
    sev  = (WeatherSeverity.EXTREME if vis < 2 else
            WeatherSeverity.SEVERE  if vis < 4 else
            WeatherSeverity.MODERATE if vis < 8 else
            WeatherSeverity.CLEAR)
    phenomena = random.choices([[], ["FG"], ["TS"], ["RA"], ["HZ"]], weights=[0.5, 0.1, 0.1, 0.2, 0.1])[0]
    return WeatherCondition(
        airport_icao=icao,
        visibility_km=round(vis, 1),
        wind_speed_kts=random.uniform(0, 30),
        wind_direction_deg=random.uniform(0, 360),
        ceiling_ft=random.choice([None, 300, 700, 1500, 3000, 10000]),
        temperature_c=random.uniform(18, 42),
        dew_point_c=random.uniform(10, 25),
        qnh_hpa=random.uniform(1000, 1025),
        weather_phenomena=phenomena,
        severity=sev,
        raw_metar=f"METAR {icao} AUTO",
    )


def seed_data(num_flights: int = 25) -> None:
    """Populate the in-memory store with realistic simulated data."""
    for i in range(num_flights):
        f = _random_flight(i)
        flights[f.flight_id] = f
    for icao in AIRPORTS:
        weather[icao] = _random_weather(icao)


def get_flight(flight_id: str) -> Optional[Flight]:
    return flights.get(flight_id)


def get_weather(icao: str) -> Optional[WeatherCondition]:
    return weather.get(icao)


def get_active_flights(airport_icao: Optional[str] = None) -> list[Flight]:
    active_statuses = {
        FlightStatus.SCHEDULED, FlightStatus.BOARDING,
        FlightStatus.APPROACHING, FlightStatus.EN_ROUTE,
    }
    result = [f for f in flights.values() if f.status in active_statuses]
    if airport_icao:
        result = [f for f in result
                  if f.origin_icao == airport_icao or f.destination_icao == airport_icao]
    return result


def get_delayed_flights() -> list[Flight]:
    return [f for f in flights.values() if f.is_delayed]


def add_alert(level: AlertLevel, source: str, message: str,
              flight_id: Optional[str] = None, airport_icao: Optional[str] = None) -> SystemAlert:
    alert = SystemAlert(
        alert_id=str(uuid.uuid4())[:8],
        level=level,
        source_agent=source,
        message=message,
        flight_id=flight_id,
        airport_icao=airport_icao,
    )
    alerts.append(alert)
    return alert


def log_decision(decision: dict) -> None:
    decisions_log.append({**decision, "timestamp": datetime.utcnow().isoformat()})
