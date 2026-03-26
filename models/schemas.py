"""
models/schemas.py
Core Pydantic v2 data models for the Indian Airport ATM system.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


# ─── Enumerations ─────────────────────────────────────────────────────────────

class FlightStatus(str, Enum):
    SCHEDULED   = "SCHEDULED"
    BOARDING    = "BOARDING"
    DEPARTED    = "DEPARTED"
    EN_ROUTE    = "EN_ROUTE"
    APPROACHING = "APPROACHING"
    LANDED      = "LANDED"
    DIVERTED    = "DIVERTED"
    CANCELLED   = "CANCELLED"
    DELAYED     = "DELAYED"


class RunwayStatus(str, Enum):
    ACTIVE    = "ACTIVE"
    INACTIVE  = "INACTIVE"
    OCCUPIED  = "OCCUPIED"
    CLOSED    = "CLOSED"
    INSPECTING = "INSPECTING"


class WeatherSeverity(str, Enum):
    CLEAR    = "CLEAR"
    LOW_VIS  = "LOW_VIS"
    MODERATE = "MODERATE"
    SEVERE   = "SEVERE"
    EXTREME  = "EXTREME"


class AlertLevel(str, Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


# ─── Airport ──────────────────────────────────────────────────────────────────

class IndianAirport(BaseModel):
    icao_code: str                          # e.g. VIDP, VABB, VOMM
    iata_code: str                          # e.g. DEL, BOM, MAA
    name: str
    city: str
    elevation_ft: float
    latitude: float
    longitude: float
    runways: list[str] = Field(default_factory=list)
    capacity_per_hour: int = 30             # movements per hour


# ─── Weather ──────────────────────────────────────────────────────────────────

class WeatherCondition(BaseModel):
    airport_icao: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    visibility_km: float
    wind_speed_kts: float
    wind_direction_deg: float
    ceiling_ft: Optional[float] = None
    temperature_c: float
    dew_point_c: float
    qnh_hpa: float
    weather_phenomena: list[str] = Field(default_factory=list)  # e.g. ["TS", "FG"]
    severity: WeatherSeverity = WeatherSeverity.CLEAR
    raw_metar: Optional[str] = None

    @computed_field
    @property
    def is_ifr(self) -> bool:
        """True when Instrument Flight Rules conditions apply."""
        return self.visibility_km < 5.0 or (self.ceiling_ft is not None and self.ceiling_ft < 1000)

    @computed_field
    @property
    def crosswind_component_kts(self) -> float:
        """Simplified crosswind – assume runway heading 270° (westerly) as default."""
        import math
        angle = math.radians(abs(self.wind_direction_deg - 270))
        return round(self.wind_speed_kts * math.sin(angle), 1)


# ─── Flight ───────────────────────────────────────────────────────────────────

class Flight(BaseModel):
    flight_id: str                          # e.g. AI-101
    callsign: str
    aircraft_type: str                      # e.g. B738, A320
    airline_icao: str                       # e.g. AIC, IGO
    origin_icao: str
    destination_icao: str
    scheduled_departure: datetime
    scheduled_arrival: datetime
    actual_departure: Optional[datetime] = None
    actual_arrival: Optional[datetime] = None
    status: FlightStatus = FlightStatus.SCHEDULED
    assigned_runway: Optional[str] = None
    assigned_gate: Optional[str] = None
    altitude_ft: Optional[float] = None
    speed_kts: Optional[float] = None
    heading_deg: Optional[float] = None
    delay_minutes: int = 0
    fuel_emergency: bool = False
    priority_level: int = Field(default=0, ge=0, le=10)  # 10 = emergency

    @computed_field
    @property
    def is_delayed(self) -> bool:
        return self.delay_minutes > 15

    @computed_field
    @property
    def eta_minutes(self) -> Optional[float]:
        if self.actual_arrival:
            delta = (self.actual_arrival - datetime.utcnow()).total_seconds()
            return round(delta / 60, 1)
        return None


# ─── Runway ───────────────────────────────────────────────────────────────────

class Runway(BaseModel):
    runway_id: str                          # e.g. "28R"
    airport_icao: str
    length_m: int
    width_m: int
    surface: str = "ASPHALT"
    ils_category: Optional[str] = None     # CAT I / II / III
    status: RunwayStatus = RunwayStatus.ACTIVE
    occupied_by: Optional[str] = None      # flight_id currently on runway
    last_inspection: Optional[datetime] = None
    ops_per_hour: int = 0                  # current throughput


# ─── Agent Decision ───────────────────────────────────────────────────────────

class AgentDecision(BaseModel):
    agent_name: str
    decision_type: str
    flight_id: Optional[str] = None
    runway_id: Optional[str] = None
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    action_taken: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = Field(default_factory=dict)


# ─── System Alert ─────────────────────────────────────────────────────────────

class SystemAlert(BaseModel):
    alert_id: str
    level: AlertLevel
    source_agent: str
    message: str
    flight_id: Optional[str] = None
    airport_icao: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    resolved: bool = False


# ─── Delay Prediction ─────────────────────────────────────────────────────────

class DelayPrediction(BaseModel):
    flight_id: str
    predicted_delay_minutes: float
    confidence: float
    contributing_factors: list[str]
    weather_impact_score: float = Field(ge=0.0, le=1.0)
    congestion_impact_score: float = Field(ge=0.0, le=1.0)
    historical_on_time_rate: float = Field(ge=0.0, le=1.0)
    recommendation: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
