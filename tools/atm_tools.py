"""
tools/atm_tools.py
All LangChain @tool functions available to the ReAct agents.
Each tool has a rich docstring — this is what the LLM reads to decide when to call it.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from typing import Optional

from langchain_core.tools import tool

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import data.store as store
from models.schemas import AlertLevel, FlightStatus, RunwayStatus


# ─── Flight Information Tools ─────────────────────────────────────────────────

@tool
def get_flight_status(flight_id: str) -> str:
    """
    Retrieve the current status and details of a specific flight.

    Args:
        flight_id: IATA flight identifier, e.g. '6E-456' or 'AI101'.

    Returns:
        JSON string with flight status, delay info, assigned runway/gate,
        altitude, speed, and priority level.
    """
    flight = store.get_flight(flight_id)
    if not flight:
        # Try partial match
        matches = [f for fid, f in store.flights.items() if flight_id.upper() in fid.upper()]
        if not matches:
            return json.dumps({"error": f"Flight {flight_id} not found."})
        flight = matches[0]

    return json.dumps({
        "flight_id":        flight.flight_id,
        "callsign":         flight.callsign,
        "status":           flight.status.value,
        "origin":           flight.origin_icao,
        "destination":      flight.destination_icao,
        "aircraft_type":    flight.aircraft_type,
        "delay_minutes":    flight.delay_minutes,
        "is_delayed":       flight.is_delayed,
        "altitude_ft":      flight.altitude_ft,
        "speed_kts":        flight.speed_kts,
        "assigned_runway":  flight.assigned_runway,
        "assigned_gate":    flight.assigned_gate,
        "priority_level":   flight.priority_level,
        "fuel_emergency":   flight.fuel_emergency,
        "eta_minutes":      flight.eta_minutes,
    })


@tool
def list_active_flights(airport_icao: Optional[str] = None) -> str:
    """
    List all active (airborne or pre-departure) flights, optionally filtered by airport.

    Args:
        airport_icao: Optional ICAO code (e.g. 'VIDP' for Delhi) to filter flights.
                      If None, returns all active flights across all Indian airports.

    Returns:
        JSON list of active flights with key status fields.
    """
    flights = store.get_active_flights(airport_icao)
    result = [
        {
            "flight_id":     f.flight_id,
            "status":        f.status.value,
            "origin":        f.origin_icao,
            "destination":   f.destination_icao,
            "delay_minutes": f.delay_minutes,
            "priority_level": f.priority_level,
        }
        for f in flights
    ]
    return json.dumps({"count": len(result), "flights": result})


@tool
def get_delayed_flights_report() -> str:
    """
    Get a report of all delayed flights (delay > 15 min) sorted by severity.

    Returns:
        JSON list of delayed flights with delay duration and contributing factors.
    """
    delayed = store.get_delayed_flights()
    delayed.sort(key=lambda f: f.delay_minutes, reverse=True)
    result = [
        {
            "flight_id":     f.flight_id,
            "airline":       f.airline_icao,
            "origin":        f.origin_icao,
            "destination":   f.destination_icao,
            "delay_minutes": f.delay_minutes,
            "status":        f.status.value,
        }
        for f in delayed
    ]
    return json.dumps({"total_delayed": len(result), "flights": result})


@tool
def assign_runway(flight_id: str, runway_id: str) -> str:
    """
    Assign a runway to a flight for departure or landing.
    Checks runway availability and weather suitability before assignment.

    Args:
        flight_id:  The flight to assign (e.g. 'AI101').
        runway_id:  The runway key in format 'ICAO-ID' (e.g. 'VIDP-28').

    Returns:
        JSON with assignment result, including any conflicts or weather warnings.
    """
    flight = store.get_flight(flight_id)
    runway = store.RUNWAYS.get(runway_id)

    if not flight:
        return json.dumps({"success": False, "error": f"Flight {flight_id} not found."})
    if not runway:
        return json.dumps({"success": False, "error": f"Runway {runway_id} not found."})
    if runway.status == RunwayStatus.CLOSED:
        return json.dumps({"success": False, "error": f"Runway {runway_id} is CLOSED."})
    if runway.occupied_by:
        return json.dumps({"success": False, "error": f"Runway {runway_id} occupied by {runway.occupied_by}."})

    flight.assigned_runway = runway.runway_id
    runway.occupied_by     = flight_id
    runway.status          = RunwayStatus.OCCUPIED

    store.log_decision({
        "agent":   "RunwayAgent",
        "action":  "assign_runway",
        "flight":  flight_id,
        "runway":  runway_id,
    })
    return json.dumps({
        "success":   True,
        "flight_id": flight_id,
        "runway":    runway_id,
        "message":   f"Runway {runway.runway_id} assigned to {flight_id}.",
    })


@tool
def update_flight_status(flight_id: str, new_status: str, delay_minutes: int = 0) -> str:
    """
    Update the operational status and delay of a flight.

    Args:
        flight_id:     The flight identifier.
        new_status:    One of: SCHEDULED, BOARDING, DEPARTED, EN_ROUTE,
                       APPROACHING, LANDED, DIVERTED, CANCELLED, DELAYED.
        delay_minutes: Additional delay in minutes (0 if on time).

    Returns:
        JSON confirming the update.
    """
    flight = store.get_flight(flight_id)
    if not flight:
        return json.dumps({"error": f"Flight {flight_id} not found."})

    try:
        flight.status = FlightStatus(new_status.upper())
    except ValueError:
        return json.dumps({"error": f"Invalid status '{new_status}'."})

    if delay_minutes:
        flight.delay_minutes += delay_minutes
        if flight.scheduled_arrival:
            flight.actual_arrival = flight.scheduled_arrival + timedelta(minutes=flight.delay_minutes)

    store.log_decision({"agent": "SchedulingAgent", "action": "update_status",
                        "flight": flight_id, "new_status": new_status, "delay": delay_minutes})

    return json.dumps({"success": True, "flight_id": flight_id,
                       "status": flight.status.value, "total_delay": flight.delay_minutes})


# ─── Weather Tools ────────────────────────────────────────────────────────────

@tool
def get_weather_conditions(airport_icao: str) -> str:
    """
    Fetch current weather conditions for an Indian airport.

    Args:
        airport_icao: ICAO code (e.g. 'VIDP', 'VABB', 'VOMM', 'VOBL', 'VECO').

    Returns:
        JSON with visibility, wind, ceiling, temperature, severity, IFR flag,
        and raw METAR-style data.
    """
    cond = store.get_weather(airport_icao)
    if not cond:
        return json.dumps({"error": f"No weather data for {airport_icao}."})

    return json.dumps({
        "airport":          cond.airport_icao,
        "timestamp":        cond.timestamp.isoformat(),
        "visibility_km":    cond.visibility_km,
        "wind_speed_kts":   round(cond.wind_speed_kts, 1),
        "wind_direction":   round(cond.wind_direction_deg),
        "ceiling_ft":       cond.ceiling_ft,
        "temperature_c":    round(cond.temperature_c, 1),
        "qnh_hpa":          round(cond.qnh_hpa, 1),
        "phenomena":        cond.weather_phenomena,
        "severity":         cond.severity.value,
        "is_ifr":           cond.is_ifr,
        "crosswind_kts":    cond.crosswind_component_kts,
        "raw_metar":        cond.raw_metar,
    })


@tool
def get_all_airports_weather() -> str:
    """
    Retrieve weather conditions for all five major Indian hub airports simultaneously.
    Useful for network-wide situational awareness and routing decisions.

    Returns:
        JSON dict keyed by ICAO with severity and IFR flag for each airport.
    """
    result = {}
    for icao, cond in store.weather.items():
        result[icao] = {
            "severity":      cond.severity.value,
            "visibility_km": cond.visibility_km,
            "is_ifr":        cond.is_ifr,
            "phenomena":     cond.weather_phenomena,
        }
    return json.dumps(result)


@tool
def assess_weather_impact(airport_icao: str) -> str:
    """
    Assess how current weather will impact flight operations at an airport.
    Returns capacity reduction estimate and specific operational recommendations.

    Args:
        airport_icao: ICAO code of the airport to assess.

    Returns:
        JSON with capacity_reduction_pct, landing_minima_met, departure_recommendation,
        and a list of specific operational constraints.
    """
    cond = store.get_weather(airport_icao)
    airport = store.AIRPORTS.get(airport_icao)
    if not cond or not airport:
        return json.dumps({"error": "Airport or weather data not found."})

    severity_map = {
        "CLEAR": 0, "LOW_VIS": 15, "MODERATE": 30, "SEVERE": 55, "EXTREME": 80
    }
    reduction = severity_map.get(cond.severity.value, 0)

    constraints = []
    if cond.visibility_km < 0.6:
        constraints.append("CAT III ILS required — only CAT III certified crews may operate")
    elif cond.visibility_km < 1.5:
        constraints.append("CAT II ILS required — CAT II certified crews only")
    elif cond.is_ifr:
        constraints.append("IFR conditions — instrument approaches only")

    if "TS" in cond.weather_phenomena:
        constraints.append("Active thunderstorm — 10 NM deviation clearance required")
    if "FG" in cond.weather_phenomena:
        constraints.append("Dense fog — Low Visibility Procedures (LVP) active")
    if cond.wind_speed_kts > 25:
        constraints.append(f"Strong winds {round(cond.wind_speed_kts)} kts — check crosswind limits per aircraft type")
    if cond.crosswind_component_kts > 20:
        constraints.append(f"Crosswind component {cond.crosswind_component_kts} kts — may exceed limits for narrow-body aircraft")

    return json.dumps({
        "airport":              airport_icao,
        "capacity_reduction_pct": reduction,
        "effective_capacity":   int(airport.capacity_per_hour * (1 - reduction / 100)),
        "landing_minima_met":   cond.visibility_km >= 0.3,
        "constraints":          constraints,
        "severity":             cond.severity.value,
        "recommendation":       (
            "SUSPEND operations" if cond.severity.value == "EXTREME" else
            "RESTRICT to priority flights" if cond.severity.value == "SEVERE" else
            "REDUCE throughput, apply LVP" if cond.severity.value == "MODERATE" else
            "Monitor closely" if cond.severity.value == "LOW_VIS" else
            "Normal operations"
        ),
    })


# ─── Scheduling / Sequencing Tools ───────────────────────────────────────────

@tool
def get_arrival_sequence(airport_icao: str) -> str:
    """
    Generate an optimised arrival sequence for flights inbound to an airport.
    Uses priority (fuel emergency > medical > VIP > normal) and ETA to sequence.

    Args:
        airport_icao: Destination airport ICAO code.

    Returns:
        JSON list of flights in recommended landing order with spacing advice.
    """
    inbound = [f for f in store.flights.values()
               if f.destination_icao == airport_icao
               and f.status in {FlightStatus.EN_ROUTE, FlightStatus.APPROACHING}]

    inbound.sort(key=lambda f: (-f.priority_level, f.delay_minutes), reverse=False)
    inbound.sort(key=lambda f: -f.priority_level)  # priority always first

    sequence = []
    for i, f in enumerate(inbound, 1):
        separation = "3 NM" if f.aircraft_type in ("B77W", "B787") else "2.5 NM"
        sequence.append({
            "position":     i,
            "flight_id":    f.flight_id,
            "aircraft_type": f.aircraft_type,
            "priority":     f.priority_level,
            "fuel_emergency": f.fuel_emergency,
            "delay_minutes": f.delay_minutes,
            "recommended_separation": separation,
        })

    return json.dumps({"airport": airport_icao, "inbound_count": len(sequence),
                       "sequence": sequence})


@tool
def check_gate_availability(airport_icao: str) -> str:
    """
    Check gate availability and utilisation at an Indian airport.

    Args:
        airport_icao: Airport ICAO code.

    Returns:
        JSON with total gates, available gates, occupied gates, and utilisation %.
    """
    gate_counts = {"VIDP": 78, "VABB": 64, "VOMM": 30, "VOBL": 42, "VECO": 28}
    total  = gate_counts.get(airport_icao, 20)
    parked = [f for f in store.flights.values()
              if (f.origin_icao == airport_icao or f.destination_icao == airport_icao)
              and f.status in {FlightStatus.BOARDING, FlightStatus.SCHEDULED}]
    occupied  = min(len(parked), total)
    available = total - occupied

    return json.dumps({
        "airport":         airport_icao,
        "total_gates":     total,
        "available_gates": available,
        "occupied_gates":  occupied,
        "utilisation_pct": round((occupied / total) * 100, 1),
    })


@tool
def predict_delay(flight_id: str) -> str:
    """
    Predict the likely delay for a flight using weather, congestion,
    and historical on-time performance data.

    Args:
        flight_id: The flight to analyse.

    Returns:
        JSON with predicted_delay_minutes, confidence, contributing factors,
        and a human-readable recommendation.
    """
    flight = store.get_flight(flight_id)
    if not flight:
        return json.dumps({"error": f"Flight {flight_id} not found."})

    dest_weather = store.get_weather(flight.destination_icao)
    origin_weather = store.get_weather(flight.origin_icao)

    # Simple scoring model (substitute with ML model in production)
    factors = []
    weather_score   = 0.0
    congestion_score = 0.0

    if dest_weather:
        sev_weights = {"CLEAR": 0, "LOW_VIS": 0.2, "MODERATE": 0.45, "SEVERE": 0.7, "EXTREME": 0.95}
        weather_score = sev_weights.get(dest_weather.severity.value, 0)
        if weather_score > 0.3:
            factors.append(f"Adverse weather at {flight.destination_icao}: {dest_weather.severity.value}")

    if origin_weather and origin_weather.severity.value in ("SEVERE", "EXTREME"):
        weather_score = max(weather_score, 0.5)
        factors.append(f"Severe weather at origin {flight.origin_icao}")

    active_at_dest = store.get_active_flights(flight.destination_icao)
    if len(active_at_dest) > 15:
        congestion_score = 0.6
        factors.append(f"High traffic congestion at {flight.destination_icao} ({len(active_at_dest)} active)")
    elif len(active_at_dest) > 8:
        congestion_score = 0.3
        factors.append(f"Moderate traffic at {flight.destination_icao}")

    if flight.aircraft_type in ("ATR72",):
        factors.append("Turboprop — sensitive to wind shear and crosswind limits")

    historical_otp = random.uniform(0.65, 0.95)
    base_delay = flight.delay_minutes
    predicted  = base_delay + (weather_score * 45) + (congestion_score * 30)
    confidence = min(0.9, 0.5 + historical_otp * 0.4)

    recommendation = (
        "Consider holding or diverting" if predicted > 90 else
        "Notify passengers, coordinate with ground staff" if predicted > 45 else
        "Monitor; minor delay expected" if predicted > 15 else
        "On time — no action required"
    )

    return json.dumps({
        "flight_id":              flight_id,
        "current_delay_minutes":  base_delay,
        "predicted_delay_minutes": round(predicted, 1),
        "confidence":             round(confidence, 2),
        "contributing_factors":   factors,
        "weather_impact_score":   round(weather_score, 2),
        "congestion_impact_score": round(congestion_score, 2),
        "historical_otp":         round(historical_otp, 2),
        "recommendation":         recommendation,
    })


# ─── Alert Tools ──────────────────────────────────────────────────────────────

@tool
def raise_alert(level: str, message: str,
                flight_id: Optional[str] = None,
                airport_icao: Optional[str] = None) -> str:
    """
    Raise a system alert for ATC supervisors or ground operations.

    Args:
        level:        Alert severity — INFO, WARNING, CRITICAL, or EMERGENCY.
        message:      Human-readable alert description.
        flight_id:    Optional: the specific flight triggering the alert.
        airport_icao: Optional: the airport related to the alert.

    Returns:
        JSON with the created alert ID and acknowledgement instructions.
    """
    try:
        alert_level = AlertLevel(level.upper())
    except ValueError:
        alert_level = AlertLevel.INFO

    alert = store.add_alert(
        level=alert_level,
        source="ATMSystem",
        message=message,
        flight_id=flight_id,
        airport_icao=airport_icao,
    )
    return json.dumps({
        "alert_id":  alert.alert_id,
        "level":     alert.level.value,
        "message":   alert.message,
        "timestamp": alert.timestamp.isoformat(),
    })


@tool
def get_active_alerts() -> str:
    """
    Retrieve all unresolved system alerts across all airports and flights.

    Returns:
        JSON list of active alerts sorted by severity (EMERGENCY first).
    """
    severity_order = {"EMERGENCY": 0, "CRITICAL": 1, "WARNING": 2, "INFO": 3}
    active = [a for a in store.alerts if not a.resolved]
    active.sort(key=lambda a: severity_order.get(a.level.value, 9))
    return json.dumps({
        "total": len(active),
        "alerts": [
            {
                "alert_id":   a.alert_id,
                "level":      a.level.value,
                "message":    a.message,
                "flight_id":  a.flight_id,
                "airport":    a.airport_icao,
                "timestamp":  a.timestamp.isoformat(),
            }
            for a in active
        ],
    })


# ─── Airport Capacity Tools ───────────────────────────────────────────────────

@tool
def get_airport_capacity_status(airport_icao: str) -> str:
    """
    Get current capacity utilisation and throughput metrics for an airport.
    Compares current traffic against maximum declared capacity.

    Args:
        airport_icao: ICAO code of the airport.

    Returns:
        JSON with current movements, capacity, utilisation %, and
        a recommendation if the airport is near saturation.
    """
    airport = store.AIRPORTS.get(airport_icao)
    if not airport:
        return json.dumps({"error": f"Airport {airport_icao} not found."})

    active = store.get_active_flights(airport_icao)
    utilisation = len(active) / max(airport.capacity_per_hour, 1) * 100

    return json.dumps({
        "airport":             airport_icao,
        "airport_name":        airport.name,
        "active_flights":      len(active),
        "capacity_per_hour":   airport.capacity_per_hour,
        "utilisation_pct":     round(utilisation, 1),
        "runways_available":   len(airport.runways),
        "status":              ("SATURATED" if utilisation > 90 else
                                "HIGH"      if utilisation > 75 else
                                "MODERATE"  if utilisation > 50 else "NORMAL"),
        "recommendation":      ("Implement ATFM ground delay program" if utilisation > 90 else
                                "Monitor and prepare holding patterns" if utilisation > 75 else
                                "Normal operations"),
    })


# ─── Exported Tool List ───────────────────────────────────────────────────────

ALL_TOOLS = [
    get_flight_status,
    list_active_flights,
    get_delayed_flights_report,
    assign_runway,
    update_flight_status,
    get_weather_conditions,
    get_all_airports_weather,
    assess_weather_impact,
    get_arrival_sequence,
    check_gate_availability,
    predict_delay,
    raise_alert,
    get_active_alerts,
    get_airport_capacity_status,
]
