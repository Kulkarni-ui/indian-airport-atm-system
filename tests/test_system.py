"""
tests/test_system.py
Unit and integration tests for the Indian Airport ATM system.
Run with: pytest tests/ -v
"""
from __future__ import annotations

import json
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import data.store as store
from data.store import AIRPORTS, RUNWAYS, seed_data
from models.schemas import FlightStatus, WeatherSeverity


@pytest.fixture(autouse=True)
def fresh_store():
    """Reset store state before each test."""
    store.flights.clear()
    store.weather.clear()
    store.alerts.clear()
    store.decisions_log.clear()
    seed_data(num_flights=10)
    yield


# ─── Tool Tests ───────────────────────────────────────────────────────────────

class TestFlightTools:
    def test_list_active_flights_returns_data(self):
        from tools.atm_tools import list_active_flights
        result = json.loads(list_active_flights.invoke({"airport_icao": None}))
        assert "flights" in result
        assert isinstance(result["flights"], list)

    def test_get_flight_status_known_flight(self):
        from tools.atm_tools import get_flight_status
        flight_id = list(store.flights.keys())[0]
        result = json.loads(get_flight_status.invoke({"flight_id": flight_id}))
        assert result.get("flight_id") == flight_id
        assert "status" in result

    def test_get_flight_status_unknown(self):
        from tools.atm_tools import get_flight_status
        result = json.loads(get_flight_status.invoke({"flight_id": "XX9999"}))
        assert "error" in result

    def test_get_delayed_flights_report(self):
        from tools.atm_tools import get_delayed_flights_report
        result = json.loads(get_delayed_flights_report.invoke({}))
        assert "total_delayed" in result
        assert isinstance(result["flights"], list)

    def test_update_flight_status(self):
        from tools.atm_tools import update_flight_status
        flight_id = list(store.flights.keys())[0]
        result = json.loads(update_flight_status.invoke({
            "flight_id": flight_id, "new_status": "LANDED", "delay_minutes": 10
        }))
        assert result["success"] is True
        assert store.flights[flight_id].status == FlightStatus.LANDED

    def test_update_flight_invalid_status(self):
        from tools.atm_tools import update_flight_status
        flight_id = list(store.flights.keys())[0]
        result = json.loads(update_flight_status.invoke({
            "flight_id": flight_id, "new_status": "FLYING_UPSIDE_DOWN"
        }))
        assert "error" in result


class TestRunwayTools:
    def test_assign_runway_success(self):
        from tools.atm_tools import assign_runway
        flight_id = list(store.flights.keys())[0]
        result = json.loads(assign_runway.invoke({
            "flight_id": flight_id, "runway_id": "VIDP-28"
        }))
        assert result["success"] is True
        assert store.flights[flight_id].assigned_runway == "28"

    def test_assign_runway_conflict(self):
        from tools.atm_tools import assign_runway
        flights = list(store.flights.keys())
        assign_runway.invoke({"flight_id": flights[0], "runway_id": "VIDP-28"})
        result = json.loads(assign_runway.invoke({"flight_id": flights[1], "runway_id": "VIDP-28"}))
        assert result["success"] is False
        assert "occupied" in result["error"].lower()

    def test_assign_unknown_runway(self):
        from tools.atm_tools import assign_runway
        flight_id = list(store.flights.keys())[0]
        result = json.loads(assign_runway.invoke({
            "flight_id": flight_id, "runway_id": "ZZZZ-99"
        }))
        assert result["success"] is False

    def test_arrival_sequence(self):
        from tools.atm_tools import get_arrival_sequence
        result = json.loads(get_arrival_sequence.invoke({"airport_icao": "VIDP"}))
        assert "sequence" in result
        if result["inbound_count"] > 1:
            positions = [s["position"] for s in result["sequence"]]
            assert positions == sorted(positions)


class TestWeatherTools:
    def test_get_weather_known_airport(self):
        from tools.atm_tools import get_weather_conditions
        result = json.loads(get_weather_conditions.invoke({"airport_icao": "VIDP"}))
        assert result["airport"] == "VIDP"
        assert "visibility_km" in result
        assert "severity" in result
        assert "is_ifr" in result

    def test_get_weather_unknown(self):
        from tools.atm_tools import get_weather_conditions
        result = json.loads(get_weather_conditions.invoke({"airport_icao": "ZZZZ"}))
        assert "error" in result

    def test_all_airports_weather(self):
        from tools.atm_tools import get_all_airports_weather
        result = json.loads(get_all_airports_weather.invoke({}))
        assert len(result) == 5
        for icao in ["VIDP", "VABB", "VOMM", "VOBL", "VECO"]:
            assert icao in result

    def test_assess_weather_impact(self):
        from tools.atm_tools import assess_weather_impact
        result = json.loads(assess_weather_impact.invoke({"airport_icao": "VABB"}))
        assert "capacity_reduction_pct" in result
        assert "recommendation" in result
        assert 0 <= result["capacity_reduction_pct"] <= 100


class TestAlertTools:
    def test_raise_and_retrieve_alert(self):
        from tools.atm_tools import raise_alert, get_active_alerts
        raise_alert.invoke({
            "level": "CRITICAL",
            "message": "Test critical alert",
            "airport_icao": "VIDP",
        })
        alerts = json.loads(get_active_alerts.invoke({}))
        assert alerts["total"] >= 1
        found = [a for a in alerts["alerts"] if "Test critical alert" in a["message"]]
        assert len(found) == 1

    def test_invalid_alert_level_defaults_info(self):
        from tools.atm_tools import raise_alert
        result = json.loads(raise_alert.invoke({
            "level": "SUPERCRITICAL", "message": "Test fallback"
        }))
        assert result["level"] == "INFO"


class TestCapacityTools:
    def test_capacity_status(self):
        from tools.atm_tools import get_airport_capacity_status
        result = json.loads(get_airport_capacity_status.invoke({"airport_icao": "VIDP"}))
        assert "utilisation_pct" in result
        assert result["status"] in ("NORMAL", "MODERATE", "HIGH", "SATURATED")

    def test_gate_availability(self):
        from tools.atm_tools import check_gate_availability
        result = json.loads(check_gate_availability.invoke({"airport_icao": "VIDP"}))
        assert result["total_gates"] == 78
        assert result["available_gates"] + result["occupied_gates"] == result["total_gates"]

    def test_predict_delay(self):
        from tools.atm_tools import predict_delay
        flight_id = list(store.flights.keys())[0]
        result = json.loads(predict_delay.invoke({"flight_id": flight_id}))
        assert "predicted_delay_minutes" in result
        assert 0.0 <= result["confidence"] <= 1.0


# ─── ML Model Tests ───────────────────────────────────────────────────────────

class TestDelayPredictor:
    def test_train_and_predict(self):
        from models.delay_predictor import train_model, predict
        metrics = train_model()
        assert metrics["mae_minutes"] < 20, "MAE too high — model may be broken"
        assert metrics["within_15min_pct"] > 50

        result = predict(
            dep_weather_severity="CLEAR",
            arr_weather_severity="MODERATE",
            wind_speed_kts=15.0,
            visibility_km=4.0,
            hour_utc=14,
            congestion_score=0.6,
            historical_otp=0.80,
            aircraft_type="A320",
        )
        assert result["predicted_delay_minutes"] >= 0
        assert result["lower_bound"] <= result["predicted_delay_minutes"] <= result["upper_bound"]


# ─── Data Store Tests ─────────────────────────────────────────────────────────

class TestDataStore:
    def test_seed_creates_flights(self):
        assert len(store.flights) == 10

    def test_seed_creates_weather_for_all_airports(self):
        for icao in AIRPORTS:
            assert icao in store.weather

    def test_add_and_retrieve_alert(self):
        from models.schemas import AlertLevel
        alert = store.add_alert(AlertLevel.WARNING, "TestAgent", "Test warning", airport_icao="VABB")
        assert alert.alert_id in [a.alert_id for a in store.alerts]

    def test_log_decision(self):
        store.log_decision({"agent": "TestAgent", "action": "test_action"})
        assert len(store.decisions_log) == 1
        assert store.decisions_log[0]["agent"] == "TestAgent"
