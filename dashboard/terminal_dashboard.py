"""
dashboard/terminal_dashboard.py
A rich-terminal real-time dashboard for the Indian ATM system.
Run this standalone — no browser required.
Uses the Python `rich` library for live updating panels.
"""
from __future__ import annotations

import os
import sys
import time
import random
from datetime import datetime

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import data.store as store
from data.store import AIRPORTS, seed_data
from models.schemas import AlertLevel, FlightStatus, WeatherSeverity

console = Console()

SEVERITY_STYLE = {
    "CLEAR":    "bold green",
    "LOW_VIS":  "bold yellow",
    "MODERATE": "bold orange1",
    "SEVERE":   "bold red",
    "EXTREME":  "bold white on red",
}

STATUS_STYLE = {
    "SCHEDULED":   "cyan",
    "BOARDING":    "yellow",
    "DEPARTED":    "green",
    "EN_ROUTE":    "bold green",
    "APPROACHING": "bold yellow",
    "LANDED":      "dim green",
    "DIVERTED":    "bold red",
    "CANCELLED":   "red",
    "DELAYED":     "bold orange1",
}

ALERT_STYLE = {
    "INFO":      "blue",
    "WARNING":   "yellow",
    "CRITICAL":  "bold red",
    "EMERGENCY": "bold white on red",
}


def _simulate_tick() -> None:
    """Simulate small real-time changes to the data each tick."""
    for f in list(store.flights.values())[:5]:
        if f.status == FlightStatus.EN_ROUTE and f.altitude_ft:
            f.altitude_ft = max(0, f.altitude_ft + random.randint(-500, 500))
        if random.random() < 0.05:
            f.delay_minutes = max(0, f.delay_minutes + random.randint(-5, 10))
    for icao in store.weather:
        w = store.weather[icao]
        w.wind_speed_kts = max(0, w.wind_speed_kts + random.uniform(-1, 1))


def _make_header() -> Panel:
    now = datetime.now().strftime("%d %b %Y  %H:%M:%S IST")
    total   = len(store.flights)
    active  = len(store.get_active_flights())
    delayed = len(store.get_delayed_flights())
    alerts  = len([a for a in store.alerts if not a.resolved])
    text = Text()
    text.append("✈  INDIAN AIRPORT ATM SYSTEM  ", style="bold white")
    text.append(f"  {now}  ", style="dim white")
    text.append(f"  Total: {total}  ", style="cyan")
    text.append(f"Active: {active}  ", style="green")
    text.append(f"Delayed: {delayed}  ", style="yellow" if delayed < 5 else "bold red")
    text.append(f"Alerts: {alerts}", style="red" if alerts > 0 else "green")
    return Panel(text, style="bold blue", height=3)


def _make_weather_table() -> Table:
    t = Table(title="Weather – Major Hubs", box=box.ROUNDED, border_style="blue",
              header_style="bold cyan", show_lines=False)
    t.add_column("Airport", style="bold white", width=8)
    t.add_column("City",    width=12)
    t.add_column("Vis (km)", justify="right", width=9)
    t.add_column("Wind (kts)", justify="right", width=10)
    t.add_column("Severity", width=12)
    t.add_column("IFR", justify="center", width=5)

    for icao, w in store.weather.items():
        ap   = AIRPORTS.get(icao)
        city = ap.city if ap else icao
        sev  = w.severity.value
        ifr  = "[bold red]YES[/]" if w.is_ifr else "[green]No[/]"
        t.add_row(
            icao, city,
            f"{w.visibility_km:.1f}",
            f"{w.wind_speed_kts:.0f}",
            f"[{SEVERITY_STYLE.get(sev,'white')}]{sev}[/]",
            ifr,
        )
    return t


def _make_flights_table() -> Table:
    t = Table(title="Active Flights", box=box.ROUNDED, border_style="green",
              header_style="bold cyan", show_lines=False)
    t.add_column("Flight",   style="bold white", width=8)
    t.add_column("Aircraft", width=7)
    t.add_column("From",     width=6)
    t.add_column("To",       width=6)
    t.add_column("Status",   width=12)
    t.add_column("Delay",    justify="right", width=7)
    t.add_column("Alt (ft)", justify="right", width=9)
    t.add_column("Priority", justify="center", width=9)

    active = store.get_active_flights()
    active.sort(key=lambda f: (-f.priority_level, -f.delay_minutes))

    for f in active[:20]:
        status_style = STATUS_STYLE.get(f.status.value, "white")
        delay_str    = f"[bold red]{f.delay_minutes}m[/]" if f.delay_minutes > 30 else \
                       f"[yellow]{f.delay_minutes}m[/]" if f.delay_minutes > 0 else "[green]On time[/]"
        alt_str = f"{int(f.altitude_ft):,}" if f.altitude_ft else "—"
        prio    = f"[bold red]P{f.priority_level}[/]" if f.priority_level > 5 else \
                  f"[yellow]P{f.priority_level}[/]" if f.priority_level > 0 else "—"

        t.add_row(
            f.flight_id, f.aircraft_type,
            f.origin_icao, f.destination_icao,
            f"[{status_style}]{f.status.value}[/]",
            delay_str, alt_str, prio,
        )
    return t


def _make_alerts_panel() -> Panel:
    active_alerts = [a for a in store.alerts if not a.resolved]
    active_alerts.sort(key=lambda a: {"EMERGENCY": 0, "CRITICAL": 1, "WARNING": 2, "INFO": 3}.get(a.level.value, 9))

    if not active_alerts:
        content = Text("No active alerts — all systems nominal", style="dim green")
        return Panel(content, title="[bold]System Alerts[/]", border_style="green")

    text = Text()
    for a in active_alerts[:8]:
        style = ALERT_STYLE.get(a.level.value, "white")
        text.append(f"[{a.level.value}] ", style=style)
        msg = a.message[:60] + ("…" if len(a.message) > 60 else "")
        text.append(f"{msg}\n", style="white")

    return Panel(text, title=f"[bold red]System Alerts ({len(active_alerts)})[/]",
                 border_style="red")


def _make_capacity_panel() -> Panel:
    text = Text()
    for icao, airport in AIRPORTS.items():
        active  = store.get_active_flights(icao)
        util    = len(active) / max(airport.capacity_per_hour, 1) * 100
        bar_len = int(util / 5)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        color   = "red" if util > 90 else "yellow" if util > 75 else "green"
        text.append(f"{icao}  ", style="bold white")
        text.append(f"[{color}]{bar}[/]")
        text.append(f"  {util:5.1f}%  ({len(active)}/{airport.capacity_per_hour})\n", style="dim white")
    return Panel(text, title="[bold]Airport Capacity[/]", border_style="blue")


def run_dashboard(refresh_seconds: float = 2.0):
    """Start the live terminal dashboard."""
    seed_data(num_flights=30)

    # Add a few demo alerts
    store.add_alert(AlertLevel.WARNING,  "WeatherAgent", "Low visibility at VABB — LVP active", airport_icao="VABB")
    store.add_alert(AlertLevel.INFO,     "SchedulingAgent", "Flight AI-302 delayed 45 min due to congestion")
    store.add_alert(AlertLevel.CRITICAL, "SupervisorAgent", "Thunderstorm cell approaching VIDP — 15 NM")

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="bottom", size=12),
    )
    layout["main"].split_row(
        Layout(name="flights", ratio=2),
        Layout(name="right"),
    )
    layout["right"].split_column(
        Layout(name="weather"),
        Layout(name="capacity"),
    )

    with Live(layout, refresh_per_second=1, screen=True, console=console):
        while True:
            _simulate_tick()

            layout["header"].update(_make_header())
            layout["flights"].update(Panel(_make_flights_table(), border_style="green"))
            layout["weather"].update(_make_weather_table())
            layout["capacity"].update(_make_capacity_panel())
            layout["bottom"].update(_make_alerts_panel())

            time.sleep(refresh_seconds)


if __name__ == "__main__":
    try:
        run_dashboard()
    except KeyboardInterrupt:
        console.print("\n[bold green]ATM Dashboard stopped.[/]")