"""
agents/atm_agents.py
Four specialised ReAct agents for Indian Airport Traffic Management:
  1. RunwayAgent          – runway allocation & sequencing
  2. WeatherAgent         – weather assessment & advisories
  3. SchedulingAgent      – delay prediction & flight scheduling
  4. SupervisorAgent      – system-wide oversight, alert triage, escalation
"""
from __future__ import annotations

import os
from textwrap import dedent
from typing import Optional

from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tools.atm_tools import (
    ALL_TOOLS,
    assign_runway, assess_weather_impact, check_gate_availability,
    get_active_alerts, get_airport_capacity_status, get_all_airports_weather,
    get_arrival_sequence, get_delayed_flights_report, get_flight_status,
    get_weather_conditions, list_active_flights, predict_delay,
    raise_alert, update_flight_status,
)


# ─── Shared LLM ──────────────────────────────────────────────────────────────

def _build_llm(temperature: float = 0.1) -> ChatGroq:
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY", ""),
    )

# ─── ReAct Prompt Factory ─────────────────────────────────────────────────────

_REACT_TEMPLATE = dedent("""
You are {agent_name}, a specialised AI agent in the Indian Air Traffic Management (ATM) system.

{role_description}

You have access to the following tools:
{tools}

Your tool names are: {tool_names}

REASONING PROTOCOL:
- Think step-by-step before taking any action.
- Always fetch current data before making a decision — never assume.
- For safety-critical actions (runway assignment, emergency alerts), reason explicitly about risks.
- Follow DGCA (Directorate General of Civil Aviation) regulations.
- Prioritise: Emergency > Fuel critical > Medical > VIP > Normal traffic.
- If weather is SEVERE or EXTREME, raise an alert immediately before any other action.

Use this format STRICTLY:

Question: the input task you must accomplish
Thought: reason about what you need to do
Action: the tool name to call (must be one of {tool_names})
Action Input: the input to the tool
Observation: the result of the tool call
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have enough information to provide a final answer
Final Answer: your complete, structured response

Begin!

Question: {input}
Thought:{agent_scratchpad}
""").strip()


def _make_prompt(agent_name: str, role_description: str) -> PromptTemplate:
    return PromptTemplate(
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        partial_variables={"agent_name": agent_name, "role_description": role_description},
        template=_REACT_TEMPLATE,
    )


# ─── 1. Runway Agent ─────────────────────────────────────────────────────────

RUNWAY_TOOLS = [
    get_flight_status,
    list_active_flights,
    assign_runway,
    get_arrival_sequence,
    get_weather_conditions,
    assess_weather_impact,
    get_airport_capacity_status,
    raise_alert,
]

_RUNWAY_ROLE = dedent("""
ROLE: You manage runway allocation and arrival/departure sequencing for Indian airports.
RESPONSIBILITIES:
- Assign runways to arriving and departing aircraft based on wind, aircraft type, and ILS category.
- Sequence inbound traffic using FCFS modified by priority (emergency/fuel always first).
- Detect runway conflicts and resolve them before they become safety issues.
- Enforce wake turbulence separation (Heavy → Medium: min 3 NM, Heavy → Light: 4 NM).
- Close runways for inspection or adverse weather and reroute affected traffic.
CONSTRAINTS:
- Never assign a closed or occupied runway.
- Check crosswind component against aircraft type limits before assignment.
- Always verify ILS category when visibility < 1500 m.
""").strip()


def build_runway_agent(verbose: bool = True) -> AgentExecutor:
    llm    = _build_llm()
    prompt = _make_prompt("RunwayAgent", _RUNWAY_ROLE)
    agent  = create_react_agent(llm, RUNWAY_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=RUNWAY_TOOLS,
        verbose=verbose,
        max_iterations=int(os.getenv("MAX_AGENT_ITERATIONS", 10)),
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


# ─── 2. Weather Agent ─────────────────────────────────────────────────────────

WEATHER_TOOLS = [
    get_weather_conditions,
    get_all_airports_weather,
    assess_weather_impact,
    raise_alert,
    get_active_alerts,
    list_active_flights,
    update_flight_status,
]

_WEATHER_ROLE = dedent("""
ROLE: You are the meteorological intelligence agent for Indian airport operations.
RESPONSIBILITIES:
- Monitor weather across all five Indian hub airports (VIDP, VABB, VOMM, VOBL, VECO).
- Issue Low Visibility Procedures (LVP) advisories when visibility drops below 1500 m.
- Alert operations when thunderstorms (TS), dense fog (FG), or extreme winds are detected.
- Recommend diversion airports when conditions deteriorate below operating minima.
- Assess crosswind components and advise on preferred runway configuration.
- Correlate weather impact with expected delay propagation across the network.
CONSTRAINTS:
- Any severity SEVERE or EXTREME triggers an immediate CRITICAL alert.
- IFR conditions must be communicated to all inbound flights within 300 NM.
- Always cite raw METAR when issuing weather advisories.
""").strip()


def build_weather_agent(verbose: bool = True) -> AgentExecutor:
    llm    = _build_llm(temperature=0.05)
    prompt = _make_prompt("WeatherAgent", _WEATHER_ROLE)
    agent  = create_react_agent(llm, WEATHER_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=WEATHER_TOOLS,
        verbose=verbose,
        max_iterations=int(os.getenv("MAX_AGENT_ITERATIONS", 10)),
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


# ─── 3. Scheduling Agent ──────────────────────────────────────────────────────

SCHEDULING_TOOLS = [
    get_flight_status,
    list_active_flights,
    get_delayed_flights_report,
    predict_delay,
    update_flight_status,
    check_gate_availability,
    get_airport_capacity_status,
    get_weather_conditions,
    raise_alert,
]

_SCHEDULING_ROLE = dedent("""
ROLE: You optimise flight schedules and manage delay propagation across the Indian aviation network.
RESPONSIBILITIES:
- Predict delays using weather, congestion, and historical on-time performance data.
- Prioritise runway/gate slots for flights with high delay penalties or tight connections.
- Coordinate turnaround times with ground operations.
- Implement ATFM (Air Traffic Flow Management) ground delay programmes during congestion.
- Notify downstream airports of expected arrival changes.
- Suggest slot swaps to minimise total passenger delay across the network.
CONSTRAINTS:
- Minimum ground time: 25 min for narrow-body, 45 min for wide-body aircraft.
- Medical and medevac flights always receive the next available slot.
- Curfew airports (e.g. VABB runway 14/32 after 23:00 IST) must be respected.
""").strip()


def build_scheduling_agent(verbose: bool = True) -> AgentExecutor:
    llm    = _build_llm()
    prompt = _make_prompt("SchedulingAgent", _SCHEDULING_ROLE)
    agent  = create_react_agent(llm, SCHEDULING_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=SCHEDULING_TOOLS,
        verbose=verbose,
        max_iterations=int(os.getenv("MAX_AGENT_ITERATIONS", 10)),
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


# ─── 4. Supervisor Agent ──────────────────────────────────────────────────────

SUPERVISOR_TOOLS = ALL_TOOLS  # Supervisor has access to everything

_SUPERVISOR_ROLE = dedent("""
ROLE: You are the senior AI supervisor overseeing all ATM agents and the entire Indian airspace.
RESPONSIBILITIES:
- Monitor system-wide alerts and escalate when human intervention is required.
- Coordinate multi-airport flow management during network disruptions.
- Arbitrate conflicts between runway, weather, and scheduling decisions.
- Provide executive-level situational awareness reports to the ATC Director.
- Trigger emergency protocols (MAYDAY, PAN-PAN) handling procedures.
- Generate post-incident reports and recommend process improvements.
AUTHORITY:
- Can override any other agent's decision in a safety-critical situation.
- Direct authority to suspend operations at any airport if safety is compromised.
- Only the Supervisor can declare a network-wide Ground Stop.
ESCALATION TRIGGERS:
- Any EMERGENCY alert → immediately notify human ATC Director.
- Airport utilisation > 95% → activate flow control measures.
- Three or more CRITICAL alerts simultaneously → convene crisis protocol.
""").strip()


def build_supervisor_agent(verbose: bool = True) -> AgentExecutor:
    llm    = _build_llm(temperature=0.0)  # Zero temperature for deterministic safety decisions
    prompt = _make_prompt("SupervisorAgent", _SUPERVISOR_ROLE)
    agent  = create_react_agent(llm, SUPERVISOR_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=SUPERVISOR_TOOLS,
        verbose=verbose,
        max_iterations=15,  # Supervisor gets extra iterations for complex reasoning
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


# ─── Multi-Agent Dispatcher ───────────────────────────────────────────────────

class ATMAgentSystem:
    """
    Top-level coordinator that routes queries to the appropriate specialist agent.
    Falls back to the SupervisorAgent for cross-cutting or ambiguous tasks.
    """

    def __init__(self, verbose: bool = True):
        self.runway_agent     = build_runway_agent(verbose)
        self.weather_agent    = build_weather_agent(verbose)
        self.scheduling_agent = build_scheduling_agent(verbose)
        self.supervisor_agent = build_supervisor_agent(verbose)

    def _route(self, query: str) -> tuple[AgentExecutor, str]:
        q = query.lower()
        if any(k in q for k in ("runway", "land", "takeoff", "sequence", "separation")):
            return self.runway_agent, "RunwayAgent"
        if any(k in q for k in ("weather", "metar", "fog", "storm", "visibility", "wind", "ifr")):
            return self.weather_agent, "WeatherAgent"
        if any(k in q for k in ("delay", "schedule", "gate", "turnaround", "slot", "capacity")):
            return self.scheduling_agent, "SchedulingAgent"
        return self.supervisor_agent, "SupervisorAgent"

    def run(self, query: str, force_agent: Optional[str] = None) -> dict:
        """
        Execute a query through the appropriate agent.

        Args:
            query:       Natural language task description.
            force_agent: Optional override: 'runway', 'weather', 'scheduling', 'supervisor'.

        Returns:
            dict with 'agent', 'output', and 'steps' keys.
        """
        agent_map = {
            "runway":     (self.runway_agent,     "RunwayAgent"),
            "weather":    (self.weather_agent,    "WeatherAgent"),
            "scheduling": (self.scheduling_agent, "SchedulingAgent"),
            "supervisor": (self.supervisor_agent, "SupervisorAgent"),
        }
        if force_agent and force_agent in agent_map:
            executor, name = agent_map[force_agent]
        else:
            executor, name = self._route(query)

        result = executor.invoke({"input": query})
        return {
            "agent":  name,
            "output": result.get("output", ""),
            "steps":  result.get("intermediate_steps", []),
        }

    async def arun(self, query: str, force_agent: Optional[str] = None) -> dict:
        """Async version for FastAPI / WebSocket use."""
        agent_map = {
            "runway":     (self.runway_agent,     "RunwayAgent"),
            "weather":    (self.weather_agent,    "WeatherAgent"),
            "scheduling": (self.scheduling_agent, "SchedulingAgent"),
            "supervisor": (self.supervisor_agent, "SupervisorAgent"),
        }
        if force_agent and force_agent in agent_map:
            executor, name = agent_map[force_agent]
        else:
            executor, name = self._route(query)

        result = await executor.ainvoke({"input": query})
        return {
            "agent":  name,
            "output": result.get("output", ""),
            "steps":  result.get("intermediate_steps", []),
        }
