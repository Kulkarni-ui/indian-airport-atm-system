"""
main.py
Entry point for the Indian Airport ATM System.
Supports three modes:
  1. demo    – Run the terminal dashboard with live simulated data (no API key needed)
  2. api     – Start the FastAPI server (REST + WebSocket)
  3. agent   – Interactive ReAct agent REPL (requires GROQ_API_KEY)
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def run_demo():
    from dashboard.terminal_dashboard import run_dashboard
    print("\n🛫  Starting Indian Airport ATM Terminal Dashboard...")
    print("   Press Ctrl+C to exit.\n")
    run_dashboard()


def run_api():
    import uvicorn
    print("\n🌐  Starting ATM FastAPI server on http://localhost:8000")
    print("   Dashboard: http://localhost:8000/dashboard")
    print("   API docs:  http://localhost:8000/docs\n")
    uvicorn.run("dashboard.api:app", host="0.0.0.0", port=8000, reload=True)


def run_agent_repl():
    if not os.getenv("GROQ_API_KEY"):
        print("❌  GROQ_API_KEY not set. Add it to .env")
        sys.exit(1)

    from agents.atm_agents import ATMAgentSystem
    import data.store as store
    from data.store import seed_data

    seed_data(num_flights=20)
    atm = ATMAgentSystem(verbose=True)

    print("\n✈  Indian Airport ATM Agent REPL")
    print("   Type your query in natural language. Type 'exit' to quit.")
    print("   Examples:")
    print("     - 'Check weather at all airports and raise alerts for any severe conditions'")
    print("     - 'What is the arrival sequence for VIDP right now?'")
    print("     - 'List all delayed flights and predict how much worse the delays will get'")
    print("     - 'Assign runway VIDP-28 to the highest priority inbound flight'\n")

    while True:
        try:
            query = input("ATM> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if query.lower() in ("exit", "quit", "q"):
            break
        if not query:
            continue

        print(f"\n🤖 Routing to agent...\n")
        result = atm.run(query)
        print(f"\n[{result['agent']}] Final Answer:\n{result['output']}\n")
        print("─" * 60)


def run_train_model():
    from models.delay_predictor import train_model
    print("\n🧠  Training delay prediction model...")
    metrics = train_model()
    print(f"   MAE:           {metrics['mae_minutes']} minutes")
    print(f"   Within 15 min: {metrics['within_15min_pct']}%")
    print(f"   Training set:  {metrics['training_samples']} samples")
    print("   Model saved to models/delay_model.pkl\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Indian Airport ATM System – Agentic AI with LangChain & ReAct"
    )
    parser.add_argument(
        "mode",
        choices=["demo", "api", "agent", "train"],
        help=(
            "demo  – Terminal dashboard (no API key needed)\n"
            "api   – FastAPI server\n"
            "agent – Interactive agent REPL (needs GROQ_API_KEY)\n"
            "train – Train delay prediction ML model"
        ),
    )
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()
    elif args.mode == "api":
        run_api()
    elif args.mode == "agent":
        run_agent_repl()
    elif args.mode == "train":
        run_train_model()