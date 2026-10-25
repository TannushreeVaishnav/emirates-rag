"""
tools.py — Mock tool used by the LangGraph flight-status node.

In a production system this would call the Emirates real-time API. For this
project I use a JSON file with a handful of hard-coded flights so I can test
the tool-routing branch of the graph without needing an API key or live data.

The flight_status_reply() function is called by the flight_status_node in
graph.py when the router classifies a question as "flight_status". It extracts
the flight number from the user's message with a simple regex, looks it up in
the mock data, and returns a human-readable status string.
"""

import json
import re
from pathlib import Path

from src.config import ROOT

# Location of the mock flight data (checked in to the repo for demos)
_MOCK_PATH = ROOT / "data" / "mock" / "flights.json"

# Regex to extract IATA flight numbers like EK501, EK0501, QR123
_FLIGHT_RE = re.compile(r"\b([A-Z]{2}\d{3,4})\b")


def _load_mock() -> dict:
    """
    Load the mock flight data from JSON.

    The file stores an array of flight objects; this function converts it to a
    dict keyed by flight number so lookups are O(1).
    Returns an empty dict if the file is missing.
    """
    if not _MOCK_PATH.exists():
        return {}
    data = json.loads(_MOCK_PATH.read_text(encoding="utf-8"))
    # Support both array format [{flight: "EK501", ...}] and dict format {"EK501": {...}}
    if isinstance(data, list):
        return {item["flight"]: item for item in data if "flight" in item}
    return data  # already a dict


def flight_status_reply(question: str) -> str:
    """
    Return a plain-text flight status for the flight number mentioned in the question.

    Args:
        question: The user's question, e.g. "What is the status of EK501?"

    Returns:
        A formatted status string, or a "not found" message if the flight
        number isn't in the mock data.

    Note:
        All responses are prefixed with "[Demo data]" so users know this is
        not connected to the live Emirates system.
    """
    flights = _load_mock()
    match = _FLIGHT_RE.search(question.upper())

    if not match:
        return (
            "[Demo data] I couldn't find a flight number in your question. "
            "Please include the full flight number, for example EK501."
        )

    flight_no = match.group(1)
    info = flights.get(flight_no)

    if not info:
        return (
            f"[Demo data] No information found for flight {flight_no}. "
            "For live status, visit emirates.com or the Emirates app."
        )

    # Build a readable summary from whatever fields the mock data contains
    parts = [f"[Demo data] Flight {flight_no}"]
    if info.get("status"):
        parts.append(f"Status: {info['status']}")
    if info.get("departure"):
        parts.append(f"Departure: {info['departure']}")
    if info.get("arrival"):
        parts.append(f"Arrival: {info['arrival']}")
    if info.get("gate"):
        parts.append(f"Gate: {info['gate']}")
    if info.get("note"):
        parts.append(info["note"])

    return " · ".join(parts)
