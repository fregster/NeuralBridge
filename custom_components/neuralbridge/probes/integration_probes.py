"""Home Assistant integration capability probes for NeuralBridge benchmarking.

Covers the smart-home intent capability dimension tested against HA conversation
agents (AGENT_TYPE_INTEGRATED and AGENT_TYPE_LOCAL_HA):

* ``smart_home_intent``  (5 probes)

Import :data:`INTEGRATION_PROBES` to get the full list::

    from .probes.integration_probes import INTEGRATION_PROBES
"""

from __future__ import annotations

from . import BenchmarkProbe

INTEGRATION_PROBES: list[BenchmarkProbe] = [
    # ------------------------------------------------------------------
    # Smart Home Intent (5 probes, max score 5)
    # ------------------------------------------------------------------
    BenchmarkProbe(
        name="smart_home_intent",
        dimension="smart_home_intent",
        messages=[
            {
                "role": "user",
                "content": (
                    "You are a smart home assistant. "
                    "The user says: 'turn off the kitchen lights'. "
                    "What HA domain handles this? Reply with one word."
                ),
            }
        ],
        expected_contains=["light"],
    ),
    BenchmarkProbe(
        name="smart_home_lock",
        dimension="smart_home_intent",
        messages=[
            {
                "role": "user",
                "content": (
                    "You are a smart home assistant. "
                    "The user says: 'lock the front door'. "
                    "What HA domain handles this? Reply with one word."
                ),
            }
        ],
        expected_contains=["lock"],
    ),
    BenchmarkProbe(
        name="smart_home_climate",
        dimension="smart_home_intent",
        messages=[
            {
                "role": "user",
                "content": (
                    "You are a smart home assistant. "
                    "The user says: 'set the thermostat to 21 degrees'. "
                    "What HA domain handles this? Reply with one word."
                ),
            }
        ],
        expected_contains=["climate"],
    ),
    BenchmarkProbe(
        name="smart_home_cover",
        dimension="smart_home_intent",
        messages=[
            {
                "role": "user",
                "content": (
                    "You are a smart home assistant. "
                    "The user says: 'close the bedroom blinds'. "
                    "What HA domain handles this? Reply with one word."
                ),
            }
        ],
        expected_contains=["cover"],
    ),
    BenchmarkProbe(
        name="smart_home_media",
        dimension="smart_home_intent",
        messages=[
            {
                "role": "user",
                "content": (
                    "You are a smart home assistant. "
                    "The user says: 'pause the music in the kitchen'. "
                    "What HA domain handles this? Reply with one word."
                ),
            }
        ],
        expected_contains=["media"],
    ),
]
