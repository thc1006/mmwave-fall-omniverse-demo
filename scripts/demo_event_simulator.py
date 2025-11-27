#!/usr/bin/env python3
"""Demo Event Simulator for mmWave Fall Detection System.

Generates realistic fall detection events for demonstration purposes.
Simulates:
- Normal activity (most common)
- Falls (with alert triggers)
- Bad posture during rehabilitation
- Chest/breathing abnormalities

Usage:
    python scripts/demo_event_simulator.py [--api-url URL] [--interval SECONDS]
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import requests

# Facility zone definitions based on 赤土崎多功能館 layout
ZONES = [
    {"id": "dementia_wandering_corridor", "rect": (0, 0, 12, 2.5), "fall_risk": "high"},
    {"id": "activity_room_a", "rect": (0, 2.5, 6, 5), "fall_risk": "medium"},
    {"id": "activity_room_b", "rect": (6, 2.5, 6, 5), "fall_risk": "medium"},
    {"id": "dining_hall", "rect": (12, 0, 8, 5), "fall_risk": "medium"},
    {"id": "rest_area", "rect": (0, 7.5, 6, 5), "fall_risk": "low"},
    {"id": "rehabilitation_room", "rect": (6, 7.5, 6, 5), "fall_risk": "high"},
    {"id": "nursing_station", "rect": (12, 7.5, 4, 2.5), "fall_risk": "low"},
    {"id": "bathroom_accessible", "rect": (16, 7.5, 4, 2.5), "fall_risk": "high"},
    {"id": "entrance_lobby", "rect": (12, 5, 8, 2.5), "fall_risk": "medium"},
]

# Event type probabilities
EVENT_PROBABILITIES = {
    "normal": 0.70,
    "fall": 0.15,
    "rehab_bad_posture": 0.10,
    "chest_abnormal": 0.05,
}


@dataclass
class SimulatedPerson:
    """Simulated person in the facility."""
    id: str
    name: str
    age: int
    current_zone: str
    position: Tuple[float, float]
    activity: str = "normal"
    fall_risk_factor: float = 1.0  # Higher = more likely to fall


class DemoEventSimulator:
    """Simulates realistic fall detection events."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.persons: List[SimulatedPerson] = []
        self._init_persons()

    def _init_persons(self) -> None:
        """Initialize simulated persons in the facility."""
        person_templates = [
            ("elder_01", "王阿公", 82, 1.5),
            ("elder_02", "李阿嬤", 78, 1.3),
            ("elder_03", "陳伯伯", 85, 1.8),
            ("elder_04", "林奶奶", 75, 1.2),
            ("elder_05", "張爺爺", 88, 2.0),
            ("caregiver_01", "護理師小美", 35, 0.3),
        ]

        for pid, name, age, risk in person_templates:
            zone = random.choice(ZONES)
            rect = zone["rect"]
            x = rect[0] + random.uniform(0.5, rect[2] - 0.5)
            y = rect[1] + random.uniform(0.5, rect[3] - 0.5)

            self.persons.append(SimulatedPerson(
                id=pid,
                name=name,
                age=age,
                current_zone=zone["id"],
                position=(x, y),
                fall_risk_factor=risk,
            ))

    def _get_zone_fall_risk(self, zone_id: str) -> float:
        """Get fall risk multiplier for a zone."""
        for zone in ZONES:
            if zone["id"] == zone_id:
                risk_map = {"high": 2.0, "medium": 1.0, "low": 0.5}
                return risk_map.get(zone["fall_risk"], 1.0)
        return 1.0

    def _generate_features(self, event_type: str) -> List[List[float]]:
        """Generate 256-dimensional feature vector for the event type."""
        np.random.seed(int(time.time() * 1000) % 2**32)

        # Base feature generation with patterns matching training data
        if event_type == "fall":
            # Fall signature: high acceleration spike, then stillness
            features = np.random.randn(256) * 2.0
            features[0:30] = np.random.randn(30) * 5.0  # Impact spike
            features[100:150] = np.random.randn(50) * 0.1  # Post-fall stillness
        elif event_type == "rehab_bad_posture":
            # Irregular movement patterns
            features = np.random.randn(256) * 1.5
            features[50:100] = np.sin(np.linspace(0, 10, 50)) * 3
        elif event_type == "chest_abnormal":
            # Abnormal breathing pattern
            features = np.random.randn(256) * 1.2
            features[0:50] = np.sin(np.linspace(0, 20, 50)) * 2  # Irregular breathing
        else:  # normal
            features = np.random.randn(256) * 0.5

        return [features.tolist()]

    def _move_person(self, person: SimulatedPerson) -> None:
        """Randomly move a person within the facility."""
        # 20% chance to move to a different zone
        if random.random() < 0.2:
            zone = random.choice(ZONES)
            person.current_zone = zone["id"]
            rect = zone["rect"]
            person.position = (
                rect[0] + random.uniform(0.5, rect[2] - 0.5),
                rect[1] + random.uniform(0.5, rect[3] - 0.5),
            )
        else:
            # Small movement within current zone
            for zone in ZONES:
                if zone["id"] == person.current_zone:
                    rect = zone["rect"]
                    dx = random.uniform(-1, 1)
                    dy = random.uniform(-1, 1)
                    new_x = max(rect[0] + 0.5, min(rect[0] + rect[2] - 0.5, person.position[0] + dx))
                    new_y = max(rect[1] + 0.5, min(rect[1] + rect[3] - 0.5, person.position[1] + dy))
                    person.position = (new_x, new_y)
                    break

    def _determine_event_type(self, person: SimulatedPerson) -> str:
        """Determine event type based on person and zone risk factors."""
        zone_risk = self._get_zone_fall_risk(person.current_zone)
        adjusted_fall_prob = min(0.5, EVENT_PROBABILITIES["fall"] * person.fall_risk_factor * zone_risk)

        # Caregivers rarely have issues
        if "caregiver" in person.id:
            return "normal"

        # Random event selection with adjusted probabilities
        r = random.random()
        if r < adjusted_fall_prob:
            return "fall"
        elif r < adjusted_fall_prob + EVENT_PROBABILITIES["rehab_bad_posture"]:
            if person.current_zone == "rehabilitation_room":
                return "rehab_bad_posture"
            return "normal"
        elif r < adjusted_fall_prob + EVENT_PROBABILITIES["rehab_bad_posture"] + EVENT_PROBABILITIES["chest_abnormal"]:
            return "chest_abnormal"
        return "normal"

    async def generate_event(self) -> dict:
        """Generate a single simulated event."""
        person = random.choice(self.persons)
        self._move_person(person)

        event_type = self._determine_event_type(person)
        features = self._generate_features(event_type)

        # Transform position to match frontend coordinate system
        # Map from facility coords to frontend coords (-10 to 10 range)
        x_mapped = (person.position[0] - 10) * 0.8
        y_mapped = (person.position[1] - 6.25) * 0.64

        payload = {
            "sequence": {"data": features},
            "position": {"x": x_mapped, "y": y_mapped},
        }

        try:
            response = requests.post(
                f"{self.api_url}/events/from_prediction",
                json=payload,
                timeout=5,
            )
            if response.status_code == 200:
                result = response.json()
                print(f"[{time.strftime('%H:%M:%S')}] Event: {result['label']:20s} | "
                      f"Person: {person.name:10s} | Zone: {person.current_zone:25s} | "
                      f"Pos: ({x_mapped:+.1f}, {y_mapped:+.1f})")
                return result
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            print(f"Request failed: {e}")
            return {}

    async def run(self, interval: float = 2.0) -> None:
        """Run the event simulator continuously."""
        print("=" * 70)
        print("mmWave Fall Detection Demo Event Simulator")
        print("=" * 70)
        print(f"API URL: {self.api_url}")
        print(f"Interval: {interval}s")
        print(f"Simulated persons: {len(self.persons)}")
        print("=" * 70)
        print()

        event_count = 0
        fall_count = 0

        while True:
            result = await self.generate_event()
            event_count += 1

            if result.get("label") == "fall":
                fall_count += 1
                print(f"  ⚠️  FALL ALERT! Total falls: {fall_count}/{event_count}")

            await asyncio.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Demo event simulator for mmWave fall detection")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API server URL")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between events")
    args = parser.parse_args()

    simulator = DemoEventSimulator(api_url=args.api_url)
    asyncio.run(simulator.run(interval=args.interval))


if __name__ == "__main__":
    main()
