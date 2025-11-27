#!/usr/bin/env python3
"""Advanced 3D Fall Detection Demo - Predictive Version.

Features:
- PRE-FALL DETECTION: Detects anomalous patterns BEFORE fall occurs
- Alert levels: NORMAL → ANOMALY → PREDICTION → ALERT
- Realistic radar wave physics with reflection/blocking
- Signal strength visualization based on person detection
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from enum import Enum

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


# ============================================================================
# Utility Functions
# ============================================================================

def draw_sphere(radius, slices=16, stacks=16):
    """Draw a sphere."""
    for i in range(stacks):
        lat0 = math.pi * (-0.5 + float(i) / stacks)
        z0 = radius * math.sin(lat0)
        zr0 = radius * math.cos(lat0)
        lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
        z1 = radius * math.sin(lat1)
        zr1 = radius * math.cos(lat1)
        glBegin(GL_QUAD_STRIP)
        for j in range(slices + 1):
            lng = 2 * math.pi * float(j) / slices
            x, y = math.cos(lng), math.sin(lng)
            glNormal3f(x * zr0, z0, y * zr0)
            glVertex3f(x * zr0, z0, y * zr0)
            glNormal3f(x * zr1, z1, y * zr1)
            glVertex3f(x * zr1, z1, y * zr1)
        glEnd()


def draw_cube(size=1.0):
    """Draw a cube."""
    s = size / 2
    faces = [
        ((0, 0, -1), [(-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s)]),
        ((0, 0, 1), [(-s, -s, s), (-s, s, s), (s, s, s), (s, -s, s)]),
        ((0, -1, 0), [(-s, -s, -s), (-s, -s, s), (s, -s, s), (s, -s, -s)]),
        ((0, 1, 0), [(-s, s, -s), (s, s, -s), (s, s, s), (-s, s, s)]),
        ((-1, 0, 0), [(-s, -s, -s), (-s, s, -s), (-s, s, s), (-s, -s, s)]),
        ((1, 0, 0), [(s, -s, -s), (s, -s, s), (s, s, s), (s, s, -s)]),
    ]
    glBegin(GL_QUADS)
    for normal, verts in faces:
        glNormal3fv(normal)
        for v in verts:
            glVertex3fv(v)
    glEnd()


def draw_cylinder(radius, height, slices=16):
    """Draw a cylinder."""
    glBegin(GL_QUAD_STRIP)
    for i in range(slices + 1):
        angle = 2 * math.pi * i / slices
        x, z = radius * math.cos(angle), radius * math.sin(angle)
        glNormal3f(math.cos(angle), 0, math.sin(angle))
        glVertex3f(x, 0, z)
        glVertex3f(x, height, z)
    glEnd()


# ============================================================================
# Enums and Data Classes
# ============================================================================

class AlertLevel(Enum):
    NORMAL = 0
    ANOMALY_DETECTED = 1  # 異常步態偵測
    FALL_PREDICTED = 2    # 跌倒風險預測
    FALL_ALERT = 3        # 跌倒警報


class PersonState(Enum):
    IDLE = "idle"
    WALKING = "walking"
    UNSTABLE = "unstable"      # 步態不穩
    STUMBLING = "stumbling"    # 蹣跚
    FALLING = "falling"
    FALLEN = "fallen"
    RECOVERING = "recovering"


@dataclass
class Person:
    id: str
    name: str
    x: float
    z: float
    color: Tuple[float, float, float]
    state: PersonState = PersonState.IDLE
    target_x: float = 0.0
    target_z: float = 0.0
    rotation: float = 0.0
    fall_progress: float = 0.0
    stumble_time: float = 0.0
    walk_cycle: float = 0.0
    speed: float = 0.5
    # Prediction data
    anomaly_score: float = 0.0
    fall_risk: float = 0.0

    def __post_init__(self):
        self.target_x = self.x
        self.target_z = self.z


@dataclass
class RadarWave:
    """Represents a radar wave pulse."""
    origin_x: float
    origin_z: float
    radius: float = 0.0
    max_radius: float = 15.0
    speed: float = 8.0
    strength: float = 1.0
    reflected: bool = False
    reflection_angle: float = 0.0
    hit_person: str = ""


@dataclass
class ScenarioEvent:
    time: float
    action: str
    target: str
    params: Dict = field(default_factory=dict)
    narration: str = ""


# ============================================================================
# Main Demo Class
# ============================================================================

class PredictiveFallDemo:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.width = 1280
        self.height = 720
        self.running = True

        # Time
        self.time = 0.0
        self.scenario_time = 0.0
        self.demo_started = False

        # Camera
        self.camera_x = 10.0
        self.camera_y = 12.0
        self.camera_z = 18.0
        self.camera_target = [10.0, 0.0, 6.0]
        self.camera_shake = 0.0

        # Facility
        self.facility_width = 20.0
        self.facility_depth = 12.5

        # Radar system
        self.radar_x = 10.0
        self.radar_z = 0.5
        self.radar_waves: List[RadarWave] = []
        self.wave_interval = 0.3
        self.last_wave_time = 0.0
        self.detected_signals: List[Dict] = []

        # Alert system
        self.alert_level = AlertLevel.NORMAL
        self.alert_target = ""
        self.alert_message = ""
        self.alert_time = 0.0
        self.prediction_confidence = 0.0

        # Statistics
        self.anomaly_count = 0
        self.prediction_count = 0
        self.fall_count = 0

        # People
        self.people = [
            Person("wang", "王阿公", 3.0, 4.0, (0.9, 0.7, 0.5)),
            Person("li", "李阿嬤", 8.0, 9.0, (0.85, 0.65, 0.5)),
            Person("chen", "陳伯伯", 14.0, 4.0, (0.8, 0.6, 0.45)),
            Person("nurse", "護理師", 16.0, 10.0, (0.95, 0.95, 0.95)),
        ]

        # Zones
        self.zones = [
            {"id": "corridor", "rect": (0, 0, 20, 2.5), "color": (0.3, 0.35, 0.4), "name": "走廊"},
            {"id": "activity", "rect": (0, 2.5, 10, 5), "color": (0.35, 0.4, 0.35), "name": "活動區"},
            {"id": "rest", "rect": (10, 2.5, 10, 5), "color": (0.4, 0.35, 0.35), "name": "休息區"},
            {"id": "rehab", "rect": (0, 7.5, 10, 5), "color": (0.35, 0.35, 0.45), "name": "復健室"},
            {"id": "nurse_station", "rect": (10, 7.5, 10, 5), "color": (0.4, 0.4, 0.4), "name": "護理站"},
        ]

        # Scenario events - with prediction flow
        self.scenario_events = [
            ScenarioEvent(0.5, "narration", "", {}, "系統啟動中...雷達波開始掃描"),
            ScenarioEvent(2.0, "walk", "wang", {"target_x": 8.0, "target_z": 4.0, "speed": 0.6},
                         "王阿公開始在活動區走動"),
            ScenarioEvent(6.0, "anomaly", "wang", {"score": 0.3},
                         "⚠ 偵測到輕微步態異常"),
            ScenarioEvent(8.0, "unstable", "wang", {},
                         "⚠ 步態不穩定度上升"),
            ScenarioEvent(10.0, "anomaly", "wang", {"score": 0.6},
                         "⚠⚠ 異常指數升高，持續監控中"),
            ScenarioEvent(12.0, "predict", "wang", {"risk": 0.75},
                         "🔴 預測跌倒風險：75% - 通知照護人員"),
            ScenarioEvent(14.0, "stumble", "wang", {"duration": 3.0},
                         "⚠⚠⚠ 偵測到蹣跚步態"),
            ScenarioEvent(16.0, "predict", "wang", {"risk": 0.92},
                         "🚨 跌倒風險極高：92% - 緊急通知！"),
            ScenarioEvent(18.0, "fall", "wang", {},
                         "🚨🚨 跌倒發生！照護人員已收到通知"),
            ScenarioEvent(22.0, "respond", "nurse", {"target_x": 8.0, "target_z": 4.0},
                         "護理師收到通知，前往協助"),
            ScenarioEvent(28.0, "recover", "wang", {},
                         "護理師協助王阿公起身"),
            ScenarioEvent(32.0, "reset", "", {},
                         "演示結束 - 按 SPACE 重新開始"),
        ]
        self.current_event_index = 0
        self.current_narration = "按 [SPACE] 開始預測式跌倒偵測演示"
        self.narration_time = 0.0

        # Initialize pygame and OpenGL
        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("mmWave 毫米波跌倒預測系統")

        # OpenGL setup
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glLightfv(GL_LIGHT0, GL_POSITION, (10, 20, 10, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.35, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.75, 1))

        glClearColor(0.1, 0.1, 0.12, 1)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

        # Font
        pygame.font.init()
        CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        try:
            self.font_large = pygame.font.Font(CJK_FONT, 32)
            self.font_medium = pygame.font.Font(CJK_FONT, 22)
            self.font_small = pygame.font.Font(CJK_FONT, 16)
        except:
            self.font_large = pygame.font.Font(None, 36)
            self.font_medium = pygame.font.Font(None, 28)
            self.font_small = pygame.font.Font(None, 20)

    def emit_radar_wave(self):
        """Emit a new radar wave from the radar position."""
        wave = RadarWave(
            origin_x=self.radar_x,
            origin_z=self.radar_z,
            radius=0.1,
            max_radius=18.0,
            speed=6.0,
            strength=1.0
        )
        self.radar_waves.append(wave)

    def update_radar_waves(self, dt):
        """Update radar waves and check for person detection."""
        self.detected_signals.clear()
        waves_to_remove = []

        for wave in self.radar_waves:
            wave.radius += wave.speed * dt
            wave.strength = max(0, 1.0 - (wave.radius / wave.max_radius))

            # Check collision with people
            if not wave.reflected:
                for person in self.people:
                    dist = math.sqrt((person.x - wave.origin_x)**2 + (person.z - wave.origin_z)**2)
                    if abs(dist - wave.radius) < 0.5:
                        # Wave hit person - create reflection
                        angle = math.atan2(person.z - wave.origin_z, person.x - wave.origin_x)
                        signal = {
                            "person": person,
                            "distance": dist,
                            "angle": angle,
                            "strength": wave.strength * 0.8,
                            "time": self.time
                        }
                        self.detected_signals.append(signal)

                        # Create reflected wave
                        if wave.strength > 0.3:
                            reflected = RadarWave(
                                origin_x=person.x,
                                origin_z=person.z,
                                radius=0.1,
                                max_radius=5.0,
                                speed=4.0,
                                strength=wave.strength * 0.5,
                                reflected=True,
                                hit_person=person.id
                            )
                            self.radar_waves.append(reflected)

            if wave.radius > wave.max_radius:
                waves_to_remove.append(wave)

        for wave in waves_to_remove:
            self.radar_waves.remove(wave)

        # Emit new waves
        if self.time - self.last_wave_time > self.wave_interval:
            self.emit_radar_wave()
            self.last_wave_time = self.time

    def update_scenario(self, dt):
        """Update scenario events."""
        if not self.demo_started:
            return

        self.scenario_time += dt

        while (self.current_event_index < len(self.scenario_events) and
               self.scenario_events[self.current_event_index].time <= self.scenario_time):
            event = self.scenario_events[self.current_event_index]
            self._execute_event(event)
            self.current_event_index += 1

    def _execute_event(self, event: ScenarioEvent):
        """Execute a scenario event."""
        if event.narration:
            self.current_narration = event.narration
            self.narration_time = self.time

        if event.action == "narration":
            pass  # Just show narration

        elif event.action == "walk":
            for p in self.people:
                if p.id == event.target:
                    p.state = PersonState.WALKING
                    p.target_x = event.params.get("target_x", p.x)
                    p.target_z = event.params.get("target_z", p.z)
                    p.speed = event.params.get("speed", 0.5)
                    break

        elif event.action == "anomaly":
            for p in self.people:
                if p.id == event.target:
                    p.anomaly_score = event.params.get("score", 0.3)
                    self.alert_level = AlertLevel.ANOMALY_DETECTED
                    self.alert_target = p.id
                    self.alert_time = self.time
                    self.anomaly_count += 1
                    break

        elif event.action == "unstable":
            for p in self.people:
                if p.id == event.target:
                    p.state = PersonState.UNSTABLE
                    break

        elif event.action == "predict":
            for p in self.people:
                if p.id == event.target:
                    p.fall_risk = event.params.get("risk", 0.5)
                    self.alert_level = AlertLevel.FALL_PREDICTED
                    self.prediction_confidence = p.fall_risk
                    self.alert_target = p.id
                    self.alert_time = self.time
                    self.prediction_count += 1
                    break

        elif event.action == "stumble":
            for p in self.people:
                if p.id == event.target:
                    p.state = PersonState.STUMBLING
                    p.stumble_time = event.params.get("duration", 2.0)
                    break

        elif event.action == "fall":
            for p in self.people:
                if p.id == event.target:
                    p.state = PersonState.FALLING
                    p.fall_progress = 0.0
                    self.alert_level = AlertLevel.FALL_ALERT
                    self.alert_target = p.id
                    self.alert_time = self.time
                    self.camera_shake = 0.5
                    self.fall_count += 1
                    break

        elif event.action == "respond":
            for p in self.people:
                if p.id == event.target:
                    p.state = PersonState.WALKING
                    p.target_x = event.params.get("target_x", p.x)
                    p.target_z = event.params.get("target_z", p.z)
                    p.speed = 1.2  # Hurrying
                    break

        elif event.action == "recover":
            for p in self.people:
                if p.id == event.target:
                    p.state = PersonState.RECOVERING
                    self.alert_level = AlertLevel.NORMAL
                    break

        elif event.action == "reset":
            self._reset_demo()

    def _reset_demo(self):
        """Reset demo to initial state."""
        self.scenario_time = 0.0
        self.current_event_index = 0
        self.demo_started = False
        self.alert_level = AlertLevel.NORMAL
        self.alert_target = ""
        self.current_narration = "按 [SPACE] 開始預測式跌倒偵測演示"
        self.radar_waves.clear()

        # Reset people
        positions = {"wang": (3.0, 4.0), "li": (8.0, 9.0), "chen": (14.0, 4.0), "nurse": (16.0, 10.0)}
        for p in self.people:
            p.state = PersonState.IDLE
            p.fall_progress = 0.0
            p.anomaly_score = 0.0
            p.fall_risk = 0.0
            if p.id in positions:
                p.x, p.z = positions[p.id]
                p.target_x, p.target_z = positions[p.id]

    def update_people(self, dt):
        """Update all people."""
        for person in self.people:
            if person.state == PersonState.WALKING:
                dx = person.target_x - person.x
                dz = person.target_z - person.z
                dist = math.sqrt(dx*dx + dz*dz)
                if dist > 0.1:
                    person.x += (dx / dist) * person.speed * dt
                    person.z += (dz / dist) * person.speed * dt
                    person.rotation = math.degrees(math.atan2(dx, dz))
                    person.walk_cycle += dt * 5
                else:
                    person.state = PersonState.IDLE

            elif person.state == PersonState.UNSTABLE:
                # Wobbly walking
                person.walk_cycle += dt * 4
                wobble = math.sin(person.walk_cycle * 3) * 0.3
                person.x += wobble * dt

            elif person.state == PersonState.STUMBLING:
                person.stumble_time -= dt
                person.walk_cycle += dt * 8
                if person.stumble_time <= 0:
                    person.state = PersonState.UNSTABLE

            elif person.state == PersonState.FALLING:
                person.fall_progress = min(1.0, person.fall_progress + dt * 1.5)
                if person.fall_progress >= 1.0:
                    person.state = PersonState.FALLEN

            elif person.state == PersonState.RECOVERING:
                person.fall_progress = max(0.0, person.fall_progress - dt * 0.5)
                if person.fall_progress <= 0:
                    person.state = PersonState.IDLE

        # Update camera shake
        if self.camera_shake > 0:
            self.camera_shake = max(0, self.camera_shake - dt * 0.5)

    def draw_floor(self):
        """Draw facility floor with zones."""
        glDisable(GL_LIGHTING)

        # Base floor
        glColor4f(0.2, 0.2, 0.22, 1.0)
        glBegin(GL_QUADS)
        glVertex3f(-1, -0.02, -1)
        glVertex3f(self.facility_width + 1, -0.02, -1)
        glVertex3f(self.facility_width + 1, -0.02, self.facility_depth + 1)
        glVertex3f(-1, -0.02, self.facility_depth + 1)
        glEnd()

        # Zones
        for zone in self.zones:
            x, z, w, d = zone["rect"]
            r, g, b = zone["color"]
            glColor4f(r, g, b, 1.0)
            glBegin(GL_QUADS)
            glVertex3f(x + 0.05, 0, z + 0.05)
            glVertex3f(x + w - 0.05, 0, z + 0.05)
            glVertex3f(x + w - 0.05, 0, z + d - 0.05)
            glVertex3f(x + 0.05, 0, z + d - 0.05)
            glEnd()

        # Grid lines
        glColor4f(0.3, 0.3, 0.32, 1.0)
        glLineWidth(1)
        glBegin(GL_LINES)
        for i in range(int(self.facility_width) + 1):
            glVertex3f(i, 0.01, 0)
            glVertex3f(i, 0.01, self.facility_depth)
        for j in range(int(self.facility_depth) + 1):
            glVertex3f(0, 0.01, j)
            glVertex3f(self.facility_width, 0.01, j)
        glEnd()

        glEnable(GL_LIGHTING)

    def draw_radar(self):
        """Draw radar unit."""
        glPushMatrix()
        glTranslatef(self.radar_x, 0, self.radar_z)

        # Radar base
        glColor4f(0.2, 0.3, 0.5, 1.0)
        glPushMatrix()
        glScalef(0.8, 0.3, 0.4)
        draw_cube(1.0)
        glPopMatrix()

        # Radar antenna
        glColor4f(0.3, 0.4, 0.6, 1.0)
        glTranslatef(0, 0.3, 0)
        glPushMatrix()
        glScalef(0.4, 0.2, 0.3)
        draw_cube(1.0)
        glPopMatrix()

        # Active indicator
        pulse = (math.sin(self.time * 5) + 1) * 0.5
        glColor4f(0.2, 0.8, 0.3, pulse)
        glTranslatef(0, 0.15, 0.2)
        draw_sphere(0.05, 8, 8)

        glPopMatrix()

    def draw_radar_waves(self):
        """Draw radar waves with realistic physics."""
        glDisable(GL_LIGHTING)
        glLineWidth(2)

        for wave in self.radar_waves:
            if wave.reflected:
                # Reflected waves - different color
                alpha = wave.strength * 0.6
                glColor4f(0.9, 0.6, 0.2, alpha)
            else:
                # Primary waves
                alpha = wave.strength * 0.4
                glColor4f(0.2, 0.7, 0.9, alpha)

            # Draw wave arc (only forward facing for non-reflected)
            glBegin(GL_LINE_STRIP)
            start_angle = -math.pi / 2 if not wave.reflected else 0
            end_angle = math.pi / 2 if not wave.reflected else 2 * math.pi
            segments = 32

            for i in range(segments + 1):
                angle = start_angle + (end_angle - start_angle) * i / segments
                x = wave.origin_x + wave.radius * math.cos(angle)
                z = wave.origin_z + wave.radius * math.sin(angle)
                glVertex3f(x, 0.1, z)
            glEnd()

        # Draw detected signal indicators
        for signal in self.detected_signals:
            person = signal["person"]
            strength = signal["strength"]

            # Signal return line
            glColor4f(0.9, 0.5, 0.2, strength)
            glBegin(GL_LINES)
            glVertex3f(self.radar_x, 0.2, self.radar_z)
            glVertex3f(person.x, 0.5, person.z)
            glEnd()

            # Detection ring around person
            glColor4f(0.9, 0.7, 0.2, strength * 0.5)
            glBegin(GL_LINE_LOOP)
            for i in range(16):
                angle = 2 * math.pi * i / 16
                x = person.x + 0.8 * math.cos(angle)
                z = person.z + 0.8 * math.sin(angle)
                glVertex3f(x, 0.1, z)
            glEnd()

        glEnable(GL_LIGHTING)

    def draw_person(self, person: Person):
        """Draw a person with state-based appearance."""
        glPushMatrix()
        glTranslatef(person.x, 0, person.z)

        # Calculate pose based on state
        if person.state == PersonState.FALLING:
            # Falling animation
            fall_angle = person.fall_progress * 90
            glRotatef(fall_angle, 0, 0, 1)
            glTranslatef(person.fall_progress * 0.5, 0, 0)
        elif person.state == PersonState.FALLEN:
            glRotatef(90, 0, 0, 1)
            glTranslatef(0.5, 0, 0)
        elif person.state == PersonState.STUMBLING:
            wobble = math.sin(person.walk_cycle) * 15
            glRotatef(wobble, 0, 0, 1)
        elif person.state == PersonState.UNSTABLE:
            wobble = math.sin(person.walk_cycle * 2) * 8
            glRotatef(wobble, 0, 0, 1)

        # Body color based on alert state
        r, g, b = person.color
        if person.id == self.alert_target:
            if self.alert_level == AlertLevel.ANOMALY_DETECTED:
                # Yellow tint
                r, g, b = min(1, r + 0.3), min(1, g + 0.2), max(0, b - 0.2)
            elif self.alert_level == AlertLevel.FALL_PREDICTED:
                # Orange tint
                r, g, b = min(1, r + 0.4), max(0, g - 0.1), max(0, b - 0.3)
            elif self.alert_level == AlertLevel.FALL_ALERT:
                # Red pulse
                pulse = (math.sin(self.time * 8) + 1) * 0.3
                r, g, b = min(1, 0.9 + pulse), 0.2, 0.2

        glColor4f(r, g, b, 1.0)

        # Body
        glPushMatrix()
        glTranslatef(0, 0.6, 0)
        glScalef(0.35, 0.5, 0.2)
        draw_cube(1.0)
        glPopMatrix()

        # Head
        glPushMatrix()
        glTranslatef(0, 1.15, 0)
        draw_sphere(0.18, 12, 12)
        glPopMatrix()

        # Legs with walking animation
        leg_swing = math.sin(person.walk_cycle) * 0.3 if person.state in [PersonState.WALKING, PersonState.UNSTABLE] else 0
        for side in [-1, 1]:
            glPushMatrix()
            glTranslatef(side * 0.1, 0.25, side * leg_swing * 0.1)
            glScalef(0.1, 0.5, 0.1)
            draw_cube(1.0)
            glPopMatrix()

        # Arms
        for side in [-1, 1]:
            glPushMatrix()
            glTranslatef(side * 0.25, 0.7, 0)
            glScalef(0.08, 0.4, 0.08)
            draw_cube(1.0)
            glPopMatrix()

        glPopMatrix()

        # Anomaly/Risk indicator above person
        if person.anomaly_score > 0 or person.fall_risk > 0:
            self._draw_risk_indicator(person)

    def _draw_risk_indicator(self, person: Person):
        """Draw risk indicator above person."""
        glDisable(GL_LIGHTING)
        glPushMatrix()
        glTranslatef(person.x, 2.0, person.z)

        # Risk bar background
        bar_width = 0.8
        bar_height = 0.15

        # Determine color and fill based on risk
        if person.fall_risk > 0:
            fill = person.fall_risk
            if person.fall_risk > 0.8:
                r, g, b = 1.0, 0.2, 0.2  # Red
            elif person.fall_risk > 0.5:
                r, g, b = 1.0, 0.6, 0.2  # Orange
            else:
                r, g, b = 1.0, 1.0, 0.3  # Yellow
        else:
            fill = person.anomaly_score
            r, g, b = 1.0, 1.0, 0.3  # Yellow

        # Background
        glColor4f(0.2, 0.2, 0.2, 0.8)
        glBegin(GL_QUADS)
        glVertex3f(-bar_width/2, 0, 0)
        glVertex3f(bar_width/2, 0, 0)
        glVertex3f(bar_width/2, bar_height, 0)
        glVertex3f(-bar_width/2, bar_height, 0)
        glEnd()

        # Fill
        glColor4f(r, g, b, 0.9)
        glBegin(GL_QUADS)
        glVertex3f(-bar_width/2, 0, 0.01)
        glVertex3f(-bar_width/2 + bar_width * fill, 0, 0.01)
        glVertex3f(-bar_width/2 + bar_width * fill, bar_height, 0.01)
        glVertex3f(-bar_width/2, bar_height, 0.01)
        glEnd()

        glPopMatrix()
        glEnable(GL_LIGHTING)

    def draw_alert_overlay(self):
        """Draw alert status overlay."""
        if self.alert_level == AlertLevel.NORMAL:
            return

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        # Screen border effect
        if self.alert_level == AlertLevel.FALL_ALERT:
            pulse = (math.sin(self.time * 6) + 1) * 0.3
            glColor4f(1.0, 0.2, 0.2, pulse)
        elif self.alert_level == AlertLevel.FALL_PREDICTED:
            pulse = (math.sin(self.time * 4) + 1) * 0.2
            glColor4f(1.0, 0.5, 0.2, pulse)
        else:
            pulse = (math.sin(self.time * 3) + 1) * 0.15
            glColor4f(1.0, 1.0, 0.3, pulse)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def draw_hud(self):
        """Draw HUD overlay."""
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        # Create surface for text
        hud = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # Top bar
        pygame.draw.rect(hud, (20, 25, 35, 230), (0, 0, self.width, 65))

        # Title
        title = self.font_medium.render("mmWave 毫米波跌倒預測系統", True, (200, 200, 220))
        hud.blit(title, (20, 18))

        # Alert status indicator
        if self.alert_level == AlertLevel.NORMAL:
            status_color = (100, 200, 100)
            status_text = "● 正常監控中"
        elif self.alert_level == AlertLevel.ANOMALY_DETECTED:
            status_color = (255, 255, 100)
            status_text = "⚠ 異常步態偵測"
        elif self.alert_level == AlertLevel.FALL_PREDICTED:
            status_color = (255, 150, 50)
            status_text = f"🔴 跌倒風險預測 {int(self.prediction_confidence*100)}%"
        else:
            status_color = (255, 80, 80)
            status_text = "🚨 跌倒警報！"

        status = self.font_medium.render(status_text, True, status_color)
        hud.blit(status, (self.width - status.get_width() - 30, 18))

        # Stats bar
        stats_y = 45
        stats = [
            f"異常偵測: {self.anomaly_count}",
            f"風險預測: {self.prediction_count}",
            f"跌倒警報: {self.fall_count}",
        ]
        x_pos = 20
        for stat in stats:
            text = self.font_small.render(stat, True, (150, 150, 160))
            hud.blit(text, (x_pos, stats_y))
            x_pos += text.get_width() + 30

        # Bottom narration bar
        if self.current_narration:
            pygame.draw.rect(hud, (20, 25, 35, 220), (0, self.height - 55, self.width, 55))

            # Narration text
            narration = self.font_medium.render(self.current_narration, True, (220, 220, 230))
            hud.blit(narration, (20, self.height - 40))

        # Controls hint
        hint = self.font_small.render("[SPACE] 開始/重置  [F] 觸發跌倒  [ESC] 退出", True, (100, 100, 110))
        hud.blit(hint, (self.width - hint.get_width() - 20, self.height - 35))

        # Convert to OpenGL texture
        texture_data = pygame.image.tostring(hud, "RGBA", True)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glEnable(GL_TEXTURE_2D)
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)

        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(self.width, 0)
        glTexCoord2f(1, 1); glVertex2f(self.width, self.height)
        glTexCoord2f(0, 1); glVertex2f(0, self.height)
        glEnd()

        glDeleteTextures([texture_id])
        glDisable(GL_TEXTURE_2D)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    if not self.demo_started:
                        self.demo_started = True
                        self.scenario_time = 0.0
                        self.current_event_index = 0
                    else:
                        self._reset_demo()
                        self.demo_started = True
                elif event.key == pygame.K_f:
                    # Manual fall trigger
                    for p in self.people:
                        if p.state not in [PersonState.FALLING, PersonState.FALLEN]:
                            p.state = PersonState.FALLING
                            p.fall_progress = 0.0
                            self.alert_level = AlertLevel.FALL_ALERT
                            self.alert_target = p.id
                            self.camera_shake = 0.5
                            self.fall_count += 1
                            self.current_narration = f"🚨 {p.name} 跌倒警報！"
                            break

    def run(self):
        """Main loop."""
        clock = pygame.time.Clock()
        last_time = time.time()

        while self.running:
            current_time = time.time()
            dt = min(current_time - last_time, 0.1)
            last_time = current_time
            self.time += dt

            self.handle_events()
            self.update_scenario(dt)
            self.update_people(dt)
            self.update_radar_waves(dt)

            # Render
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Camera with shake
            shake_x = random.uniform(-1, 1) * self.camera_shake * 0.3
            shake_y = random.uniform(-1, 1) * self.camera_shake * 0.3

            glLoadIdentity()
            gluLookAt(
                self.camera_x + shake_x, self.camera_y, self.camera_z + shake_y,
                self.camera_target[0], self.camera_target[1], self.camera_target[2],
                0, 1, 0
            )

            self.draw_floor()
            self.draw_radar()
            self.draw_radar_waves()

            for person in self.people:
                self.draw_person(person)

            self.draw_alert_overlay()
            self.draw_hud()

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()

    demo = PredictiveFallDemo(api_url=args.api_url)
    demo.run()


if __name__ == "__main__":
    main()
