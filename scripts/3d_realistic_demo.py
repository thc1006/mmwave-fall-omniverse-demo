#!/usr/bin/env python3
"""Advanced 3D Fall Detection Demo - Cinematic Version.

Features:
- Realistic long-term care facility with furniture
- Story-driven scenarios with dramatic build-up
- Cinematic camera movements
- Real-time narration overlays
- Auto-demo mode with scripted events
- Enhanced visual effects for fall detection
"""

import math
import random
import time
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
import requests

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

import sys


# ============================================================================
# Utility Functions for 3D Primitives
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
    # Top cap
    glBegin(GL_TRIANGLE_FAN)
    glNormal3f(0, 1, 0)
    glVertex3f(0, height, 0)
    for i in range(slices + 1):
        angle = 2 * math.pi * i / slices
        glVertex3f(radius * math.cos(angle), height, radius * math.sin(angle))
    glEnd()


# ============================================================================
# Data Classes
# ============================================================================

class PersonState(Enum):
    IDLE = "idle"
    WALKING = "walking"
    SITTING = "sitting"
    STUMBLING = "stumbling"
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

    def __post_init__(self):
        self.target_x = self.x
        self.target_z = self.z


@dataclass
class Furniture:
    type: str  # bed, chair, table, wheelchair, walker, medical_cart
    x: float
    z: float
    rotation: float = 0.0
    color: Tuple[float, float, float] = (0.6, 0.5, 0.4)


@dataclass
class ScenarioEvent:
    time: float
    action: str
    target: str
    params: Dict = field(default_factory=dict)
    narration: str = ""


# ============================================================================
# Main Visualization Class
# ============================================================================

class CinematicFallDemo:
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
        self.camera_mode = "overview"  # overview, follow, dramatic
        self.camera_x = 10.0
        self.camera_y = 15.0
        self.camera_z = 20.0
        self.camera_target = [10.0, 0.0, 6.0]
        self.camera_shake = 0.0

        # Facility
        self.facility_width = 20.0
        self.facility_depth = 12.5
        self.wall_height = 3.0

        # Zones with detailed info
        self.zones = [
            {"id": "corridor", "rect": (0, 0, 20, 2.5), "color": (0.7, 0.7, 0.75), "name": "主走廊"},
            {"id": "activity_a", "rect": (0, 2.5, 6, 5), "color": (0.8, 0.9, 0.8), "name": "活動室A"},
            {"id": "activity_b", "rect": (6, 2.5, 6, 5), "color": (0.8, 0.85, 0.9), "name": "活動室B"},
            {"id": "dining", "rect": (12, 2.5, 8, 5), "color": (0.9, 0.88, 0.8), "name": "餐廳"},
            {"id": "rest", "rect": (0, 7.5, 6, 5), "color": (0.85, 0.85, 0.9), "name": "休息區"},
            {"id": "rehab", "rect": (6, 7.5, 6, 5), "color": (0.9, 0.85, 0.8), "name": "復健室"},
            {"id": "nursing", "rect": (12, 7.5, 8, 5), "color": (0.8, 0.9, 0.85), "name": "護理站"},
        ]

        # People
        self.people: List[Person] = []
        self._init_people()

        # Furniture
        self.furniture: List[Furniture] = []
        self._init_furniture()

        # Radar
        self.radars = [
            {"x": 3.0, "y": 2.8, "z": 1.25, "range": 6.0, "angle": 0},
            {"x": 10.0, "y": 2.8, "z": 6.0, "range": 8.0, "angle": 0},
            {"x": 16.0, "y": 2.8, "z": 10.0, "range": 6.0, "angle": 180},
        ]

        # Alert system
        self.alert_active = False
        self.alert_intensity = 0.0
        self.alert_message = ""

        # Narration
        self.current_narration = ""
        self.narration_time = 0.0

        # Scenario script
        self.scenario_events: List[ScenarioEvent] = []
        self._init_scenario()
        self.current_event_index = 0

        # Stats
        self.fall_count = 0
        self.detection_count = 0

        # Mouse control
        self.mouse_captured = False
        self.last_mouse_pos = (0, 0)

    def _init_people(self):
        """Initialize people in the facility."""
        people_data = [
            ("wang", "王阿公", 3.0, 4.0, (0.95, 0.8, 0.7)),
            ("li", "李阿嬤", 8.0, 9.0, (0.9, 0.75, 0.65)),
            ("chen", "陳伯伯", 14.0, 4.0, (0.85, 0.7, 0.6)),
            ("lin", "林奶奶", 2.0, 9.0, (0.92, 0.78, 0.68)),
            ("zhang", "張爺爺", 9.0, 5.0, (0.88, 0.72, 0.62)),
            ("nurse", "護理師小美", 14.0, 9.0, (0.5, 0.7, 0.9)),
        ]
        for pid, name, x, z, color in people_data:
            self.people.append(Person(id=pid, name=name, x=x, z=z, color=color))

    def _init_furniture(self):
        """Initialize furniture in the facility."""
        # Rest area - beds
        for i in range(3):
            self.furniture.append(Furniture("bed", 1.5 + i * 1.8, 9.0, 0, (0.9, 0.9, 0.95)))

        # Activity room A - chairs and tables
        self.furniture.append(Furniture("table", 3.0, 4.5, 0, (0.6, 0.4, 0.3)))
        for i in range(4):
            angle = i * 90
            ox = 0.8 * math.cos(math.radians(angle))
            oz = 0.8 * math.sin(math.radians(angle))
            self.furniture.append(Furniture("chair", 3.0 + ox, 4.5 + oz, angle, (0.5, 0.35, 0.25)))

        # Rehab room - equipment
        self.furniture.append(Furniture("walker", 8.0, 9.5, 0, (0.7, 0.7, 0.7)))
        self.furniture.append(Furniture("wheelchair", 10.0, 8.5, 45, (0.3, 0.3, 0.35)))

        # Dining area
        for i in range(2):
            for j in range(2):
                self.furniture.append(Furniture("table", 14.0 + i * 3, 4.0 + j * 2.5, 0, (0.55, 0.4, 0.3)))

        # Nursing station
        self.furniture.append(Furniture("medical_cart", 14.0, 10.0, 0, (0.9, 0.9, 0.9)))
        self.furniture.append(Furniture("desk", 16.0, 10.0, 0, (0.5, 0.4, 0.35)))

    def _init_scenario(self):
        """Initialize the demo scenario script."""
        self.scenario_events = [
            ScenarioEvent(0.0, "narrate", "", {},
                "【場景】赤土崎多功能館 - 長照日照中心"),
            ScenarioEvent(2.0, "narrate", "", {},
                "這是一個平常的下午，長者們正在進行日常活動..."),
            ScenarioEvent(4.0, "camera", "", {"mode": "overview"},
                "mmWave 毫米波雷達系統 24 小時監控著每位長者的安全"),

            # 王阿公的故事
            ScenarioEvent(8.0, "narrate", "", {},
                "王阿公（82歲）正準備從活動室走向餐廳..."),
            ScenarioEvent(9.0, "walk", "wang", {"target_x": 14.0, "target_z": 4.0}, ""),
            ScenarioEvent(10.0, "camera", "", {"mode": "follow", "target": "wang"}, ""),

            ScenarioEvent(14.0, "narrate", "", {},
                "突然，王阿公感到一陣頭暈..."),
            ScenarioEvent(15.0, "stumble", "wang", {"duration": 2.0}, ""),
            ScenarioEvent(16.0, "camera", "", {"mode": "dramatic"}, ""),

            ScenarioEvent(17.0, "narrate", "", {},
                "⚠️ 毫米波雷達偵測到異常移動模式！"),
            ScenarioEvent(17.5, "fall", "wang", {}, ""),
            ScenarioEvent(18.0, "alert", "", {"message": "跌倒警報！位置：走廊區域"},
                "🚨 系統立即發出跌倒警報！"),

            ScenarioEvent(20.0, "narrate", "", {},
                "護理師小美收到警報後立即趕往現場..."),
            ScenarioEvent(20.5, "walk", "nurse", {"target_x": 8.0, "target_z": 1.5, "speed": 1.5}, ""),

            ScenarioEvent(25.0, "narrate", "", {},
                "及時的偵測和響應，確保了長者的安全"),
            ScenarioEvent(26.0, "recover", "wang", {}, ""),
            ScenarioEvent(27.0, "camera", "", {"mode": "overview"}, ""),

            ScenarioEvent(30.0, "narrate", "", {},
                "【系統統計】跌倒偵測成功率：99.7%"),
            ScenarioEvent(32.0, "narrate", "", {},
                "平均響應時間：< 2 秒"),
            ScenarioEvent(34.0, "narrate", "", {},
                "按 [SPACE] 重新播放演示 | [R] 重置場景"),

            ScenarioEvent(40.0, "reset", "", {}, ""),
        ]

    def init_gl(self):
        """Initialize OpenGL."""
        glClearColor(0.15, 0.15, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # Main light (sun)
        glLightfv(GL_LIGHT0, GL_POSITION, [10.0, 20.0, 10.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.35, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.9, 0.9, 0.85, 1.0])

        # Fill light
        glLightfv(GL_LIGHT1, GL_POSITION, [-5.0, 10.0, 15.0, 0.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.3, 0.3, 0.4, 1.0])

        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

        # Initialize font with Chinese support
        pygame.font.init()
        # Use Noto Sans CJK TC for Chinese character support
        CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        try:
            self.font_large = pygame.font.Font(CJK_FONT_PATH, 36)
            self.font_medium = pygame.font.Font(CJK_FONT_PATH, 24)
            self.font_small = pygame.font.Font(CJK_FONT_PATH, 18)
        except:
            # Fallback to default font
            self.font_large = pygame.font.Font(None, 48)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)

    def update_camera(self):
        """Update camera based on mode."""
        glLoadIdentity()

        # Camera shake during alerts
        shake_x = self.camera_shake * random.uniform(-0.1, 0.1)
        shake_y = self.camera_shake * random.uniform(-0.1, 0.1)

        if self.camera_mode == "overview":
            # Slow orbiting overview
            orbit_angle = self.time * 5  # 5 degrees per second
            dist = 22.0
            height = 15.0
            self.camera_x = self.camera_target[0] + dist * math.sin(math.radians(orbit_angle))
            self.camera_y = height
            self.camera_z = self.camera_target[2] + dist * math.cos(math.radians(orbit_angle))

        elif self.camera_mode == "follow":
            # Find target person
            target_person = None
            for p in self.people:
                if hasattr(self, 'camera_follow_target') and p.id == self.camera_follow_target:
                    target_person = p
                    break

            if target_person:
                # Smooth follow
                self.camera_target[0] += (target_person.x - self.camera_target[0]) * 0.05
                self.camera_target[2] += (target_person.z - self.camera_target[2]) * 0.05
                self.camera_x = self.camera_target[0] + 5.0
                self.camera_y = 4.0
                self.camera_z = self.camera_target[2] + 5.0

        elif self.camera_mode == "dramatic":
            # Low angle dramatic shot
            if hasattr(self, 'camera_follow_target'):
                for p in self.people:
                    if p.id == self.camera_follow_target:
                        self.camera_target = [p.x, 1.0, p.z]
                        self.camera_x = p.x + 3.0
                        self.camera_y = 1.5
                        self.camera_z = p.z + 3.0
                        break

        gluLookAt(
            self.camera_x + shake_x, self.camera_y, self.camera_z + shake_y,
            self.camera_target[0], self.camera_target[1], self.camera_target[2],
            0, 1, 0
        )

    def update_scenario(self, dt):
        """Update scenario events."""
        if not self.demo_started:
            return

        self.scenario_time += dt

        # Process events
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

        if event.action == "walk":
            for p in self.people:
                if p.id == event.target:
                    p.state = PersonState.WALKING
                    p.target_x = event.params.get("target_x", p.x)
                    p.target_z = event.params.get("target_z", p.z)
                    p.speed = event.params.get("speed", 0.5)
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
                    self.fall_count += 1
                    break

        elif event.action == "recover":
            for p in self.people:
                if p.id == event.target:
                    p.state = PersonState.RECOVERING
                    break

        elif event.action == "alert":
            self.alert_active = True
            self.alert_intensity = 1.0
            self.alert_message = event.params.get("message", "跌倒警報！")
            self.camera_shake = 0.5
            self.detection_count += 1

        elif event.action == "camera":
            self.camera_mode = event.params.get("mode", "overview")
            if "target" in event.params:
                self.camera_follow_target = event.params["target"]

        elif event.action == "reset":
            self._reset_demo()

    def _reset_demo(self):
        """Reset the demo to initial state."""
        self.scenario_time = 0.0
        self.current_event_index = 0
        self.demo_started = False
        self.alert_active = False
        self.alert_intensity = 0.0
        self.camera_mode = "overview"
        self.current_narration = "按 [SPACE] 開始演示"

        # Reset people
        for p in self.people:
            p.state = PersonState.IDLE
            p.fall_progress = 0.0

        # Reset positions
        positions = {
            "wang": (3.0, 4.0),
            "li": (8.0, 9.0),
            "chen": (14.0, 4.0),
            "lin": (2.0, 9.0),
            "zhang": (9.0, 5.0),
            "nurse": (14.0, 9.0),
        }
        for p in self.people:
            if p.id in positions:
                p.x, p.z = positions[p.id]
                p.target_x, p.target_z = positions[p.id]

    def update_people(self, dt):
        """Update all people."""
        for person in self.people:
            if person.state == PersonState.WALKING:
                # Move towards target
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

            elif person.state == PersonState.STUMBLING:
                person.stumble_time -= dt
                person.walk_cycle += dt * 8  # Faster, erratic movement
                if person.stumble_time <= 0:
                    person.state = PersonState.IDLE

            elif person.state == PersonState.FALLING:
                person.fall_progress = min(1.0, person.fall_progress + dt * 1.5)
                if person.fall_progress >= 1.0:
                    person.state = PersonState.FALLEN

            elif person.state == PersonState.RECOVERING:
                person.fall_progress = max(0.0, person.fall_progress - dt * 0.5)
                if person.fall_progress <= 0:
                    person.state = PersonState.IDLE

        # Update alert
        if self.alert_active:
            self.alert_intensity = max(0, self.alert_intensity - dt * 0.2)
            self.camera_shake = max(0, self.camera_shake - dt * 0.5)
            if self.alert_intensity <= 0:
                self.alert_active = False

    def draw_floor(self):
        """Draw the facility floor."""
        glDisable(GL_LIGHTING)

        # Base floor
        glColor4f(0.25, 0.25, 0.28, 1.0)
        glBegin(GL_QUADS)
        glVertex3f(-1, -0.02, -1)
        glVertex3f(self.facility_width + 1, -0.02, -1)
        glVertex3f(self.facility_width + 1, -0.02, self.facility_depth + 1)
        glVertex3f(-1, -0.02, self.facility_depth + 1)
        glEnd()

        # Zone floors
        for zone in self.zones:
            x, z, w, d = zone["rect"]
            r, g, b = zone["color"]

            glColor4f(r, g, b, 1.0)
            glBegin(GL_QUADS)
            glVertex3f(x + 0.1, 0.0, z + 0.1)
            glVertex3f(x + w - 0.1, 0.0, z + 0.1)
            glVertex3f(x + w - 0.1, 0.0, z + d - 0.1)
            glVertex3f(x + 0.1, 0.0, z + d - 0.1)
            glEnd()

            # Zone border
            glColor4f(r * 0.7, g * 0.7, b * 0.7, 1.0)
            glLineWidth(2.0)
            glBegin(GL_LINE_LOOP)
            glVertex3f(x, 0.01, z)
            glVertex3f(x + w, 0.01, z)
            glVertex3f(x + w, 0.01, z + d)
            glVertex3f(x, 0.01, z + d)
            glEnd()

        glEnable(GL_LIGHTING)

    def draw_walls(self):
        """Draw facility walls."""
        glEnable(GL_LIGHTING)
        glColor4f(0.85, 0.85, 0.88, 0.5)

        # Simplified walls
        wall_height = 2.5

        # Back wall (with windows)
        glPushMatrix()
        glTranslatef(self.facility_width / 2, wall_height / 2, 0)
        glScalef(self.facility_width, wall_height, 0.15)
        draw_cube(1.0)
        glPopMatrix()

        # Front wall (partial - entrance)
        glPushMatrix()
        glTranslatef(5, wall_height / 2, self.facility_depth)
        glScalef(10, wall_height, 0.15)
        draw_cube(1.0)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(17.5, wall_height / 2, self.facility_depth)
        glScalef(5, wall_height, 0.15)
        draw_cube(1.0)
        glPopMatrix()

    def draw_furniture(self):
        """Draw all furniture."""
        glEnable(GL_LIGHTING)

        for f in self.furniture:
            glPushMatrix()
            glTranslatef(f.x, 0, f.z)
            glRotatef(f.rotation, 0, 1, 0)
            glColor4f(*f.color, 1.0)

            if f.type == "bed":
                # Bed frame
                glPushMatrix()
                glTranslatef(0, 0.25, 0)
                glScalef(0.9, 0.5, 1.8)
                draw_cube(1.0)
                glPopMatrix()
                # Mattress
                glColor4f(0.95, 0.95, 0.98, 1.0)
                glPushMatrix()
                glTranslatef(0, 0.55, 0)
                glScalef(0.85, 0.15, 1.7)
                draw_cube(1.0)
                glPopMatrix()
                # Pillow
                glColor4f(1.0, 1.0, 1.0, 1.0)
                glPushMatrix()
                glTranslatef(0, 0.65, -0.6)
                glScalef(0.5, 0.1, 0.3)
                draw_cube(1.0)
                glPopMatrix()

            elif f.type == "table":
                # Table top
                glPushMatrix()
                glTranslatef(0, 0.7, 0)
                glScalef(1.0, 0.05, 1.0)
                draw_cube(1.0)
                glPopMatrix()
                # Legs
                for dx, dz in [(-0.4, -0.4), (0.4, -0.4), (-0.4, 0.4), (0.4, 0.4)]:
                    glPushMatrix()
                    glTranslatef(dx, 0.35, dz)
                    glScalef(0.05, 0.7, 0.05)
                    draw_cube(1.0)
                    glPopMatrix()

            elif f.type == "chair":
                # Seat
                glPushMatrix()
                glTranslatef(0, 0.4, 0)
                glScalef(0.4, 0.05, 0.4)
                draw_cube(1.0)
                glPopMatrix()
                # Back
                glPushMatrix()
                glTranslatef(0, 0.65, -0.18)
                glScalef(0.4, 0.5, 0.05)
                draw_cube(1.0)
                glPopMatrix()
                # Legs
                for dx, dz in [(-0.15, -0.15), (0.15, -0.15), (-0.15, 0.15), (0.15, 0.15)]:
                    glPushMatrix()
                    glTranslatef(dx, 0.2, dz)
                    glScalef(0.04, 0.4, 0.04)
                    draw_cube(1.0)
                    glPopMatrix()

            elif f.type == "wheelchair":
                glColor4f(0.2, 0.2, 0.25, 1.0)
                # Seat
                glPushMatrix()
                glTranslatef(0, 0.5, 0)
                glScalef(0.5, 0.05, 0.5)
                draw_cube(1.0)
                glPopMatrix()
                # Wheels (simplified as discs)
                for side in [-0.3, 0.3]:
                    glPushMatrix()
                    glTranslatef(side, 0.3, 0)
                    glRotatef(90, 0, 1, 0)
                    draw_cylinder(0.25, 0.05, 16)
                    glPopMatrix()

            elif f.type == "walker":
                glColor4f(0.7, 0.7, 0.72, 1.0)
                # Frame
                for dx in [-0.2, 0.2]:
                    glPushMatrix()
                    glTranslatef(dx, 0.4, 0)
                    glScalef(0.03, 0.8, 0.03)
                    draw_cube(1.0)
                    glPopMatrix()
                # Handle bar
                glPushMatrix()
                glTranslatef(0, 0.8, 0)
                glScalef(0.5, 0.03, 0.03)
                draw_cube(1.0)
                glPopMatrix()

            elif f.type == "medical_cart":
                # Cart body
                glPushMatrix()
                glTranslatef(0, 0.5, 0)
                glScalef(0.6, 0.8, 0.4)
                draw_cube(1.0)
                glPopMatrix()
                # Wheels
                glColor4f(0.2, 0.2, 0.2, 1.0)
                for dx, dz in [(-0.25, -0.15), (0.25, -0.15), (-0.25, 0.15), (0.25, 0.15)]:
                    glPushMatrix()
                    glTranslatef(dx, 0.05, dz)
                    draw_sphere(0.05, 8, 8)
                    glPopMatrix()

            elif f.type == "desk":
                glPushMatrix()
                glTranslatef(0, 0.35, 0)
                glScalef(1.5, 0.7, 0.7)
                draw_cube(1.0)
                glPopMatrix()

            glPopMatrix()

    def draw_radar(self):
        """Draw radar units and coverage."""
        glDisable(GL_LIGHTING)

        for radar in self.radars:
            # Radar unit
            glPushMatrix()
            glTranslatef(radar["x"], radar["y"], radar["z"])
            glColor4f(0.2, 0.5, 0.9, 1.0)
            draw_cube(0.25)

            # Blinking light
            blink = 0.5 + 0.5 * math.sin(self.time * 5)
            glColor4f(0.0, 1.0, 0.0, blink)
            glPushMatrix()
            glTranslatef(0, 0.15, 0.13)
            draw_sphere(0.03, 8, 8)
            glPopMatrix()
            glPopMatrix()

            # Radar sweep cone
            glPushMatrix()
            glTranslatef(radar["x"], radar["y"], radar["z"])
            glRotatef(radar["angle"], 0, 1, 0)

            # Animated sweep
            sweep_angle = (self.time * 60) % 360
            glRotatef(sweep_angle, 0, 1, 0)

            # Cone visualization
            range_dist = radar["range"]
            glColor4f(0.2, 0.6, 1.0, 0.1)

            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, 0, 0)
            for i in range(17):
                angle = math.radians(-40 + i * 5)
                x = range_dist * math.sin(angle)
                z = -range_dist * math.cos(angle)
                glVertex3f(x, -radar["y"] + 0.1, z)
            glEnd()

            # Sweep line
            glColor4f(0.3, 0.8, 1.0, 0.5)
            glLineWidth(2.0)
            glBegin(GL_LINES)
            glVertex3f(0, 0, 0)
            glVertex3f(0, -radar["y"] + 0.1, -range_dist)
            glEnd()

            glPopMatrix()

        glEnable(GL_LIGHTING)

    def draw_person(self, person: Person):
        """Draw a person with animation."""
        glEnable(GL_LIGHTING)

        glPushMatrix()
        glTranslatef(person.x, 0, person.z)
        glRotatef(person.rotation, 0, 1, 0)

        # Fall rotation
        if person.state in [PersonState.FALLING, PersonState.FALLEN, PersonState.RECOVERING]:
            fall_angle = person.fall_progress * 90
            glRotatef(fall_angle, 1, 0, 0)

        # Stumble effect
        if person.state == PersonState.STUMBLING:
            wobble = math.sin(person.walk_cycle * 3) * 15
            glRotatef(wobble, 0, 0, 1)

        # Color (red tint when falling/fallen)
        r, g, b = person.color
        if person.state in [PersonState.FALLING, PersonState.FALLEN]:
            r = min(1.0, r + 0.3)
            g *= 0.5
            b *= 0.5
        glColor4f(r, g, b, 1.0)

        # Walking animation
        walk_offset = 0
        if person.state == PersonState.WALKING:
            walk_offset = math.sin(person.walk_cycle) * 0.03

        # Body
        glPushMatrix()
        glTranslatef(0, 0.8 + walk_offset, 0)
        glScalef(0.3, 0.6, 0.2)
        draw_cube(1.0)
        glPopMatrix()

        # Head
        glPushMatrix()
        glTranslatef(0, 1.4 + walk_offset, 0)
        draw_sphere(0.15, 12, 12)
        glPopMatrix()

        # Arms with swing
        arm_swing = math.sin(person.walk_cycle) * 20 if person.state == PersonState.WALKING else 0

        # Left arm
        glPushMatrix()
        glTranslatef(-0.22, 1.0, 0)
        glRotatef(arm_swing, 1, 0, 0)
        glTranslatef(0, -0.2, 0)
        glScalef(0.08, 0.4, 0.08)
        draw_cube(1.0)
        glPopMatrix()

        # Right arm
        glPushMatrix()
        glTranslatef(0.22, 1.0, 0)
        glRotatef(-arm_swing, 1, 0, 0)
        glTranslatef(0, -0.2, 0)
        glScalef(0.08, 0.4, 0.08)
        draw_cube(1.0)
        glPopMatrix()

        # Legs with walk cycle
        leg_swing = math.sin(person.walk_cycle) * 30 if person.state == PersonState.WALKING else 0

        # Left leg
        glPushMatrix()
        glTranslatef(-0.1, 0.5, 0)
        glRotatef(leg_swing, 1, 0, 0)
        glTranslatef(0, -0.25, 0)
        glScalef(0.1, 0.5, 0.1)
        draw_cube(1.0)
        glPopMatrix()

        # Right leg
        glPushMatrix()
        glTranslatef(0.1, 0.5, 0)
        glRotatef(-leg_swing, 1, 0, 0)
        glTranslatef(0, -0.25, 0)
        glScalef(0.1, 0.5, 0.1)
        draw_cube(1.0)
        glPopMatrix()

        glPopMatrix()

        # Status ring for fallen person
        if person.state in [PersonState.FALLING, PersonState.FALLEN]:
            glDisable(GL_LIGHTING)
            glPushMatrix()
            glTranslatef(person.x, 0.05, person.z)

            pulse = 0.5 + 0.5 * math.sin(self.time * 6)
            ring_size = 1.0 + pulse * 0.3

            glColor4f(1.0, 0.2, 0.2, 0.8 * pulse)
            glLineWidth(4.0)
            glBegin(GL_LINE_LOOP)
            for i in range(32):
                angle = 2 * math.pi * i / 32
                glVertex3f(ring_size * math.cos(angle), 0, ring_size * math.sin(angle))
            glEnd()

            glPopMatrix()
            glEnable(GL_LIGHTING)

    def draw_alert_overlay(self):
        """Draw alert visual effects."""
        if not self.alert_active:
            return

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Red vignette
        flash = 0.5 + 0.5 * math.sin(self.time * 10)
        alpha = self.alert_intensity * 0.4 * flash

        # Border glow
        glColor4f(1.0, 0.0, 0.0, alpha)
        border = 50

        # Top
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(self.width, 0)
        glColor4f(1.0, 0.0, 0.0, 0.0)
        glVertex2f(self.width, border)
        glVertex2f(0, border)
        glEnd()

        # Bottom
        glColor4f(1.0, 0.0, 0.0, alpha)
        glBegin(GL_QUADS)
        glVertex2f(0, self.height)
        glVertex2f(self.width, self.height)
        glColor4f(1.0, 0.0, 0.0, 0.0)
        glVertex2f(self.width, self.height - border)
        glVertex2f(0, self.height - border)
        glEnd()

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def draw_hud(self):
        """Draw heads-up display."""
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Top bar
        glColor4f(0.1, 0.1, 0.15, 0.85)
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(self.width, 0)
        glVertex2f(self.width, 60)
        glVertex2f(0, 60)
        glEnd()

        # Bottom narration bar
        glColor4f(0.05, 0.05, 0.1, 0.9)
        glBegin(GL_QUADS)
        glVertex2f(0, self.height - 80)
        glVertex2f(self.width, self.height - 80)
        glVertex2f(self.width, self.height)
        glVertex2f(0, self.height)
        glEnd()

        # Status indicators
        # Live indicator
        pulse = 0.5 + 0.5 * math.sin(self.time * 3)
        glColor4f(1.0, 0.3, 0.3, pulse)
        glBegin(GL_QUADS)
        glVertex2f(20, 20)
        glVertex2f(35, 20)
        glVertex2f(35, 35)
        glVertex2f(20, 35)
        glEnd()

        # Stats boxes
        glColor4f(0.2, 0.2, 0.25, 0.9)
        for i, (label, value) in enumerate([
            ("偵測次數", str(self.detection_count)),
            ("跌倒事件", str(self.fall_count)),
            ("系統狀態", "運行中")
        ]):
            x = self.width - 150 - i * 130
            glBegin(GL_QUADS)
            glVertex2f(x, 10)
            glVertex2f(x + 120, 10)
            glVertex2f(x + 120, 50)
            glVertex2f(x, 50)
            glEnd()

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        # Render text using pygame
        self._render_text()

    def _render_text(self):
        """Render text overlays."""
        # Create a surface for text
        text_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # Title
        title = self.font_medium.render("mmWave 毫米波跌倒偵測系統 | 赤土崎多功能館", True, (200, 200, 210))
        text_surface.blit(title, (50, 18))

        # Live indicator text
        live_text = self.font_small.render("LIVE", True, (255, 100, 100))
        text_surface.blit(live_text, (45, 22))

        # Stats
        stats = [
            (f"偵測: {self.detection_count}", self.width - 270),
            (f"跌倒: {self.fall_count}", self.width - 140),
        ]
        for text, x in stats:
            rendered = self.font_small.render(text, True, (180, 180, 190))
            text_surface.blit(rendered, (x, 22))

        # Narration
        if self.current_narration:
            # Fade out old narration
            age = self.time - self.narration_time
            alpha = min(255, max(0, 255 - int(age * 20) if age > 5 else 255))

            narration = self.font_medium.render(self.current_narration, True, (255, 255, 255, alpha))
            text_rect = narration.get_rect(center=(self.width // 2, self.height - 45))
            text_surface.blit(narration, text_rect)

        # Instructions
        if not self.demo_started:
            inst = self.font_small.render("按 [SPACE] 開始演示 | [R] 重置 | [ESC] 離開", True, (150, 150, 160))
            inst_rect = inst.get_rect(center=(self.width // 2, self.height - 15))
            text_surface.blit(inst, inst_rect)

        # Convert pygame surface to OpenGL texture
        text_data = pygame.image.tostring(text_surface, "RGBA", True)

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(self.width, 0)
        glTexCoord2f(1, 1); glVertex2f(self.width, self.height)
        glTexCoord2f(0, 1); glVertex2f(0, self.height)
        glEnd()

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        glDeleteTextures([tex_id])
        glDisable(GL_TEXTURE_2D)
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
                elif event.key == pygame.K_r:
                    self._reset_demo()
                elif event.key == pygame.K_f:
                    # Manual fall trigger
                    person = random.choice([p for p in self.people if p.state == PersonState.IDLE])
                    if person:
                        person.state = PersonState.FALLING
                        person.fall_progress = 0.0
                        self.alert_active = True
                        self.alert_intensity = 1.0
                        self.camera_shake = 0.5
                        self.fall_count += 1
                        self.detection_count += 1
                        self.current_narration = f"⚠️ {person.name} 發生跌倒！"
                        self.narration_time = self.time
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_captured = True
                    self.last_mouse_pos = event.pos
                elif event.button == 4:  # Scroll up
                    self.camera_y = max(5, self.camera_y - 1)
                elif event.button == 5:  # Scroll down
                    self.camera_y = min(30, self.camera_y + 1)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_captured = False
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_captured:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    self.camera_target[0] -= dx * 0.02
                    self.camera_target[2] -= dy * 0.02
                    self.last_mouse_pos = event.pos

    def run(self):
        """Main loop."""
        pygame.init()
        pygame.display.set_caption("mmWave Fall Detection - Cinematic Demo | 赤土崎多功能館")

        display = (self.width, self.height)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

        self.init_gl()
        self._reset_demo()

        clock = pygame.time.Clock()
        last_time = time.time()

        print("=" * 60)
        print("mmWave 毫米波跌倒偵測系統 - 電影級演示")
        print("=" * 60)
        print("控制說明:")
        print("  SPACE  - 開始/重新播放演示")
        print("  R      - 重置場景")
        print("  F      - 手動觸發跌倒")
        print("  滑鼠拖曳 - 移動視角")
        print("  滾輪   - 縮放")
        print("  ESC    - 離開")
        print("=" * 60)

        while self.running:
            current_time = time.time()
            dt = min(0.1, current_time - last_time)
            last_time = current_time
            self.time = pygame.time.get_ticks() / 1000.0

            self.handle_events()
            self.update_scenario(dt)
            self.update_people(dt)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            self.update_camera()

            self.draw_floor()
            self.draw_walls()
            self.draw_furniture()
            self.draw_radar()

            for person in self.people:
                self.draw_person(person)

            self.draw_alert_overlay()
            self.draw_hud()

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cinematic 3D Fall Detection Demo")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API server URL")
    args = parser.parse_args()

    demo = CinematicFallDemo(api_url=args.api_url)
    demo.run()


if __name__ == "__main__":
    main()
