#!/usr/bin/env python3
"""5G O-RAN DAS Indoor Positioning & Fall Detection Demo.

Features:
- Multiple distributed antenna units (RRU) for dense coverage
- Triangulation-based indoor positioning using signal fusion
- ISAC (Integrated Sensing and Communication) visualization
- Coverage overlap areas for accurate positioning
- Pre-fall anomaly detection via sensing network

Based on:
- 5G O-RAN architecture with DAS deployment
- ISAC (Integrated Sensing and Communication) technology
- Multi-beam positioning with <1m accuracy
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


# ============================================================================
# Utility Functions
# ============================================================================

def draw_sphere(radius, slices=12, stacks=12):
    """Draw a sphere."""
    for i in range(stacks):
        lat0 = math.pi * (-0.5 + float(i) / stacks)
        z0, zr0 = radius * math.sin(lat0), radius * math.cos(lat0)
        lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
        z1, zr1 = radius * math.sin(lat1), radius * math.cos(lat1)
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


# ============================================================================
# Enums and Data Classes
# ============================================================================

class AlertLevel(Enum):
    NORMAL = 0
    ANOMALY = 1
    WARNING = 2
    ALERT = 3


class PersonState(Enum):
    IDLE = "idle"
    WALKING = "walking"
    UNSTABLE = "unstable"
    STUMBLING = "stumbling"
    FALLING = "falling"
    FALLEN = "fallen"


@dataclass
class AntennaUnit:
    """5G O-RAN Remote Radio Unit (RRU)."""
    id: str
    x: float
    z: float
    y: float = 2.8  # Ceiling mounted
    coverage_radius: float = 6.0
    color: Tuple[float, float, float] = (0.2, 0.6, 0.9)
    active: bool = True
    # Detection data
    detected_persons: List[str] = field(default_factory=list)
    signal_strengths: Dict[str, float] = field(default_factory=dict)


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
    walk_cycle: float = 0.0
    speed: float = 0.4
    # Positioning data
    positioned: bool = False
    position_accuracy: float = 0.0
    detecting_antennas: List[str] = field(default_factory=list)
    # Risk assessment
    anomaly_score: float = 0.0
    fall_risk: float = 0.0

    def __post_init__(self):
        self.target_x = self.x
        self.target_z = self.z


@dataclass
class WavePulse:
    """Radio wave pulse from antenna."""
    antenna_id: str
    origin_x: float
    origin_z: float
    radius: float = 0.0
    max_radius: float = 6.0
    speed: float = 4.0
    alpha: float = 0.5


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

class ORANDASDemo:
    def __init__(self):
        self.width = 1280
        self.height = 720
        self.running = True

        # Time
        self.time = 0.0
        self.scenario_time = 0.0
        self.demo_started = False

        # Camera
        self.camera_x = 10.0
        self.camera_y = 14.0
        self.camera_z = 18.0
        self.camera_target = [10.0, 0.0, 6.25]
        self.camera_shake = 0.0

        # Facility dimensions (赤土崎多功能館)
        self.facility_width = 20.0
        self.facility_depth = 12.5

        # 5G O-RAN DAS Antenna Units (dense deployment)
        self.antennas = [
            AntennaUnit("RRU-A1", 3.0, 2.0, coverage_radius=5.5, color=(0.2, 0.7, 0.9)),
            AntennaUnit("RRU-A2", 10.0, 2.0, coverage_radius=5.5, color=(0.2, 0.9, 0.7)),
            AntennaUnit("RRU-A3", 17.0, 2.0, coverage_radius=5.5, color=(0.9, 0.7, 0.2)),
            AntennaUnit("RRU-B1", 3.0, 6.25, coverage_radius=5.5, color=(0.7, 0.2, 0.9)),
            AntennaUnit("RRU-B2", 10.0, 6.25, coverage_radius=5.5, color=(0.9, 0.2, 0.7)),
            AntennaUnit("RRU-B3", 17.0, 6.25, coverage_radius=5.5, color=(0.2, 0.9, 0.9)),
            AntennaUnit("RRU-C1", 3.0, 10.5, coverage_radius=5.5, color=(0.9, 0.5, 0.2)),
            AntennaUnit("RRU-C2", 10.0, 10.5, coverage_radius=5.5, color=(0.5, 0.9, 0.2)),
            AntennaUnit("RRU-C3", 17.0, 10.5, coverage_radius=5.5, color=(0.2, 0.5, 0.9)),
        ]

        # Wave pulses
        self.wave_pulses: List[WavePulse] = []
        self.wave_interval = 0.8
        self.last_wave_time = 0.0

        # Alert system
        self.alert_level = AlertLevel.NORMAL
        self.alert_target = ""
        self.alert_message = ""

        # Statistics
        self.positioned_count = 0
        self.anomaly_count = 0
        self.fall_count = 0

        # People
        self.people = [
            Person("wang", "王阿公", 4.0, 4.0, (0.9, 0.75, 0.55)),
            Person("li", "李阿嬤", 8.0, 8.0, (0.85, 0.7, 0.55)),
            Person("chen", "陳伯伯", 15.0, 5.0, (0.8, 0.65, 0.5)),
            Person("nurse", "護理師", 17.0, 10.0, (0.95, 0.95, 0.95)),
        ]

        # Zones
        self.zones = [
            {"id": "corridor", "rect": (0, 0, 20, 2.5), "color": (0.28, 0.3, 0.35), "name": "走廊"},
            {"id": "activity_a", "rect": (0, 2.5, 7, 5), "color": (0.32, 0.38, 0.32), "name": "活動區A"},
            {"id": "activity_b", "rect": (7, 2.5, 6, 5), "color": (0.35, 0.35, 0.32), "name": "活動區B"},
            {"id": "dining", "rect": (13, 2.5, 7, 5), "color": (0.38, 0.32, 0.32), "name": "餐廳"},
            {"id": "rehab", "rect": (0, 7.5, 7, 5), "color": (0.32, 0.32, 0.4), "name": "復健室"},
            {"id": "rest", "rect": (7, 7.5, 6, 5), "color": (0.35, 0.32, 0.38), "name": "休息區"},
            {"id": "nurse_station", "rect": (13, 7.5, 7, 5), "color": (0.36, 0.36, 0.38), "name": "護理站"},
        ]

        # Scenario - 故事：赤土崎多功能館的一個早晨
        self.scenario_events = [
            # === 序幕：系統啟動 ===
            ScenarioEvent(0.5, "narration", "", {}, "【赤土崎多功能館】早上 8:30"),
            ScenarioEvent(2.5, "narration", "", {}, "5G O-RAN 專網啟動，9 個天線單元開始感知掃描..."),
            ScenarioEvent(4.5, "narration", "", {}, "ISAC 感知網路已連線，開始監測長者活動"),

            # === 第一幕：日常早晨 ===
            ScenarioEvent(7.0, "walk", "nurse", {"target_x": 10.0, "target_z": 8.0, "speed": 0.6},
                         "護理師小美開始早班巡房，確認長者狀況"),
            ScenarioEvent(9.0, "walk", "li", {"target_x": 14.0, "target_z": 4.0, "speed": 0.35},
                         "李阿嬤（78歲）緩步走向餐廳準備用早餐"),
            ScenarioEvent(11.0, "narration", "", {},
                         "📍 系統定位：李阿嬤 位置精度 32cm（3天線融合）"),
            ScenarioEvent(13.0, "walk", "chen", {"target_x": 3.0, "target_z": 9.0, "speed": 0.4},
                         "陳伯伯（85歲）前往復健室做早操"),

            # === 第二幕：王阿公的故事 ===
            ScenarioEvent(16.0, "narration", "", {},
                         "【關注對象】王阿公（82歲）- 昨晚睡眠品質不佳"),
            ScenarioEvent(18.0, "walk", "wang", {"target_x": 6.0, "target_z": 4.0, "speed": 0.45},
                         "王阿公從休息區出發，想去活動區看電視"),
            ScenarioEvent(20.0, "narration", "", {},
                         "📍 三角定位：RRU-A1、RRU-B1、RRU-B2 同時偵測"),

            # === 第三幕：異常偵測 ===
            ScenarioEvent(23.0, "anomaly", "wang", {"score": 0.25},
                         "⚠ 感知異常：步態週期不規則（正常 1.2s → 實測 1.8s）"),
            ScenarioEvent(25.0, "walk", "wang", {"target_x": 8.0, "target_z": 5.0, "speed": 0.35},
                         "王阿公繼續前進，但步伐明顯變慢"),
            ScenarioEvent(27.0, "anomaly", "wang", {"score": 0.45},
                         "⚠⚠ 多普勒特徵異常：身體晃動幅度增加 +40%"),
            ScenarioEvent(29.0, "unstable", "wang", {},
                         "ISAC 分析：步態穩定性下降，疑似頭暈症狀"),

            # === 第四幕：風險預測與預警 ===
            ScenarioEvent(32.0, "predict", "wang", {"risk": 0.65},
                         "🔴 AI 預測：跌倒風險 65% - 系統發送黃色預警"),
            ScenarioEvent(33.5, "narration", "", {},
                         "📱 護理師小美手機收到推播：「王阿公步態異常，請關注」"),
            ScenarioEvent(35.0, "walk", "nurse", {"target_x": 6.0, "target_z": 6.0, "speed": 0.5},
                         "護理師注意到警示，開始移動觀察"),
            ScenarioEvent(37.0, "stumble", "wang", {"duration": 3.0},
                         "⚠⚠⚠ 緊急！偵測到蹣跚步態 - 訊號特徵急遽變化"),

            # === 第五幕：危機時刻 ===
            ScenarioEvent(40.0, "predict", "wang", {"risk": 0.88},
                         "🚨 風險飆升至 88%！系統發送紅色警報"),
            ScenarioEvent(41.5, "narration", "", {},
                         "📱💥 緊急推播：「王阿公極高跌倒風險！位置：活動區B (8.0, 5.0)」"),
            ScenarioEvent(43.0, "respond", "nurse", {"target_x": 8.0, "target_z": 5.0, "speed": 1.2},
                         "🏃 護理師小美快步趕往現場！"),
            ScenarioEvent(45.0, "fall", "wang", {},
                         "🚨🚨 跌倒發生！RRU-A2、B1、B2 同時偵測到急遽訊號變化"),

            # === 第六幕：緊急應變 ===
            ScenarioEvent(47.0, "narration", "", {},
                         "⏱ 從預警到跌倒：13 秒 | 護理師距離：2.5 公尺"),
            ScenarioEvent(49.0, "narration", "", {},
                         "🏥 系統自動記錄：跌倒時間、精確位置、前後 30 秒感測數據"),
            ScenarioEvent(51.0, "recover", "wang", {},
                         "✅ 護理師抵達！因提前預警，反應時間縮短 70%"),
            ScenarioEvent(53.0, "narration", "", {},
                         "📋 系統建議：安排神經內科檢查，評估頭暈原因"),

            # === 尾聲 ===
            ScenarioEvent(56.0, "walk", "nurse", {"target_x": 17.0, "target_z": 10.0, "speed": 0.4},
                         "護理師協助王阿公返回護理站休息"),
            ScenarioEvent(58.0, "walk", "wang", {"target_x": 16.0, "target_z": 9.0, "speed": 0.25},
                         ""),
            ScenarioEvent(60.0, "narration", "", {},
                         "【本次事件統計】預警提前 13 秒 | 定位精度 38cm | 0 嚴重傷害"),
            ScenarioEvent(63.0, "narration", "", {},
                         "5G O-RAN ISAC 感知網路 - 讓照護更及時、更精準"),
            ScenarioEvent(67.0, "reset", "", {},
                         "🔄 演示結束 - 按 SPACE 重新開始"),
        ]
        self.current_event_index = 0
        self.current_narration = "【赤土崎多功能館】按 [SPACE] 開始 5G 感知網路演示"
        self.narration_time = 0.0

        # Initialize pygame
        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("5G O-RAN DAS 室內定位與跌倒偵測")

        # OpenGL setup
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glLightfv(GL_LIGHT0, GL_POSITION, (10, 20, 10, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.35, 0.35, 0.4, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.75, 0.75, 0.7, 1))

        glClearColor(0.08, 0.08, 0.1, 1)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

        # Font
        pygame.font.init()
        CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        try:
            self.font_large = pygame.font.Font(CJK_FONT, 28)
            self.font_medium = pygame.font.Font(CJK_FONT, 20)
            self.font_small = pygame.font.Font(CJK_FONT, 14)
        except:
            self.font_large = pygame.font.Font(None, 32)
            self.font_medium = pygame.font.Font(None, 24)
            self.font_small = pygame.font.Font(None, 18)

    def emit_wave_pulses(self):
        """Emit wave pulses from all active antennas."""
        for antenna in self.antennas:
            if antenna.active:
                pulse = WavePulse(
                    antenna_id=antenna.id,
                    origin_x=antenna.x,
                    origin_z=antenna.z,
                    max_radius=antenna.coverage_radius,
                    speed=3.0
                )
                self.wave_pulses.append(pulse)

    def update_sensing(self, dt):
        """Update sensing network and positioning."""
        # Update wave pulses
        pulses_to_remove = []
        for pulse in self.wave_pulses:
            pulse.radius += pulse.speed * dt
            pulse.alpha = max(0, 0.4 * (1 - pulse.radius / pulse.max_radius))
            if pulse.radius > pulse.max_radius:
                pulses_to_remove.append(pulse)
        for p in pulses_to_remove:
            self.wave_pulses.remove(p)

        # Emit new pulses
        if self.time - self.last_wave_time > self.wave_interval:
            self.emit_wave_pulses()
            self.last_wave_time = self.time

        # Update antenna detection and person positioning
        for antenna in self.antennas:
            antenna.detected_persons.clear()
            antenna.signal_strengths.clear()

        for person in self.people:
            person.detecting_antennas.clear()
            detecting_count = 0

            for antenna in self.antennas:
                dist = math.sqrt((person.x - antenna.x)**2 + (person.z - antenna.z)**2)
                if dist < antenna.coverage_radius:
                    # Calculate signal strength (inverse square law)
                    strength = max(0, 1 - (dist / antenna.coverage_radius)**1.5)
                    antenna.detected_persons.append(person.id)
                    antenna.signal_strengths[person.id] = strength
                    person.detecting_antennas.append(antenna.id)
                    detecting_count += 1

            # Positioning accuracy based on number of detecting antennas
            if detecting_count >= 3:
                person.positioned = True
                person.position_accuracy = 0.3 + random.uniform(-0.1, 0.1)  # <50cm
            elif detecting_count >= 2:
                person.positioned = True
                person.position_accuracy = 0.8 + random.uniform(-0.2, 0.2)  # <1m
            elif detecting_count >= 1:
                person.positioned = True
                person.position_accuracy = 2.0 + random.uniform(-0.5, 0.5)  # ~2m
            else:
                person.positioned = False

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
        """Execute scenario event."""
        if event.narration:
            self.current_narration = event.narration
            self.narration_time = self.time

        if event.action == "walk":
            for p in self.people:
                if p.id == event.target:
                    p.state = PersonState.WALKING
                    p.target_x = event.params.get("target_x", p.x)
                    p.target_z = event.params.get("target_z", p.z)
                    p.speed = event.params.get("speed", 0.4)
                    break

        elif event.action == "anomaly":
            for p in self.people:
                if p.id == event.target:
                    p.anomaly_score = event.params.get("score", 0.3)
                    self.alert_level = AlertLevel.ANOMALY
                    self.alert_target = p.id
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
                    self.alert_level = AlertLevel.WARNING
                    self.alert_target = p.id
                    break

        elif event.action == "stumble":
            for p in self.people:
                if p.id == event.target:
                    p.state = PersonState.STUMBLING
                    break

        elif event.action == "fall":
            for p in self.people:
                if p.id == event.target:
                    p.state = PersonState.FALLING
                    p.fall_progress = 0.0
                    self.alert_level = AlertLevel.ALERT
                    self.alert_target = p.id
                    self.camera_shake = 0.4
                    self.fall_count += 1
                    break

        elif event.action == "respond":
            for p in self.people:
                if p.id == event.target:
                    p.state = PersonState.WALKING
                    p.target_x = event.params.get("target_x", p.x)
                    p.target_z = event.params.get("target_z", p.z)
                    p.speed = 1.0
                    break

        elif event.action == "recover":
            for p in self.people:
                if p.id == event.target:
                    p.state = PersonState.IDLE
                    p.fall_progress = 0.0
                    self.alert_level = AlertLevel.NORMAL
                    break

        elif event.action == "reset":
            self._reset_demo()

    def _reset_demo(self):
        """Reset demo."""
        self.scenario_time = 0.0
        self.current_event_index = 0
        self.demo_started = False
        self.alert_level = AlertLevel.NORMAL
        self.current_narration = "按 [SPACE] 開始 5G O-RAN DAS 感知網路演示"
        self.wave_pulses.clear()

        positions = {"wang": (4.0, 4.0), "li": (8.0, 8.0), "chen": (15.0, 5.0), "nurse": (17.0, 10.0)}
        for p in self.people:
            p.state = PersonState.IDLE
            p.fall_progress = 0.0
            p.anomaly_score = 0.0
            p.fall_risk = 0.0
            if p.id in positions:
                p.x, p.z = positions[p.id]
                p.target_x, p.target_z = positions[p.id]

    def update_people(self, dt):
        """Update people states."""
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
                person.walk_cycle += dt * 4
                wobble = math.sin(person.walk_cycle * 3) * 0.2
                person.x += wobble * dt

            elif person.state == PersonState.STUMBLING:
                person.walk_cycle += dt * 6
                wobble = math.sin(person.walk_cycle * 4) * 0.4
                person.x += wobble * dt

            elif person.state == PersonState.FALLING:
                person.fall_progress = min(1.0, person.fall_progress + dt * 1.5)
                if person.fall_progress >= 1.0:
                    person.state = PersonState.FALLEN

        if self.camera_shake > 0:
            self.camera_shake = max(0, self.camera_shake - dt * 0.5)

    def draw_floor(self):
        """Draw floor with zones."""
        glDisable(GL_LIGHTING)

        glColor4f(0.15, 0.15, 0.18, 1.0)
        glBegin(GL_QUADS)
        glVertex3f(-1, -0.02, -1)
        glVertex3f(self.facility_width + 1, -0.02, -1)
        glVertex3f(self.facility_width + 1, -0.02, self.facility_depth + 1)
        glVertex3f(-1, -0.02, self.facility_depth + 1)
        glEnd()

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

        # Grid
        glColor4f(0.25, 0.25, 0.28, 1.0)
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

    def draw_antenna(self, antenna: AntennaUnit):
        """Draw antenna unit."""
        glPushMatrix()
        glTranslatef(antenna.x, antenna.y, antenna.z)

        # Antenna body
        r, g, b = antenna.color
        glColor4f(r * 0.6, g * 0.6, b * 0.6, 1.0)
        glPushMatrix()
        glScalef(0.3, 0.15, 0.3)
        draw_cube(1.0)
        glPopMatrix()

        # Active indicator
        if antenna.detected_persons:
            pulse = (math.sin(self.time * 6) + 1) * 0.5
            glColor4f(r, g, b, 0.5 + pulse * 0.5)
        else:
            glColor4f(r * 0.3, g * 0.3, b * 0.3, 0.5)
        glTranslatef(0, -0.1, 0)
        draw_sphere(0.08, 8, 8)

        glPopMatrix()

        # Coverage circle on floor
        glDisable(GL_LIGHTING)
        glLineWidth(2)

        # Coverage boundary
        if antenna.detected_persons:
            alpha = 0.4
            glColor4f(r, g, b, alpha)
        else:
            glColor4f(r * 0.5, g * 0.5, b * 0.5, 0.15)

        glBegin(GL_LINE_LOOP)
        for i in range(32):
            angle = 2 * math.pi * i / 32
            x = antenna.x + antenna.coverage_radius * math.cos(angle)
            z = antenna.z + antenna.coverage_radius * math.sin(angle)
            glVertex3f(x, 0.02, z)
        glEnd()

        # Fill coverage area
        glColor4f(r, g, b, 0.05 if not antenna.detected_persons else 0.1)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(antenna.x, 0.01, antenna.z)
        for i in range(33):
            angle = 2 * math.pi * i / 32
            x = antenna.x + antenna.coverage_radius * math.cos(angle)
            z = antenna.z + antenna.coverage_radius * math.sin(angle)
            glVertex3f(x, 0.01, z)
        glEnd()

        glEnable(GL_LIGHTING)

    def draw_wave_pulses(self):
        """Draw expanding wave pulses."""
        glDisable(GL_LIGHTING)
        glLineWidth(1.5)

        for pulse in self.wave_pulses:
            # Find antenna color
            color = (0.3, 0.6, 0.9)
            for ant in self.antennas:
                if ant.id == pulse.antenna_id:
                    color = ant.color
                    break

            glColor4f(color[0], color[1], color[2], pulse.alpha)
            glBegin(GL_LINE_LOOP)
            for i in range(24):
                angle = 2 * math.pi * i / 24
                x = pulse.origin_x + pulse.radius * math.cos(angle)
                z = pulse.origin_z + pulse.radius * math.sin(angle)
                glVertex3f(x, 0.05, z)
            glEnd()

        glEnable(GL_LIGHTING)

    def draw_triangulation(self, person: Person):
        """Draw triangulation lines from detecting antennas to person."""
        if len(person.detecting_antennas) < 2:
            return

        glDisable(GL_LIGHTING)
        glLineWidth(2)

        for ant_id in person.detecting_antennas:
            for ant in self.antennas:
                if ant.id == ant_id:
                    strength = ant.signal_strengths.get(person.id, 0.5)
                    r, g, b = ant.color

                    # Line from antenna to person
                    glColor4f(r, g, b, strength * 0.6)
                    glBegin(GL_LINES)
                    glVertex3f(ant.x, ant.y - 0.5, ant.z)
                    glVertex3f(person.x, 0.8, person.z)
                    glEnd()

                    # Signal strength indicator at midpoint
                    mx = (ant.x + person.x) / 2
                    mz = (ant.z + person.z) / 2
                    my = (ant.y + 0.8) / 2
                    glColor4f(r, g, b, strength)
                    glPushMatrix()
                    glTranslatef(mx, my, mz)
                    draw_sphere(0.08 * strength, 6, 6)
                    glPopMatrix()

        glEnable(GL_LIGHTING)

    def draw_person(self, person: Person):
        """Draw person."""
        glPushMatrix()
        glTranslatef(person.x, 0, person.z)

        # Pose based on state
        if person.state == PersonState.FALLING:
            glRotatef(person.fall_progress * 90, 0, 0, 1)
            glTranslatef(person.fall_progress * 0.4, 0, 0)
        elif person.state == PersonState.FALLEN:
            glRotatef(90, 0, 0, 1)
            glTranslatef(0.4, 0, 0)
        elif person.state == PersonState.STUMBLING:
            wobble = math.sin(person.walk_cycle) * 20
            glRotatef(wobble, 0, 0, 1)
        elif person.state == PersonState.UNSTABLE:
            wobble = math.sin(person.walk_cycle * 2) * 10
            glRotatef(wobble, 0, 0, 1)

        # Color based on alert
        r, g, b = person.color
        if person.id == self.alert_target:
            if self.alert_level == AlertLevel.ANOMALY:
                r, g, b = min(1, r + 0.3), min(1, g + 0.2), max(0, b - 0.2)
            elif self.alert_level == AlertLevel.WARNING:
                r, g, b = min(1, r + 0.4), max(0, g - 0.1), max(0, b - 0.3)
            elif self.alert_level == AlertLevel.ALERT:
                pulse = (math.sin(self.time * 8) + 1) * 0.3
                r, g, b = min(1, 0.9 + pulse), 0.2, 0.2

        glColor4f(r, g, b, 1.0)

        # Body
        glPushMatrix()
        glTranslatef(0, 0.55, 0)
        glScalef(0.3, 0.45, 0.18)
        draw_cube(1.0)
        glPopMatrix()

        # Head
        glPushMatrix()
        glTranslatef(0, 1.05, 0)
        draw_sphere(0.15, 10, 10)
        glPopMatrix()

        # Legs
        leg_swing = math.sin(person.walk_cycle) * 0.25 if person.state == PersonState.WALKING else 0
        for side in [-1, 1]:
            glPushMatrix()
            glTranslatef(side * 0.08, 0.22, side * leg_swing * 0.08)
            glScalef(0.08, 0.44, 0.08)
            draw_cube(1.0)
            glPopMatrix()

        glPopMatrix()

        # Position accuracy indicator
        if person.positioned:
            self._draw_position_indicator(person)

        # Risk indicator
        if person.fall_risk > 0 or person.anomaly_score > 0:
            self._draw_risk_bar(person)

    def _draw_position_indicator(self, person: Person):
        """Draw position accuracy circle."""
        glDisable(GL_LIGHTING)

        accuracy = person.position_accuracy
        if accuracy < 0.5:
            color = (0.2, 0.9, 0.3)  # Green - excellent
        elif accuracy < 1.0:
            color = (0.9, 0.9, 0.2)  # Yellow - good
        else:
            color = (0.9, 0.5, 0.2)  # Orange - acceptable

        glColor4f(*color, 0.4)
        glLineWidth(2)
        glBegin(GL_LINE_LOOP)
        for i in range(16):
            angle = 2 * math.pi * i / 16
            x = person.x + accuracy * math.cos(angle)
            z = person.z + accuracy * math.sin(angle)
            glVertex3f(x, 0.03, z)
        glEnd()

        glEnable(GL_LIGHTING)

    def _draw_risk_bar(self, person: Person):
        """Draw risk indicator bar above person."""
        glDisable(GL_LIGHTING)
        glPushMatrix()
        glTranslatef(person.x, 1.8, person.z)

        # Determine fill and color
        if person.fall_risk > 0:
            fill = person.fall_risk
            if fill > 0.8:
                r, g, b = 1.0, 0.2, 0.2
            elif fill > 0.5:
                r, g, b = 1.0, 0.6, 0.2
            else:
                r, g, b = 1.0, 1.0, 0.3
        else:
            fill = person.anomaly_score
            r, g, b = 1.0, 1.0, 0.3

        # Background
        bar_w, bar_h = 0.7, 0.12
        glColor4f(0.15, 0.15, 0.15, 0.8)
        glBegin(GL_QUADS)
        glVertex3f(-bar_w/2, 0, 0)
        glVertex3f(bar_w/2, 0, 0)
        glVertex3f(bar_w/2, bar_h, 0)
        glVertex3f(-bar_w/2, bar_h, 0)
        glEnd()

        # Fill
        glColor4f(r, g, b, 0.9)
        glBegin(GL_QUADS)
        glVertex3f(-bar_w/2, 0, 0.01)
        glVertex3f(-bar_w/2 + bar_w * fill, 0, 0.01)
        glVertex3f(-bar_w/2 + bar_w * fill, bar_h, 0.01)
        glVertex3f(-bar_w/2, bar_h, 0.01)
        glEnd()

        glPopMatrix()
        glEnable(GL_LIGHTING)

    def draw_hud(self):
        """Draw HUD overlay."""
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        hud = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # Top bar
        pygame.draw.rect(hud, (15, 20, 30, 235), (0, 0, self.width, 60))

        # Title
        title = self.font_medium.render("5G O-RAN DAS 室內定位與跌倒偵測系統", True, (200, 205, 220))
        hud.blit(title, (15, 8))

        # Network status
        active_antennas = sum(1 for a in self.antennas if a.detected_persons)
        net_text = self.font_small.render(f"活躍天線: {active_antennas}/{len(self.antennas)}", True, (100, 180, 100))
        hud.blit(net_text, (15, 35))

        # Positioned count
        positioned = sum(1 for p in self.people if p.positioned)
        pos_text = self.font_small.render(f"定位中: {positioned}", True, (100, 150, 200))
        hud.blit(pos_text, (150, 35))

        # Alert status
        if self.alert_level == AlertLevel.NORMAL:
            status_color = (80, 180, 80)
            status = "● 正常監控"
        elif self.alert_level == AlertLevel.ANOMALY:
            status_color = (220, 220, 80)
            status = "⚠ 異常偵測"
        elif self.alert_level == AlertLevel.WARNING:
            status_color = (230, 150, 50)
            status = "🔴 風險預測"
        else:
            status_color = (230, 70, 70)
            status = "🚨 跌倒警報"

        status_surf = self.font_medium.render(status, True, status_color)
        hud.blit(status_surf, (self.width - status_surf.get_width() - 20, 15))

        # Bottom narration
        if self.current_narration:
            pygame.draw.rect(hud, (15, 20, 30, 225), (0, self.height - 50, self.width, 50))
            narr = self.font_medium.render(self.current_narration, True, (210, 215, 230))
            hud.blit(narr, (15, self.height - 38))

        # Controls
        hint = self.font_small.render("[SPACE] 開始  [F] 跌倒  [ESC] 退出", True, (90, 95, 110))
        hud.blit(hint, (self.width - hint.get_width() - 15, self.height - 35))

        # Convert to texture
        texture_data = pygame.image.tostring(hud, "RGBA", True)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glEnable(GL_TEXTURE_2D)
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
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

        glDeleteTextures([tex_id])
        glDisable(GL_TEXTURE_2D)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def handle_events(self):
        """Handle events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    if not self.demo_started:
                        self.demo_started = True
                    else:
                        self._reset_demo()
                        self.demo_started = True
                elif event.key == pygame.K_f:
                    for p in self.people:
                        if p.state not in [PersonState.FALLING, PersonState.FALLEN]:
                            p.state = PersonState.FALLING
                            p.fall_progress = 0.0
                            self.alert_level = AlertLevel.ALERT
                            self.alert_target = p.id
                            self.camera_shake = 0.4
                            self.fall_count += 1
                            self.current_narration = f"🚨 {p.name} 跌倒！位置: ({p.x:.1f}, {p.z:.1f})"
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
            self.update_sensing(dt)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            shake_x = random.uniform(-1, 1) * self.camera_shake * 0.25
            shake_y = random.uniform(-1, 1) * self.camera_shake * 0.25

            glLoadIdentity()
            gluLookAt(
                self.camera_x + shake_x, self.camera_y, self.camera_z + shake_y,
                self.camera_target[0], self.camera_target[1], self.camera_target[2],
                0, 1, 0
            )

            self.draw_floor()

            for antenna in self.antennas:
                self.draw_antenna(antenna)

            self.draw_wave_pulses()

            for person in self.people:
                self.draw_triangulation(person)
                self.draw_person(person)

            self.draw_hud()

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


def main():
    demo = ORANDASDemo()
    demo.run()


if __name__ == "__main__":
    main()
