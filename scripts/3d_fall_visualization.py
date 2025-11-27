#!/usr/bin/env python3
"""3D Fall Detection Visualization for mmWave System.

Real-time 3D visualization of:
- Facility floor plan (赤土崎多功能館)
- Animated avatars (elderly people)
- mmWave radar coverage cones
- Fall detection alerts with visual effects

Uses PyGame + OpenGL for hardware-accelerated 3D rendering.
"""

import math
import random
import time
import threading
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import requests

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

import sys


def draw_sphere(radius, slices=16, stacks=16):
    """Draw a sphere without GLUT."""
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
            x = math.cos(lng)
            y = math.sin(lng)

            glNormal3f(x * zr0, y * zr0, z0)
            glVertex3f(x * zr0, y * zr0, z0)
            glNormal3f(x * zr1, y * zr1, z1)
            glVertex3f(x * zr1, y * zr1, z1)
        glEnd()


def draw_cube(size):
    """Draw a cube without GLUT."""
    s = size / 2
    vertices = [
        (-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
        (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)
    ]
    faces = [
        (0, 1, 2, 3), (4, 5, 6, 7),
        (0, 1, 5, 4), (2, 3, 7, 6),
        (0, 3, 7, 4), (1, 2, 6, 5)
    ]
    normals = [
        (0, 0, -1), (0, 0, 1),
        (0, -1, 0), (0, 1, 0),
        (-1, 0, 0), (1, 0, 0)
    ]

    glBegin(GL_QUADS)
    for i, face in enumerate(faces):
        glNormal3fv(normals[i])
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()


@dataclass
class Avatar:
    """Represents a person in the facility."""
    id: str
    name: str
    x: float
    y: float
    z: float
    color: Tuple[float, float, float]
    status: str = "normal"  # normal, fall, rehab_bad_posture, chest_abnormal
    fall_progress: float = 0.0
    animation_time: float = 0.0


@dataclass
class RadarCone:
    """mmWave radar visualization."""
    id: str
    x: float
    y: float
    z: float
    fov_h: float
    fov_v: float
    range_max: float
    rotation: float = 0.0


@dataclass
class FallEvent:
    """Fall detection event."""
    id: str
    x: float
    y: float
    label: str
    confidence: float
    timestamp: float


class FallDetectionVisualizer:
    """3D visualization for mmWave fall detection system."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.width = 1280
        self.height = 720
        self.running = True

        # Camera
        self.camera_distance = 25.0
        self.camera_angle_h = 45.0
        self.camera_angle_v = 35.0
        self.camera_target = [10.0, 0.0, 6.0]

        # Facility dimensions (meters)
        self.facility_width = 20.0
        self.facility_depth = 12.5
        self.wall_height = 3.0

        # Zones based on 赤土崎多功能館 layout
        self.zones = [
            {"id": "dementia_corridor", "rect": (0, 0, 12, 2.5), "color": (0.8, 0.3, 0.3), "name": "失智遊走走廊"},
            {"id": "activity_a", "rect": (0, 2.5, 6, 5), "color": (0.3, 0.7, 0.3), "name": "活動室A"},
            {"id": "activity_b", "rect": (6, 2.5, 6, 5), "color": (0.3, 0.6, 0.7), "name": "活動室B"},
            {"id": "dining", "rect": (12, 0, 8, 5), "color": (0.7, 0.7, 0.3), "name": "餐廳"},
            {"id": "rest", "rect": (0, 7.5, 6, 5), "color": (0.5, 0.5, 0.8), "name": "休息區"},
            {"id": "rehab", "rect": (6, 7.5, 6, 5), "color": (0.8, 0.5, 0.3), "name": "復健室"},
            {"id": "nursing", "rect": (12, 7.5, 4, 2.5), "color": (0.4, 0.8, 0.4), "name": "護理站"},
            {"id": "bathroom", "rect": (16, 7.5, 4, 2.5), "color": (0.3, 0.5, 0.7), "name": "無障礙廁所"},
            {"id": "entrance", "rect": (12, 5, 8, 2.5), "color": (0.6, 0.6, 0.6), "name": "入口大廳"},
        ]

        # Radars
        self.radars = [
            RadarCone("radar_1", 6.0, 2.8, 12.0, 120.0, 60.0, 8.0, 180),
            RadarCone("radar_2", 16.0, 2.8, 2.5, 120.0, 60.0, 8.0, 90),
            RadarCone("radar_3", 9.0, 2.8, 10.0, 120.0, 60.0, 8.0, 270),
        ]

        # Avatars
        self.avatars: List[Avatar] = []
        self._init_avatars()

        # Events
        self.events: List[FallEvent] = []
        self.alert_active = False
        self.alert_time = 0.0

        # Animation
        self.time = 0.0
        self.last_api_fetch = 0.0

        # Start API polling thread
        self.api_thread = threading.Thread(target=self._poll_api, daemon=True)
        self.api_thread.start()

    def _init_avatars(self):
        """Initialize avatars in the facility."""
        avatar_data = [
            ("elder_1", "王阿公", 3.0, 4.0, (0.9, 0.7, 0.5)),
            ("elder_2", "李阿嬤", 8.0, 9.0, (0.8, 0.6, 0.5)),
            ("elder_3", "陳伯伯", 14.0, 2.0, (0.7, 0.6, 0.5)),
            ("elder_4", "林奶奶", 2.0, 9.0, (0.85, 0.65, 0.5)),
            ("elder_5", "張爺爺", 9.0, 9.5, (0.75, 0.55, 0.45)),
            ("caregiver", "護理師", 13.0, 8.5, (0.4, 0.6, 0.8)),
        ]

        for aid, name, x, z, color in avatar_data:
            self.avatars.append(Avatar(
                id=aid, name=name, x=x, y=0, z=z, color=color
            ))

    def _poll_api(self):
        """Poll API for events in background thread."""
        while self.running:
            try:
                response = requests.get(f"{self.api_url}/events/recent", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    events = data.get("events", [])

                    # Process new events
                    for ev in events[-10:]:  # Last 10 events
                        # Map position to facility coordinates
                        x = (ev["position"]["x"] + 10) * 1.0  # Map from -10,10 to 0,20
                        z = (ev["position"]["y"] + 5) * 1.25  # Map to facility depth

                        # Update avatar if fall detected
                        if ev["label"] == "fall":
                            self._trigger_fall_event(x, z, ev)

            except Exception as e:
                pass  # Silent fail for API errors

            time.sleep(1.0)

    def _trigger_fall_event(self, x: float, z: float, event_data: dict):
        """Trigger fall event visualization."""
        # Find nearest avatar or create temporary one
        nearest_avatar = None
        min_dist = float('inf')

        for avatar in self.avatars:
            dist = math.sqrt((avatar.x - x)**2 + (avatar.z - z)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_avatar = avatar

        if nearest_avatar and min_dist < 3.0:
            nearest_avatar.status = "fall"
            nearest_avatar.fall_progress = 0.0
            nearest_avatar.x = x
            nearest_avatar.z = z

        # Trigger alert
        self.alert_active = True
        self.alert_time = time.time()

        # Add event marker
        self.events.append(FallEvent(
            id=event_data.get("id", str(time.time())),
            x=x, y=z,
            label=event_data.get("label", "fall"),
            confidence=max(event_data.get("probabilities", {}).values()) if event_data.get("probabilities") else 0.5,
            timestamp=time.time()
        ))

        # Keep only recent events
        if len(self.events) > 20:
            self.events = self.events[-20:]

    def init_gl(self):
        """Initialize OpenGL settings."""
        glClearColor(0.1, 0.1, 0.15, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Light setup
        glLightfv(GL_LIGHT0, GL_POSITION, [10.0, 15.0, 10.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])

        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def update_camera(self):
        """Update camera position."""
        glLoadIdentity()

        # Calculate camera position from angles
        rad_h = math.radians(self.camera_angle_h)
        rad_v = math.radians(self.camera_angle_v)

        cam_x = self.camera_target[0] + self.camera_distance * math.cos(rad_v) * math.sin(rad_h)
        cam_y = self.camera_target[1] + self.camera_distance * math.sin(rad_v)
        cam_z = self.camera_target[2] + self.camera_distance * math.cos(rad_v) * math.cos(rad_h)

        gluLookAt(
            cam_x, cam_y, cam_z,
            self.camera_target[0], self.camera_target[1], self.camera_target[2],
            0, 1, 0
        )

    def draw_floor(self):
        """Draw the facility floor with zones."""
        glDisable(GL_LIGHTING)

        # Main floor
        glColor4f(0.2, 0.2, 0.25, 1.0)
        glBegin(GL_QUADS)
        glVertex3f(0, -0.01, 0)
        glVertex3f(self.facility_width, -0.01, 0)
        glVertex3f(self.facility_width, -0.01, self.facility_depth)
        glVertex3f(0, -0.01, self.facility_depth)
        glEnd()

        # Zone floors with colors
        for zone in self.zones:
            x, z, w, d = zone["rect"]
            r, g, b = zone["color"]

            glColor4f(r * 0.4, g * 0.4, b * 0.4, 0.8)
            glBegin(GL_QUADS)
            glVertex3f(x + 0.05, 0.0, z + 0.05)
            glVertex3f(x + w - 0.05, 0.0, z + 0.05)
            glVertex3f(x + w - 0.05, 0.0, z + d - 0.05)
            glVertex3f(x + 0.05, 0.0, z + d - 0.05)
            glEnd()

            # Zone border
            glColor4f(r, g, b, 1.0)
            glLineWidth(2.0)
            glBegin(GL_LINE_LOOP)
            glVertex3f(x, 0.01, z)
            glVertex3f(x + w, 0.01, z)
            glVertex3f(x + w, 0.01, z + d)
            glVertex3f(x, 0.01, z + d)
            glEnd()

        # Grid lines
        glColor4f(0.3, 0.3, 0.35, 0.5)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        for i in range(int(self.facility_width) + 1):
            glVertex3f(i, 0.02, 0)
            glVertex3f(i, 0.02, self.facility_depth)
        for i in range(int(self.facility_depth) + 1):
            glVertex3f(0, 0.02, i)
            glVertex3f(self.facility_width, 0.02, i)
        glEnd()

        glEnable(GL_LIGHTING)

    def draw_walls(self):
        """Draw facility walls."""
        glEnable(GL_LIGHTING)
        glColor4f(0.6, 0.6, 0.65, 0.3)

        # Outer walls (semi-transparent)
        wall_thickness = 0.1

        # North wall
        glPushMatrix()
        glTranslatef(self.facility_width / 2, self.wall_height / 2, 0)
        glScalef(self.facility_width, self.wall_height, wall_thickness)
        draw_cube(1.0)
        glPopMatrix()

        # South wall
        glPushMatrix()
        glTranslatef(self.facility_width / 2, self.wall_height / 2, self.facility_depth)
        glScalef(self.facility_width, self.wall_height, wall_thickness)
        draw_cube(1.0)
        glPopMatrix()

        # West wall
        glPushMatrix()
        glTranslatef(0, self.wall_height / 2, self.facility_depth / 2)
        glScalef(wall_thickness, self.wall_height, self.facility_depth)
        draw_cube(1.0)
        glPopMatrix()

        # East wall
        glPushMatrix()
        glTranslatef(self.facility_width, self.wall_height / 2, self.facility_depth / 2)
        glScalef(wall_thickness, self.wall_height, self.facility_depth)
        draw_cube(1.0)
        glPopMatrix()

    def draw_radar(self, radar: RadarCone):
        """Draw radar cone visualization."""
        glDisable(GL_LIGHTING)

        # Radar unit (box)
        glPushMatrix()
        glTranslatef(radar.x, radar.y, radar.z)
        glColor4f(0.2, 0.5, 0.9, 1.0)
        draw_cube(0.3)
        glPopMatrix()

        # Radar cone (coverage area)
        glPushMatrix()
        glTranslatef(radar.x, radar.y, radar.z)
        glRotatef(radar.rotation, 0, 1, 0)

        # Pulsing effect
        pulse = 0.5 + 0.2 * math.sin(self.time * 3)

        # Draw cone as triangular fan
        glColor4f(0.2, 0.6, 1.0, 0.15 * pulse)

        fov_rad = math.radians(radar.fov_h / 2)
        segments = 20

        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, 0, 0)  # Apex
        for i in range(segments + 1):
            angle = -fov_rad + (2 * fov_rad * i / segments)
            x = radar.range_max * math.sin(angle)
            z = -radar.range_max * math.cos(angle)
            glVertex3f(x, -radar.y + 0.1, z)
        glEnd()

        # Cone outline
        glColor4f(0.3, 0.7, 1.0, 0.5)
        glLineWidth(2.0)
        glBegin(GL_LINE_STRIP)
        glVertex3f(0, 0, 0)
        for i in range(segments + 1):
            angle = -fov_rad + (2 * fov_rad * i / segments)
            x = radar.range_max * math.sin(angle)
            z = -radar.range_max * math.cos(angle)
            glVertex3f(x, -radar.y + 0.1, z)
        glVertex3f(0, 0, 0)
        glEnd()

        glPopMatrix()
        glEnable(GL_LIGHTING)

    def draw_avatar(self, avatar: Avatar):
        """Draw an avatar (person)."""
        glEnable(GL_LIGHTING)

        r, g, b = avatar.color

        # Fall animation
        if avatar.status == "fall":
            avatar.fall_progress = min(1.0, avatar.fall_progress + 0.02)
            rotation = avatar.fall_progress * 90  # Rotate to lying down

            glPushMatrix()
            glTranslatef(avatar.x, avatar.y + 0.8 * (1 - avatar.fall_progress * 0.5), avatar.z)
            glRotatef(rotation, 1, 0, 0)

            # Red tint when fallen
            glColor4f(min(1.0, r + 0.3), g * 0.5, b * 0.5, 1.0)
        else:
            glPushMatrix()
            glTranslatef(avatar.x, avatar.y, avatar.z)

            # Idle animation (slight bobbing)
            bob = 0.02 * math.sin(self.time * 2 + hash(avatar.id) % 10)
            glTranslatef(0, bob, 0)

            glColor4f(r, g, b, 1.0)

        # Body (cylinder)
        glPushMatrix()
        glTranslatef(0, 0.5, 0)
        glRotatef(-90, 1, 0, 0)

        # Draw cylinder manually using quads
        slices = 16
        height = 0.8
        radius = 0.25

        glBegin(GL_QUAD_STRIP)
        for i in range(slices + 1):
            angle = 2 * math.pi * i / slices
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            glNormal3f(math.cos(angle), 0, math.sin(angle))
            glVertex3f(x, 0, z)
            glVertex3f(x, height, z)
        glEnd()
        glPopMatrix()

        # Head (sphere)
        glPushMatrix()
        glTranslatef(0, 1.4, 0)
        draw_sphere(0.2, 16, 16)
        glPopMatrix()

        # Arms
        glPushMatrix()
        glTranslatef(-0.35, 0.9, 0)
        glRotatef(30, 0, 0, 1)
        glScalef(0.1, 0.5, 0.1)
        draw_cube(1.0)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(0.35, 0.9, 0)
        glRotatef(-30, 0, 0, 1)
        glScalef(0.1, 0.5, 0.1)
        draw_cube(1.0)
        glPopMatrix()

        # Legs
        glPushMatrix()
        glTranslatef(-0.12, 0.25, 0)
        glScalef(0.12, 0.5, 0.12)
        draw_cube(1.0)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(0.12, 0.25, 0)
        glScalef(0.12, 0.5, 0.12)
        draw_cube(1.0)
        glPopMatrix()

        glPopMatrix()

        # Status indicator ring
        if avatar.status == "fall":
            glDisable(GL_LIGHTING)
            glPushMatrix()
            glTranslatef(avatar.x, 0.05, avatar.z)

            # Pulsing red ring
            pulse = 0.5 + 0.5 * math.sin(self.time * 5)
            ring_radius = 0.8 + 0.2 * pulse

            glColor4f(1.0, 0.2, 0.2, 0.8 * pulse)
            glLineWidth(4.0)
            glBegin(GL_LINE_LOOP)
            for i in range(32):
                angle = 2 * math.pi * i / 32
                glVertex3f(ring_radius * math.cos(angle), 0, ring_radius * math.sin(angle))
            glEnd()

            glPopMatrix()
            glEnable(GL_LIGHTING)

    def draw_event_markers(self):
        """Draw markers for fall events."""
        glDisable(GL_LIGHTING)

        current_time = time.time()

        for event in self.events:
            age = current_time - event.timestamp
            if age > 30:  # Fade out after 30 seconds
                continue

            alpha = max(0, 1.0 - age / 30.0)

            if event.label == "fall":
                glColor4f(1.0, 0.2, 0.2, alpha * 0.6)
            else:
                glColor4f(0.2, 0.8, 0.2, alpha * 0.4)

            # Marker on floor
            glPushMatrix()
            glTranslatef(event.x, 0.03, event.y)

            # Expanding ring
            ring_size = 0.5 + age * 0.1
            glLineWidth(3.0)
            glBegin(GL_LINE_LOOP)
            for i in range(24):
                angle = 2 * math.pi * i / 24
                glVertex3f(ring_size * math.cos(angle), 0, ring_size * math.sin(angle))
            glEnd()

            glPopMatrix()

        glEnable(GL_LIGHTING)

    def draw_alert_overlay(self):
        """Draw alert overlay when fall detected."""
        if not self.alert_active:
            return

        alert_duration = 3.0
        elapsed = time.time() - self.alert_time

        if elapsed > alert_duration:
            self.alert_active = False
            return

        # Flash effect
        flash = 0.5 + 0.5 * math.sin(elapsed * 10)
        alpha = (1.0 - elapsed / alert_duration) * 0.3 * flash

        # Switch to 2D overlay mode
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Red overlay
        glColor4f(1.0, 0.0, 0.0, alpha)
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(self.width, 0)
        glVertex2f(self.width, self.height)
        glVertex2f(0, self.height)
        glEnd()

        # Border
        glColor4f(1.0, 0.2, 0.2, alpha * 3)
        glLineWidth(10.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(5, 5)
        glVertex2f(self.width - 5, 5)
        glVertex2f(self.width - 5, self.height - 5)
        glVertex2f(5, self.height - 5)
        glEnd()

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def draw_hud(self):
        """Draw heads-up display with stats."""
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Title bar
        glColor4f(0.1, 0.1, 0.15, 0.8)
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(self.width, 0)
        glVertex2f(self.width, 50)
        glVertex2f(0, 50)
        glEnd()

        # Stats bar
        glColor4f(0.1, 0.1, 0.15, 0.8)
        glBegin(GL_QUADS)
        glVertex2f(0, self.height - 40)
        glVertex2f(self.width, self.height - 40)
        glVertex2f(self.width, self.height)
        glVertex2f(0, self.height)
        glEnd()

        # Event count indicators
        fall_count = sum(1 for e in self.events if e.label == "fall" and time.time() - e.timestamp < 60)

        # Status indicator
        if fall_count > 0:
            glColor4f(1.0, 0.3, 0.3, 1.0)
        else:
            glColor4f(0.3, 0.8, 0.3, 1.0)

        glBegin(GL_QUADS)
        glVertex2f(self.width - 30, 15)
        glVertex2f(self.width - 10, 15)
        glVertex2f(self.width - 10, 35)
        glVertex2f(self.width - 30, 35)
        glEnd()

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def handle_events(self):
        """Handle PyGame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_f:
                    # Trigger test fall
                    avatar = random.choice(self.avatars)
                    avatar.status = "fall"
                    avatar.fall_progress = 0.0
                    self.alert_active = True
                    self.alert_time = time.time()
                elif event.key == pygame.K_r:
                    # Reset all avatars
                    for av in self.avatars:
                        av.status = "normal"
                        av.fall_progress = 0.0
                elif event.key == pygame.K_LEFT:
                    self.camera_angle_h += 5
                elif event.key == pygame.K_RIGHT:
                    self.camera_angle_h -= 5
                elif event.key == pygame.K_UP:
                    self.camera_angle_v = min(80, self.camera_angle_v + 5)
                elif event.key == pygame.K_DOWN:
                    self.camera_angle_v = max(10, self.camera_angle_v - 5)
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.camera_distance = max(10, self.camera_distance - 2)
                elif event.key == pygame.K_MINUS:
                    self.camera_distance = min(50, self.camera_distance + 2)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Scroll up
                    self.camera_distance = max(10, self.camera_distance - 1)
                elif event.button == 5:  # Scroll down
                    self.camera_distance = min(50, self.camera_distance + 1)

    def run(self):
        """Main loop."""
        pygame.init()
        pygame.display.set_caption("mmWave Fall Detection - 3D Visualization | 赤土崎多功能館")

        # Set display
        display = (self.width, self.height)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

        self.init_gl()

        clock = pygame.time.Clock()

        print("=" * 60)
        print("mmWave Fall Detection 3D Visualization")
        print("=" * 60)
        print("Controls:")
        print("  Arrow keys: Rotate camera")
        print("  +/-: Zoom in/out")
        print("  Mouse wheel: Zoom")
        print("  F: Trigger test fall")
        print("  R: Reset all avatars")
        print("  ESC: Quit")
        print("=" * 60)

        while self.running:
            self.time = pygame.time.get_ticks() / 1000.0

            self.handle_events()

            # Update avatar positions slightly (wandering)
            for avatar in self.avatars:
                if avatar.status == "normal":
                    avatar.x += random.uniform(-0.01, 0.01)
                    avatar.z += random.uniform(-0.01, 0.01)
                    # Keep within bounds
                    avatar.x = max(0.5, min(self.facility_width - 0.5, avatar.x))
                    avatar.z = max(0.5, min(self.facility_depth - 0.5, avatar.z))

            # Clear and draw
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            self.update_camera()

            self.draw_floor()
            self.draw_walls()

            for radar in self.radars:
                self.draw_radar(radar)

            for avatar in self.avatars:
                self.draw_avatar(avatar)

            self.draw_event_markers()
            self.draw_alert_overlay()
            self.draw_hud()

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="3D Fall Detection Visualization")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API server URL")
    args = parser.parse_args()

    viz = FallDetectionVisualizer(api_url=args.api_url)
    viz.run()


if __name__ == "__main__":
    main()
