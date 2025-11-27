#!/usr/bin/env python3
"""
Load the Chih-Tu-Qi LTC scene into Isaac Sim.
Run this script from within Isaac Sim's Script Editor or via command line.
"""

import os
import carb
import omni.usd

# Scene file path
SCENE_PATH = "/home/thc1006/hsinchu/mmwave-fall-omniverse-demo/chih-tu-qi-ltc-mmwave-sim/sim/usd/chih_tu_qi_floor1_ltc.usd"

def load_scene():
    """Load the USD scene into Isaac Sim."""
    print(f"Loading scene: {SCENE_PATH}")

    if not os.path.exists(SCENE_PATH):
        carb.log_error(f"Scene file not found: {SCENE_PATH}")
        return False

    # Get the USD context
    usd_context = omni.usd.get_context()

    # Open the stage
    result, error = usd_context.open_stage(SCENE_PATH)

    if result:
        print(f"Successfully loaded scene: {SCENE_PATH}")
        carb.log_info(f"Scene loaded: {SCENE_PATH}")
        return True
    else:
        print(f"Failed to load scene: {error}")
        carb.log_error(f"Failed to load scene: {error}")
        return False

if __name__ == "__main__":
    load_scene()
