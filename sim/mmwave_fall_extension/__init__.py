import omni.ext

from . import scene
from .ui import FallNotificationUI


class MmwaveFallExtension(omni.ext.IExt):
    """Entry point for the mmWave fall detection Omniverse extension.

    When the extension is enabled from the Extension Manager, `on_startup` is called.
    We use this hook to set up the hall scene, avatar, RTX Radar sensors and a
    simple in-scene notification UI.
    """

    def on_startup(self, ext_id: str):
        print(f"[mmwave.fall.extension] Starting extension {ext_id}")
        try:
            self._ui = FallNotificationUI()
            scene.setup_scene()
        except Exception as exc:
            print(f"[mmwave.fall.extension] Failed to set up scene: {exc}")

    def on_shutdown(self):
        print("[mmwave.fall.extension] Shutting down extension")
        # UI will be garbage-collected when the extension is disabled.
