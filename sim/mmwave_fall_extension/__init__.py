import omni.ext

from . import scene


class MmwaveFallExtension(omni.ext.IExt):
    """Entry point for the mmWave fall detection Omniverse extension.

    When the extension is enabled from the Extension Manager, `on_startup` is called.
    We use this hook to set up the hall scene, avatar, and RTX Radar sensor.
    """

    def on_startup(self, ext_id: str):
        print(f"[mmwave.fall.extension] Starting extension {ext_id}")
        try:
            scene.setup_scene()
        except Exception as exc:
            # We intentionally keep error handling simple here; improve as needed.
            print(f"[mmwave.fall.extension] Failed to set up scene: {exc}")

    def on_shutdown(self):
        print("[mmwave.fall.extension] Shutting down extension")
        # In a more complete implementation you would clean up UI, timelines, etc.
