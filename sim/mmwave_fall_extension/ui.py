from __future__ import annotations

import carb
import omni.ui as ui


class FallNotificationUI:
    """Simple omni.ui window to show fall-related notifications in-scene.

    This is not wired to the FastAPI backend automatically; instead, provide
    lightweight methods that your extension or scripts can call when a fall
    or other abnormal event is detected.
    """

    def __init__(self):
        self._window = ui.Window(
            title="Fall Detection Status",
            width=320,
            height=120,
            visible=True,
            dockPreference=ui.DockPreference.TOP,
        )
        with self._window.frame:
            with ui.VStack(spacing=4, height=0):
                ui.Label("mmWave Fall Detection", style={"font_size": 18})
                self._status = ui.Label("Waiting for events...", style={"color": 0xFFAAAAAA})
                self._detail = ui.Label("", word_wrap=True)

    def show_event(self, label: str, probabilities: dict, position: tuple[float, float, float] | None = None):
        """Update the UI with a new event.

        Parameters
        ----------
        label:
            Predicted label, e.g. "fall", "rehab_bad_posture", "chest_abnormal", "normal".
        probabilities:
            Mapping from label -> probability (0–1).
        position:
            Optional (x, y, z) world coordinates for display.
        """
        if label == "fall":
            color = 0xFFFF5555
            prefix = "⚠ 跌倒偵測 (FALL DETECTED)"
        elif label == "rehab_bad_posture":
            color = 0xFFFFA500
            prefix = "⚠ 復健姿勢異常 (REHAB POSTURE)"
        elif label == "chest_abnormal":
            color = 0xFF22D3EE
            prefix = "⚠ 胸腔異常 (CHEST ABNORMALITY)"
        else:
            color = 0xFF22C55E
            prefix = "✓ 正常活動 (NORMAL)"

        self._status.text = prefix
        self._status.style = {"color": color}

        best_label = max(probabilities.items(), key=lambda kv: kv[1])[0] if probabilities else label
        best_prob = probabilities.get(best_label, 0.0) if probabilities else 1.0

        pos_text = ""
        if position is not None:
            x, y, z = position
            pos_text = f" @ (x={x:.1f}, y={y:.1f}, z={z:.1f})"

        self._detail.text = f"label={label}, p({best_label})={best_prob:.3f}{pos_text}"

        carb.log_info(f"[mmwave.fall.ui] {prefix} {pos_text}")
