# /run-sim – Run Isaac Sim data capture for mmWave fall demo

The user wants to record synthetic RTX Radar data for the mmWave fall detection demo.

1. Briefly remind them that this script must be run using the **Isaac Sim Python** entrypoint
   (for example `./python.sh` in the Isaac Sim install directory).
2. Show how to:

   - Enable the `mmwave.fall.extension` extension from the Omniverse Extension Manager, if needed.
   - Run `sim/mmwave_fall_extension/record_fall_data.py` in headless mode to populate `ml/data/normal/`
     and `ml/data/fall/` with `.npz` episodes.

3. Provide sample commands in a fenced shell block, using placeholder Isaac Sim paths that the
   user can adapt to their environment.
4. If appropriate, suggest that they edit `record_fall_data.py` to wire in real RTX Radar buffers
   based on the official `rtx_radar.py` example.
