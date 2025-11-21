# /train-model – Train the mmWave fall detection model

The user wants to train the fall vs normal classifier from recorded RTX Radar episodes.

1. Confirm that radar data is expected under `ml/data/normal/` and `ml/data/fall/` as `.npz` files.
2. Show how to run `ml/train_fallnet.py` with reasonable default hyperparameters.
3. Explain where the model artifact (`ml/fallnet.pt`) will be written.
4. If a GPU is available, note that PyTorch will automatically use it; otherwise training falls back to CPU.

End with a fenced shell block containing the exact commands to run training from the project root.
