# /run-demo – Run the FastAPI server and describe demo wiring

The user wants to run a complete end-to-end demo:

- Isaac Sim / Omniverse produces RTX Radar data for a hall scene.
- The trained model (`ml/fallnet.pt`) predicts fall vs normal.
- A FastAPI server exposes a `/predict` endpoint.

1. Explain how to start the API server using **uvicorn** from the project root.
2. Show how to send a test request using `curl` or Python (with dummy data) to verify the endpoint.
3. Describe, at a high level, how Omniverse Python code could call this endpoint every time a
   radar episode is ready (without writing full code unless the user asks).

Provide concrete shell commands in fenced code blocks where helpful.
