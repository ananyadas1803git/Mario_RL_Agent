# 🍄 Super Mario Bros Agent (Reinforcement Learning Model)

An end-to-end Deep Reinforcement Learning project training an autonomous agent to play **Super Mario Bros (NES)** using **Proximal Policy Optimization (PPO)** and an Actor-Critic Convolutional Neural Network (`CnnPolicy`).



https://github.com/user-attachments/assets/a1edd785-1c44-4dca-9e3f-9a7e336b353e



---

## 📌 Project Overview

This project implements a complete pipeline that ingests raw NES visual frames, applies preprocessing and temporal frame-stacking wrappers, and trains an agent using Stable-Baselines3.

### Key Highlights
- **Discretized Action Space:** Mapped the full NES controller to a reduced set of 7 directional and jumping combinations using `SIMPLE_MOVEMENT`.
- **Dimensionality Reduction:** Downsampled frames from $240 \times 256 \times 3$ to $84 \times 84 \times 1$ grayscale luminance arrays via `WarpFrame`, cutting raw tensor computation by over 80%.
- **Temporal Context (Frame Stacking):** Stacked 4 sequential grayscale frames (`VecFrameStack`, shape: `(84, 84, 4)`) with 4-frame skipping and max-pooling (`MaxAndSkipEnv`) to grant the neural network direct awareness of momentum, jump trajectories, and moving enemy vectors.
- **On-Policy Training:** Optimized an Actor-Critic CNN using PPO over 100,000+ timesteps with regular checkpointing and real-time TensorBoard telemetry.
- **Dual Visualizer:** Real-time headless evaluation supporting both native inline Jupyter rendering and an external OpenCV window scaled for recording.

---

## 🛠️ Tech Stack & Dependencies

- **Language & Runtime:** Python 3.10+ / JupyterLab
- **Deep RL Framework:** [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (PPO, `CnnPolicy`)
- **Neural Network Backend:** [PyTorch](https://pytorch.org/)
- **Game Emulation:** `gym-super-mario-bros`, `nes-py`
- **Wrappers & Preprocessing:** `WarpFrame`, `MaxAndSkipEnv`, `DummyVecEnv`, `VecFrameStack`
- **Visualization & Logging:** OpenCV (`cv2`), Matplotlib, TensorBoard, ImageIO

---

## 📂 Repository Structure

```text
├── train/                      # Model checkpoints saved every 10k steps
│   └── mario_model_*.zip
├── logs/                       # TensorBoard scalar event logs
├── Mario_game.ipynb            # Interactive notebook covering pipeline, training & testing
├── mario_ppo_100k.zip          # Saved final weights after 100k steps
├── mario_agent_run.mp4         # High-resolution gameplay recording
└── README.md
