"""
DQN agent for LunarLander (PyTorch + Gymnasium + Box2D) with video recording,
extended with AES-ECB state encryption (arXiv:2109.08236).

Pipeline:  env state -> phi (256-bin discretisation) -> PKCS#7 + AES-ECB -> agent
All hyperparameters are identical to the original lunarlander_dqn.py.
Lines changed or added for encryption are marked with  # [AES]

Install:
    pip install torch "gymnasium[box2d]" numpy moviepy cryptography

Train (encrypted):
    python lunarlander_dqn_ecb.py train --encrypt ecb
    python lunarlander_dqn_ecb.py train --encrypt ecb --double

Train (plaintext, identical to the original script):
    python lunarlander_dqn_ecb.py train --encrypt none --model dqn_lunarlander.pt

Watch / record (the key saved next to the model is loaded automatically):
    python lunarlander_dqn_ecb.py play --model dqn_lunarlander_ecb.pt
    python lunarlander_dqn_ecb.py play --model dqn_lunarlander_ecb.pt --episodes 3 --video-dir videos

The environment counts as "solved" at an average score of 200 over 100 episodes.
Training stops automatically when that happens.
"""

import argparse
import os  # [AES]
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cryptography.hazmat.primitives import padding  # [AES]
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes  # [AES]
from gymnasium.wrappers import RecordVideo

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# --------------------------------------------------------------------------
# [AES] Encryption step:  Enc(phi(S_t))
# --------------------------------------------------------------------------
class Discretizer:
    """phi(S): maps each of the 8 state values to one byte (0..bins-1)."""

    def __init__(self, low, high, bins=256):
        if not 2 <= bins <= 256:
            raise ValueError("bins must be between 2 and 256")
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.bins = bins

    def __call__(self, obs):
        x = np.clip(np.asarray(obs, dtype=np.float64), self.low, self.high)
        frac = (x - self.low) / (self.high - self.low)
        return np.minimum((frac * self.bins).astype(np.int64),
                          self.bins - 1).astype(np.uint8)


class AESECBEncoder:
    """8 discretised bytes -> PKCS#7 pad to 16 bytes -> AES-ECB -> 16 floats in [0,1]."""
    out_dim = 16

    def __init__(self, disc, key: bytes):
        if len(key) not in (16, 24, 32):
            raise ValueError("AES key must be 16, 24 or 32 bytes")
        self.disc = disc
        self.key = key
        cipher = Cipher(algorithms.AES(key), modes.ECB())
        self._enc = cipher.encryptor()
        self._dec = cipher.decryptor()

    def encrypt_bytes(self, plain: bytes) -> bytes:
        padder = padding.PKCS7(128).padder()
        return self._enc.update(padder.update(plain) + padder.finalize())

    def decrypt_bytes(self, cipher: bytes) -> bytes:
        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(self._dec.update(cipher)) + unpadder.finalize()

    def __call__(self, obs):
        ct = self.encrypt_bytes(self.disc(obs).tobytes())
        return np.frombuffer(ct, dtype=np.uint8).astype(np.float32) / 255.0


class NoEncryption:
    """Plaintext: behaves exactly like the original script."""
    out_dim = 8
    key = b""

    def __call__(self, obs):
        return np.asarray(obs, dtype=np.float32)


def key_path(model_path):
    return model_path + ".key"


def build_encoder(args, env, load_key=False):
    if args.encrypt == "none":
        return NoEncryption()

    disc = Discretizer(env.observation_space.low, env.observation_space.high, args.bins)
    if load_key:  # play mode: must reuse the key the agent trained with
        with open(key_path(args.model)) as f:
            key = bytes.fromhex(f.read().strip())
    elif args.key_hex:
        key = bytes.fromhex(args.key_hex)
    else:
        key = os.urandom(args.key_size)
    return AESECBEncoder(disc, key)


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
def make_env(render=False, video_dir=None, episode_trigger=None):
    if video_dir:
        render_mode = "rgb_array"
    elif render:
        render_mode = "human"
    else:
        render_mode = None

    try:
        env = gym.make("LunarLander-v3", render_mode=render_mode)
    except gym.error.Error:  # older gymnasium versions
        env = gym.make("LunarLander-v2", render_mode=render_mode)

    if video_dir:
        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=episode_trigger or (lambda ep: True),
            name_prefix="lunarlander",
        )
    return env


# --------------------------------------------------------------------------
# Network: encoded state -> Q-value for each of the 4 actions
# (do nothing, fire left engine, fire main engine, fire right engine)
# --------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------
# Replay buffer (now stores the ENCRYPTED states)
# --------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.idx = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, s, a, r, s2, done):
        self.states[self.idx] = s
        self.next_states[self.idx] = s2
        self.actions[self.idx] = a
        self.rewards[self.idx] = r
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ids = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[ids]).to(DEVICE),
            torch.from_numpy(self.actions[ids]).to(DEVICE),
            torch.from_numpy(self.rewards[ids]).to(DEVICE),
            torch.from_numpy(self.next_states[ids]).to(DEVICE),
            torch.from_numpy(self.dones[ids]).to(DEVICE),
        )

    def __len__(self):
        return self.size


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(args):
    if args.video_dir:
        every = args.record_every
        env = make_env(video_dir=args.video_dir,
                       episode_trigger=lambda ep: ep % every == 0)
    else:
        env = make_env()

    n_actions = env.action_space.n

    encoder = build_encoder(args, env)  # [AES]
    state_dim = encoder.out_dim  # [AES] 16 ciphertext bytes (8 if plaintext)

    if isinstance(encoder, AESECBEncoder):  # [AES] sanity check + save key
        test_obs, _ = env.reset(seed=args.seed)
        plain = encoder.disc(test_obs).tobytes()
        assert encoder.decrypt_bytes(encoder.encrypt_bytes(plain)) == plain
        with open(key_path(args.model), "w") as f:
            f.write(encoder.key.hex())
        print(f"AES-ECB enabled | key {len(encoder.key)} bytes | {args.bins} bins "
              f"| key saved to {key_path(args.model)}")
    else:
        print("Encryption disabled (plaintext baseline)")

    policy_net = QNetwork(state_dim, n_actions).to(DEVICE)
    target_net = QNetwork(state_dim, n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size, state_dim)

    obs, _ = env.reset(seed=args.seed)
    obs = encoder(obs)  # [AES] the agent only ever sees ciphertext
    episode_reward = 0.0
    recent = deque(maxlen=100)
    episodes, best_avg = 0, -float("inf")
    start = time.time()

    print(f"Device: {DEVICE} | State dim: {state_dim} | Actions: {n_actions}")

    for step in range(1, args.steps + 1):
        # Linear epsilon decay
        frac = min(1.0, step / args.eps_decay_steps)
        epsilon = args.eps_start + frac * (args.eps_end - args.eps_start)

        # Epsilon-greedy action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
                action = policy_net(t).argmax(dim=1).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = encoder(next_obs)  # [AES] Enc(phi(S_{t+1})); reward stays plaintext
        done = terminated or truncated
        episode_reward += reward

        # Only a true terminal (crash/land) cuts the bootstrap, not a time-limit
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs

        if done:
            episodes += 1
            recent.append(episode_reward)
            avg = np.mean(recent)

            if episodes % 10 == 0:
                sps = step / (time.time() - start)
                print(f"step {step:>7} | ep {episodes:>5} | last {episode_reward:7.1f} "
                      f"| avg100 {avg:7.1f} | eps {epsilon:.3f} | {sps:.0f} steps/s")

            if len(recent) == 100 and avg > best_avg:
                best_avg = avg
                torch.save(policy_net.state_dict(), args.model)

            if len(recent) == 100 and avg >= args.solve_score:
                print(f"Solved at step {step}, episode {episodes}! "
                      f"avg100 = {avg:.1f}")
                torch.save(policy_net.state_dict(), args.model)
                env.close()
                return

            episode_reward = 0.0
            obs, _ = env.reset()
            obs = encoder(obs)  # [AES]

        # Learn
        if len(buffer) >= args.learning_starts and step % args.train_freq == 0:
            s, a, r, s2, d = buffer.sample(args.batch_size)

            q = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                if args.double:
                    # Double DQN: policy net picks the action, target net scores it
                    best_a = policy_net(s2).argmax(dim=1, keepdim=True)
                    next_q = target_net(s2).gather(1, best_a).squeeze(1)
                else:
                    next_q = target_net(s2).max(dim=1).values
                target = r + args.gamma * next_q * (1.0 - d)

            loss = F.smooth_l1_loss(q, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
            optimizer.step()

            # Soft update of the target network
            with torch.no_grad():
                for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
                    tp.data.mul_(1.0 - args.tau).add_(args.tau * pp.data)

    torch.save(policy_net.state_dict(), args.model.replace(".pt", "_final.pt"))
    env.close()
    best = f"{best_avg:.1f}" if best_avg > -float("inf") else "n/a"
    print(f"Training finished. Best avg100: {best}")


# --------------------------------------------------------------------------
# Evaluation / play
# --------------------------------------------------------------------------
def play(args):
    env = make_env(render=not args.video_dir, video_dir=args.video_dir)

    encoder = build_encoder(args, env, load_key=(args.encrypt == "ecb"))  # [AES]
    state_dim = encoder.out_dim  # [AES]
    net = QNetwork(state_dim, env.action_space.n).to(DEVICE)
    net.load_state_dict(torch.load(args.model, map_location=DEVICE))
    net.eval()

    scores = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        obs = encoder(obs)  # [AES]
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
                action = net(t).argmax(dim=1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            obs = encoder(obs)  # [AES]
            total += reward
            done = terminated or truncated
        scores.append(total)
        print(f"Episode {ep + 1}: score = {total:.1f}")

    print(f"Average score: {np.mean(scores):.1f}")
    env.close()  # finishes writing the video files
    if args.video_dir:
        print(f"Videos saved in: {args.video_dir}/")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "play"])
    p.add_argument("--encrypt", choices=["ecb", "none"], default="ecb")  # [AES]
    p.add_argument("--bins", type=int, default=256,
                   help="discretisation levels per value (2-256)")  # [AES]
    p.add_argument("--key-size", type=int, choices=[16, 24, 32], default=32,
                   help="AES key length in bytes")  # [AES]
    p.add_argument("--key-hex", type=str, default=None,
                   help="optional fixed key as hex; random if omitted")  # [AES]
    # ---- everything below is unchanged from the original script ----
    p.add_argument("--steps", type=int, default=400_000)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--tau", type=float, default=0.005,
                   help="soft target-update rate")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--eps-decay-steps", type=int, default=100_000)
    p.add_argument("--learning-starts", type=int, default=1_000)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--solve-score", type=float, default=200.0)
    p.add_argument("--double", action="store_true",
                   help="use Double DQN targets")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", type=str, default="dqn_lunarlander_ecb.pt")  # [AES] new default name
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--video-dir", type=str, default=None,
                   help="folder to save .mp4 videos (train and play)")
    p.add_argument("--record-every", type=int, default=50,
                   help="during training, record every Nth episode")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(args) if args.mode == "train" else play(args)