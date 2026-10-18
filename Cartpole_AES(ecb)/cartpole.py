"""
Tabular Q-learning for CartPole on QUANTIZED + AES-ECB ENCRYPTED states.

Pipeline (the extended MDP of arXiv:2109.08236):

    state (4 floats) --phi--> 4 quantized bin indices --PKCS#7--> 16 bytes
                      --AES-ECB--> ciphertext --> used as the Q-table key

The agent never sees the real state, only the 16-byte ciphertext. AES is a
permutation and ECB is deterministic, so the same quantized state always maps to
the same key and different states never collide. A Q-table therefore learns
exactly as it would on plaintext states (compare --encrypt ecb with
--encrypt none using the same --seed).

CartPole state: [cart position, cart velocity, pole angle, pole angular velocity]
Each value is clipped to a range and quantized into a few bins (--bins).
Default bins 1,1,6,12 -> 72 possible states (cart position and velocity are
ignored with 1 bin each; the pole angle and its angular velocity do the work).
More bins, e.g. 3,3,8,10 (720 states), also solve it but need more episodes (about 700 in one test).

Install:
    pip install gymnasium numpy cryptography
    pip install "gymnasium[classic-control]" moviepy      # only needed for videos

Training stops automatically when the FROZEN table scores >= 475 in a greedy
evaluation (the training score is not used, because the table keeps adapting
while it trains and that hides weaknesses).

Train:
    python cartpole_qtable_aes_ecb.py train --encrypt ecb
    python cartpole_qtable_aes_ecb.py train --encrypt none --model q_plain.pkl

Play (the key saved next to the table is loaded automatically):
    python cartpole_qtable_aes_ecb.py play --model q_cartpole_ecb.pkl
    python cartpole_qtable_aes_ecb.py play --model q_cartpole_ecb.pkl --video-dir videos

NOTE: AES-ECB is deterministic, so it is cryptographically weak. The paper uses
it precisely because the determinism keeps states learnable.
"""

import argparse
import os
import pickle
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from gymnasium.wrappers import RecordVideo

ENV_ID = "CartPole-v1"
N_ACTIONS = 2  # push cart left / push cart right

# Clipping range for each state value: position, velocity, angle (rad), angular velocity
LOW = np.array([-2.4, -3.0, -0.21, -3.5])
HIGH = np.array([2.4, 3.0, 0.21, 3.5])


# --------------------------------------------------------------------------
# phi(S): quantize each continuous value into a small number of bins
# --------------------------------------------------------------------------
class Quantizer:
    """Maps the 4 state values to small integers (bin indices), packed into bytes."""

    def __init__(self, bins):
        if len(bins) != 4 or not all(1 <= b <= 256 for b in bins):
            raise ValueError("--bins needs 4 numbers between 1 and 256, e.g. 1,1,6,12")
        self.bins = np.array(bins, dtype=np.int64)

    def __call__(self, obs):
        x = np.clip(np.asarray(obs, dtype=np.float64), LOW, HIGH)
        frac = (x - LOW) / (HIGH - LOW)
        idx = np.minimum((frac * self.bins).astype(np.int64), self.bins - 1)
        return bytes(idx.astype(np.uint8))  # 4 bytes, one per value

    @property
    def n_states(self):
        return int(np.prod(self.bins))


# --------------------------------------------------------------------------
# Enc(.): PKCS#7 padding (4 -> 16 bytes) + AES-ECB, or plaintext baseline
# --------------------------------------------------------------------------
class AESECBEncoder:
    def __init__(self, key: bytes):
        if len(key) not in (16, 24, 32):
            raise ValueError("AES key must be 16, 24 or 32 bytes")
        self.key = key
        cipher = Cipher(algorithms.AES(key), modes.ECB())
        self._enc = cipher.encryptor()
        self._dec = cipher.decryptor()

    def encrypt(self, plain: bytes) -> bytes:
        padder = padding.PKCS7(128).padder()
        return self._enc.update(padder.update(plain) + padder.finalize())

    def decrypt(self, cipher: bytes) -> bytes:
        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(self._dec.update(cipher)) + unpadder.finalize()


class NoEncryption:
    key = b""

    def encrypt(self, plain: bytes) -> bytes:
        return plain


def key_path(model_path):
    return model_path + ".key"


def build_encryptor(args, load_key=False):
    if args.encrypt == "none":
        return NoEncryption()
    if load_key:  # play mode: must use the key the table was trained with
        with open(key_path(args.model)) as f:
            key = bytes.fromhex(f.read().strip())
    elif args.key_hex:
        key = bytes.fromhex(args.key_hex)
    else:
        key = os.urandom(args.key_size)
    return AESECBEncoder(key)


def parse_bins(text):
    return [int(v) for v in text.split(",")]


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
    env = gym.make(ENV_ID, render_mode=render_mode)
    if video_dir:
        env = RecordVideo(env, video_folder=video_dir,
                          episode_trigger=episode_trigger or (lambda ep: True),
                          name_prefix="cartpole_qtable")
    return env


# --------------------------------------------------------------------------
# Q-table helpers
# --------------------------------------------------------------------------
def best_action(q_row, rng):
    """argmax with random tie-breaking (an all-zero row must not always pick action 0)."""
    best = np.flatnonzero(q_row == q_row.max())
    return int(rng.choice(best))


def lookup(Q, key):
    row = Q.get(key)
    if row is None:
        row = np.zeros(N_ACTIONS, dtype=np.float64)
        Q[key] = row
    return row


def evaluate(Q, encode, env, n_episodes, rng, seed_base):
    """Run the FROZEN table greedily (no learning, no exploration) and return the
    average score. This is what `play` will see, so it is the honest measure:
    the training score is inflated because the table keeps adapting while it plays."""
    scores = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + i)
        total, done = 0.0, False
        while not done:
            row = Q.get(encode(obs))  # unseen state -> random action
            action = int(rng.integers(N_ACTIONS)) if row is None else best_action(row, rng)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        scores.append(total)
    return float(np.mean(scores))


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

    quantizer = Quantizer(parse_bins(args.bins))
    encryptor = build_encryptor(args)
    rng = np.random.default_rng(args.seed)

    if isinstance(encryptor, AESECBEncoder):
        probe = bytes([1, 2, 3, 4])
        assert encryptor.decrypt(encryptor.encrypt(probe)) == probe
        with open(key_path(args.model), "w") as f:
            f.write(encryptor.key.hex())
        print(f"AES-ECB enabled | key {len(encryptor.key)} bytes "
              f"| key saved to {key_path(args.model)}")
    else:
        print("Encryption disabled (plaintext baseline)")
    print(f"Quantization bins {args.bins} -> {quantizer.n_states} possible states")

    def encode(obs):
        return encryptor.encrypt(quantizer(obs))

    Q = {}  # ciphertext (or plaintext) bytes -> [value of push-left, value of push-right]
    eval_env = make_env()  # separate plain env used only for frozen-table evaluation
    eval_rng = np.random.default_rng(args.seed + 12345)
    recent = deque(maxlen=100)
    best_eval = -float("inf")
    epsilon = args.eps_start
    start = time.time()
    total_steps = 0

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)
        key = encode(obs)
        ep_reward, done = 0.0, False

        while not done:
            if rng.random() < epsilon:
                action = int(rng.integers(N_ACTIONS))
            else:
                action = best_action(lookup(Q, key), rng)

            obs, reward, terminated, truncated, _ = env.step(action)
            next_key = encode(obs)
            done = terminated or truncated
            ep_reward += reward

            # Q-learning update. Only a real failure (terminated) cuts the
            # bootstrap; hitting the 500-step time limit (truncated) does not.
            row = lookup(Q, key)
            next_val = 0.0 if terminated else lookup(Q, next_key).max()
            row[action] += args.alpha * (reward + args.gamma * next_val - row[action])

            key = next_key
            total_steps += 1

        recent.append(ep_reward)
        epsilon = max(args.eps_end, epsilon * args.eps_decay)

        if ep % args.log_every == 0:
            sps = total_steps / (time.time() - start)
            print(f"ep {ep:>5} | last {ep_reward:5.0f} | train avg{len(recent)} "
                  f"{np.mean(recent):6.1f} | eps {epsilon:.3f} "
                  f"| states seen {len(Q):>3} | {sps:.0f} steps/s")

        # Judge the frozen table, save the best one, stop when it is good enough
        if ep % args.eval_every == 0:
            score = evaluate(Q, encode, eval_env, args.eval_episodes, eval_rng, 10_000)
            marker = ""
            if score > best_eval:
                best_eval = score
                with open(args.model, "wb") as f:
                    pickle.dump(Q, f)
                marker = "  <- saved"
            print(f"      greedy eval ({args.eval_episodes} eps, frozen table): "
                  f"{score:6.1f}{marker}")
            if score >= args.solve_score:
                print(f"Solved at episode {ep}! greedy eval = {score:.1f}")
                env.close()
                eval_env.close()
                return

    with open(args.model.replace(".pkl", "_final.pkl"), "wb") as f:
        pickle.dump(Q, f)
    env.close()
    eval_env.close()
    best = f"{best_eval:.1f}" if best_eval > -float("inf") else "n/a"
    print(f"Training finished. States visited: {len(Q)} / {quantizer.n_states} "
          f"| best greedy eval: {best}")


# --------------------------------------------------------------------------
# Evaluation / play (uses the same key and bins as training)
# --------------------------------------------------------------------------
def play(args):
    env = make_env(render=not args.video_dir, video_dir=args.video_dir)
    quantizer = Quantizer(parse_bins(args.bins))
    encryptor = build_encryptor(args, load_key=(args.encrypt == "ecb"))
    rng = np.random.default_rng(args.seed)

    with open(args.model, "rb") as f:
        Q = pickle.load(f)

    scores = []
    for ep in range(args.play_episodes):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            key = encryptor.encrypt(quantizer(obs))
            row = Q.get(key)
            action = int(rng.integers(N_ACTIONS)) if row is None else best_action(row, rng)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        scores.append(total)
        print(f"Episode {ep + 1}: score = {total:.0f}")

    print(f"Average score: {np.mean(scores):.1f}  (max possible: 500)")
    env.close()  # finishes writing the video files
    if args.video_dir:
        print(f"Videos saved in: {args.video_dir}/")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "play"])
    p.add_argument("--encrypt", choices=["ecb", "none"], default="ecb")
    p.add_argument("--key-size", type=int, choices=[16, 24, 32], default=32,
                   help="AES key length in bytes")
    p.add_argument("--key-hex", type=str, default=None,
                   help="optional fixed key as hex; random if omitted")
    p.add_argument("--bins", type=str, default="1,1,6,12",
                   help="bins for position,velocity,angle,angular velocity")
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--alpha", type=float, default=0.1, help="learning rate")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--eps-decay", type=float, default=0.995,
                   help="epsilon is multiplied by this after every episode")
    p.add_argument("--solve-score", type=float, default=475.0,
                   help="stop when the frozen-table greedy eval reaches this")
    p.add_argument("--eval-every", type=int, default=50,
                   help="evaluate the frozen table every N training episodes")
    p.add_argument("--eval-episodes", type=int, default=20,
                   help="greedy episodes per evaluation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--model", type=str, default="q_cartpole_ecb.pkl")
    p.add_argument("--play-episodes", type=int, default=5)
    p.add_argument("--video-dir", type=str, default=None,
                   help="folder to save .mp4 videos (train and play)")
    p.add_argument("--record-every", type=int, default=500,
                   help="during training, record every Nth episode")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    train(args) if args.mode == "train" else play(args)