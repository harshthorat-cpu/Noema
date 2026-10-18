"""
Tabular Q-learning for Blackjack on AES-ECB ENCRYPTED states.

Pipeline (the extended MDP of arXiv:2109.08236):

    state (player_sum, dealer_card, usable_ace) --phi--> 3 bytes --PKCS#7--> 16 bytes
                                   --AES-ECB--> ciphertext --> used as the Q-table key

Blackjack's state is already discrete (about 280 reachable states), so phi is
lossless: nothing needs to be quantized. Because AES is a permutation and ECB is
deterministic, the same state always maps to the same key and different states
never collide, so an encrypted Q-table learns exactly like a plaintext one
(compare --encrypt ecb with --encrypt none using the same --seed).

Rules (Gymnasium Blackjack-v1): actions are 0 = stick, 1 = hit.
Reward is +1 for a win, 0 for a draw, -1 for a loss. Blackjack is a game of
chance, so even a perfect player loses a little on average (about -0.05 per hand).

Install:
    pip install gymnasium numpy cryptography
    pip install "gymnasium[toy-text]" moviepy      # only needed for videos

Train:
    python blackjack_qtable_aes_ecb.py train --encrypt ecb
    python blackjack_qtable_aes_ecb.py train --encrypt none --model q_plain.pkl

Play (the key saved next to the table is loaded automatically):
    python blackjack_qtable_aes_ecb.py play --model q_blackjack_ecb.pkl
    python blackjack_qtable_aes_ecb.py play --model q_blackjack_ecb.pkl --video-dir videos --play-episodes 5

Show the learned strategy as a readable table:
    python blackjack_qtable_aes_ecb.py policy --model q_blackjack_ecb.pkl

NOTE: AES-ECB is deterministic, so it is cryptographically weak. The paper uses
it precisely because the determinism keeps states learnable.
"""

import argparse
import os
import pickle
import random
import time

import gymnasium as gym
import numpy as np
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from gymnasium.wrappers import RecordVideo

ENV_ID = "Blackjack-v1"
N_ACTIONS = 2  # 0 = stick, 1 = hit
ACTION_NAMES = {0: "STICK", 1: "HIT"}


# --------------------------------------------------------------------------
# phi(S): pack the (already discrete) state into 3 bytes
# --------------------------------------------------------------------------
def pack_state(obs):
    player_sum, dealer_card, usable_ace = obs
    return bytes([player_sum, dealer_card, int(usable_ace)])


# --------------------------------------------------------------------------
# Enc(.): PKCS#7 padding (3 -> 16 bytes) + AES-ECB, or plaintext baseline
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


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
def make_env(args, video_dir=None, episode_trigger=None):
    render_mode = "rgb_array" if video_dir else None
    env = gym.make(ENV_ID, natural=args.natural, sab=args.sab, render_mode=render_mode)
    if video_dir:
        env = RecordVideo(env, video_folder=video_dir,
                          episode_trigger=episode_trigger or (lambda ep: True),
                          name_prefix="blackjack_qtable")
    return env


# --------------------------------------------------------------------------
# Q-table helpers
# --------------------------------------------------------------------------
def best_action(q_row, rng):
    """argmax with random tie-breaking (an all-zero row must not always pick STICK)."""
    best = np.flatnonzero(q_row == q_row.max())
    return int(rng.choice(best))


def lookup(Q, key):
    row = Q.get(key)
    if row is None:
        row = np.zeros(N_ACTIONS, dtype=np.float64)
        Q[key] = row
    return row


def evaluate(Q, encode, env, n_episodes, rng):
    """Play the FROZEN table greedily (no learning, no exploration).
    Returns (average reward, win %, draw %, loss %)."""
    wins = draws = losses = 0
    total = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            row = Q.get(encode(obs))  # unseen state -> random action
            action = int(rng.integers(N_ACTIONS)) if row is None else best_action(row, rng)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        total += reward
        wins += reward > 0
        draws += reward == 0
        losses += reward < 0
    n = float(n_episodes)
    return total / n, 100 * wins / n, 100 * draws / n, 100 * losses / n


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(args):
    env = make_env(args)
    eval_env = make_env(args)
    encryptor = build_encryptor(args)
    rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 999)

    if isinstance(encryptor, AESECBEncoder):
        probe = bytes([15, 6, 1])
        assert encryptor.decrypt(encryptor.encrypt(probe)) == probe
        with open(key_path(args.model), "w") as f:
            f.write(encryptor.key.hex())
        print(f"AES-ECB enabled | key {len(encryptor.key)} bytes "
              f"| key saved to {key_path(args.model)}")
    else:
        print("Encryption disabled (plaintext baseline)")

    def encode(obs):
        return encryptor.encrypt(pack_state(obs))

    env.reset(seed=args.seed)
    eval_env.reset(seed=args.seed + 1)

    Q = {}  # ciphertext (or plaintext) bytes -> [value of STICK, value of HIT]
    best_eval = -float("inf")
    decay_episodes = max(1, int(args.episodes * args.eps_decay_frac))
    start = time.time()

    for ep in range(1, args.episodes + 1):
        epsilon = max(args.eps_end, args.eps_start
                      - (args.eps_start - args.eps_end) * ep / decay_episodes)
        # Learning rate shrinks over training so the table settles instead of
        # flipping borderline decisions on noisy rewards
        alpha = args.alpha + (args.alpha_end - args.alpha) * ep / args.episodes

        obs, _ = env.reset()
        key = encode(obs)
        done = False

        while not done:
            if rng.random() < epsilon:
                action = int(rng.integers(N_ACTIONS))
            else:
                action = best_action(lookup(Q, key), rng)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_key = encode(obs)

            # Q-learning update; the terminal state has value 0
            row = lookup(Q, key)
            next_val = 0.0 if terminated else lookup(Q, next_key).max()
            row[action] += alpha * (reward + args.gamma * next_val - row[action])
            key = next_key

        if ep % args.eval_every == 0:
            avg, win, draw, loss = evaluate(Q, encode, eval_env, args.eval_episodes, eval_rng)
            marker = ""
            if avg > best_eval:
                best_eval = avg
                with open(args.model, "wb") as f:
                    pickle.dump(Q, f)
                marker = "  <- saved"
            print(f"ep {ep:>8} | eps {epsilon:.3f} | states {len(Q):>3} | "
                  f"greedy eval: avg reward {avg:+.3f}  "
                  f"win {win:4.1f}%  draw {draw:4.1f}%  loss {loss:4.1f}%{marker}")

    with open(args.model.replace(".pkl", "_final.pkl"), "wb") as f:
        pickle.dump(Q, f)

    # Blackjack is very noisy, so the "best" checkpoint can just be a lucky reading.
    # Re-judge the best checkpoint and the final table on a big fresh sample and
    # keep the winner as the main model file.
    final_eval, best_ckpt_eval = None, None
    if os.path.exists(args.model):
        with open(args.model, "rb") as f:
            best_Q = pickle.load(f)
        big_env = make_env(args)
        big_env.reset(seed=args.seed + 2)
        big_rng = np.random.default_rng(args.seed + 3)
        best_ckpt_eval = evaluate(best_Q, encode, big_env, args.final_eval_episodes, big_rng)
        final_eval = evaluate(Q, encode, big_env, args.final_eval_episodes, big_rng)
        big_env.close()
        print(f"Final check on {args.final_eval_episodes} fresh games: "
              f"best checkpoint {best_ckpt_eval[0]:+.3f} | final table {final_eval[0]:+.3f}")
        if final_eval[0] >= best_ckpt_eval[0]:
            with open(args.model, "wb") as f:
                pickle.dump(Q, f)
            print("Saved the final table as the main model.")
        else:
            print("Kept the best checkpoint as the main model.")

    env.close()
    eval_env.close()
    print(f"Training finished in {time.time() - start:.0f}s | states visited: {len(Q)}")


# --------------------------------------------------------------------------
# Play (uses the same key as training)
# --------------------------------------------------------------------------
def load_table(args):
    with open(args.model, "rb") as f:
        Q = pickle.load(f)
    encryptor = build_encryptor(args, load_key=(args.encrypt == "ecb"))
    return Q, encryptor


def play(args):
    Q, encryptor = load_table(args)
    env = make_env(args, video_dir=args.video_dir)
    rng = np.random.default_rng(args.seed)
    env.reset(seed=args.seed)

    wins = draws = losses = 0
    total = 0.0
    for ep in range(1, args.play_episodes + 1):
        obs, _ = env.reset()
        done = False
        moves = []
        while not done:
            row = Q.get(encryptor.encrypt(pack_state(obs)))
            action = int(rng.integers(N_ACTIONS)) if row is None else best_action(row, rng)
            moves.append(f"{ACTION_NAMES[action]} at {obs[0]}")
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        total += reward
        wins += reward > 0
        draws += reward == 0
        losses += reward < 0
        result = "WIN" if reward > 0 else ("DRAW" if reward == 0 else "LOSS")
        if args.play_episodes <= 20 or ep % 1000 == 0:
            print(f"Game {ep}: {', '.join(moves)} -> final {obs[0]} vs dealer "
                  f"{sum(env.unwrapped.dealer)} : {result}")

    n = float(args.play_episodes)
    print(f"\n{args.play_episodes} games | avg reward {total / n:+.3f} | "
          f"win {100 * wins / n:.1f}%  draw {100 * draws / n:.1f}%  loss {100 * losses / n:.1f}%")
    env.close()  # finishes writing the video files
    if args.video_dir:
        print(f"Videos saved in: {args.video_dir}/")


def show_policy(args):
    """Print the learned strategy as a readable table. Keys are ciphertexts, so we
    encrypt every possible state with the saved key and look it up, exactly as the
    agent does."""
    Q, encryptor = load_table(args)
    rng = np.random.default_rng(0)
    for ace, title in ((0, "HARD hands (no usable ace)"), (1, "SOFT hands (usable ace)")):
        print(f"\n{title}   [H = hit, S = stick, . = never seen]")
        print("player \\ dealer shows: " + " ".join(f"{d:>2}" for d in range(1, 11)))
        for ps in range(21, 3, -1):
            cells = []
            for d in range(1, 11):
                row = Q.get(encryptor.encrypt(bytes([ps, d, ace])))
                cells.append(" ." if row is None else (" H" if best_action(row, rng) == 1 else " S"))
            if ace == 1 and ps < 12:
                continue
            print(f"        {ps:>2}           " + " ".join(cells))


# --------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "play", "policy"])
    p.add_argument("--encrypt", choices=["ecb", "none"], default="ecb")
    p.add_argument("--key-size", type=int, choices=[16, 24, 32], default=32,
                   help="AES key length in bytes")
    p.add_argument("--key-hex", type=str, default=None,
                   help="optional fixed key as hex; random if omitted")
    p.add_argument("--natural", action="store_true",
                   help="a natural blackjack (21 on the deal) pays extra")
    p.add_argument("--sab", action="store_true",
                   help="follow the exact Sutton & Barto rules")
    p.add_argument("--episodes", type=int, default=500_000)
    p.add_argument("--alpha", type=float, default=0.05, help="starting learning rate")
    p.add_argument("--alpha-end", type=float, default=0.002,
                   help="learning rate at the end of training (linear decay)")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-frac", type=float, default=0.7,
                   help="fraction of training over which epsilon decays")
    p.add_argument("--eval-every", type=int, default=25_000)
    p.add_argument("--eval-episodes", type=int, default=20_000,
                   help="greedy games per evaluation (Blackjack is very noisy)")
    p.add_argument("--final-eval-episodes", type=int, default=200_000,
                   help="games used for the final best-vs-final comparison")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", type=str, default="q_blackjack_ecb.pkl")
    p.add_argument("--play-episodes", type=int, default=10)
    p.add_argument("--video-dir", type=str, default=None,
                   help="folder to save .mp4 videos (play mode)")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    {"train": train, "play": play, "policy": show_policy}[args.mode](args)