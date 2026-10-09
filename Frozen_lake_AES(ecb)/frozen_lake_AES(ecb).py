import os
from collections import defaultdict

import gymnasium as gym
import gymnasium.envs.toy_text.frozen_lake as fl
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# ---------------- AES-ECB (environment side) ----------------
KEY = os.urandom(32)  # AES-256, the agent never sees this


def encrypt_state(state):
    """State int -> 1 byte -> PKCS#7 pad to 16 bytes -> AES-ECB."""
    padder = padding.PKCS7(128).padder()
    padded = padder.update(bytes([state])) + padder.finalize()
    enc = Cipher(algorithms.AES(KEY), modes.ECB()).encryptor()
    return enc.update(padded) + enc.finalize()


# ---------------- Training ----------------
env = gym.make("FrozenLake-v1", is_slippery=False)

# Keys are ciphertext bytes instead of state numbers
q_table = defaultdict(lambda: np.zeros(env.action_space.n))

learning_rate = 0.9
discount_factor = 0.989
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999
episodes = 5000

success = []

print("Training started...\n")

for episode in range(episodes):

    state, info = env.reset()
    obs = encrypt_state(state)          # agent sees only this
    done = False

    while not done:

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q_table[obs]))

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_obs = encrypt_state(next_state)

        q_table[obs][action] += learning_rate * (
            reward
            + discount_factor * np.max(q_table[next_obs])
            - q_table[obs][action]
        )

        obs = next_obs

    success.append(reward)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (episode + 1) % 500 == 0:
        rate = np.mean(success[-500:]) * 100
        print(
            f"Episode {episode + 1}/{episodes} | "
            f"Success Rate: {rate:.2f}% | "
            f"Epsilon: {epsilon:.4f}"
        )
env.close()
print("\nTraining complete!")

print(f"\nDistinct ciphertexts seen by agent: {len(q_table)}")
print("\nFinal Q-Table (ciphertext hex -> Q-values):")
for cipher, values in q_table.items():
    print(cipher.hex(), np.round(values, 3))

rates = [
    np.mean(success[i - 500:i]) * 100
    for i in range(500, episodes + 1, 500)
]

plt.plot(range(500, episodes + 1, 500), rates, marker="o")
plt.xlabel("Episodes")
plt.ylabel("Success Rate (%)")
plt.title("Q-Learning on FrozenLake (AES-ECB encrypted states)")
plt.grid()
plt.savefig("learning_curve_ecb.png")
plt.close()

print("\nLearning graph saved as: learning_curve_ecb.png")

# ---------------- Video ----------------
print("\nPreparing video recording...")

CELL = 64
IMG_DIR = os.path.join(os.path.dirname(fl.__file__), "img")


def load_sprite(name):
    path = os.path.join(IMG_DIR, name)
    return Image.open(path).convert("RGBA").resize((CELL, CELL))


ice = load_sprite("ice.png")
hole = load_sprite("hole.png")
cracked_hole = load_sprite("cracked_hole.png")
goal = load_sprite("goal.png")
stool = load_sprite("stool.png")
elf = {
    0: load_sprite("elf_left.png"),
    1: load_sprite("elf_down.png"),
    2: load_sprite("elf_right.png"),
    3: load_sprite("elf_up.png"),
}

video_env = gym.make("FrozenLake-v1", is_slippery=False)
desc = video_env.unwrapped.desc
nrow, ncol = desc.shape


def render_frame(state, last_action=None):
    canvas = Image.new("RGBA", (ncol * CELL, nrow * CELL))

    for r in range(nrow):
        for c in range(ncol):
            pos = (c * CELL, r * CELL)
            canvas.alpha_composite(ice, pos)

            tile = desc[r, c]
            if tile == b"H":
                is_agent_here = (r * ncol + c) == state
                canvas.alpha_composite(cracked_hole if is_agent_here else hole, pos)
            elif tile == b"G":
                canvas.alpha_composite(goal, pos)
            elif tile == b"S":
                canvas.alpha_composite(stool, pos)
    r, c = divmod(state, ncol)
    direction = last_action if last_action is not None else 1
    canvas.alpha_composite(elf[direction], (c * CELL, r * CELL))

    return np.array(canvas.convert("RGB"))


frames = []

state, info = video_env.reset()
frames.append(render_frame(state))

done = False
steps = 0
max_steps = 100

while not done and steps < max_steps:

    # Agent decides from the ciphertext; rendering uses the true state
    action = int(np.argmax(q_table[encrypt_state(state)]))

    state, reward, terminated, truncated, info = video_env.step(action)
    done = terminated or truncated
    steps += 1

    frames.append(render_frame(state, action))

video_env.close()
frames.extend([frames[-1]] * 3)

imageio.mimsave("frozenlake_agent.mp4", frames, fps=2)

print("Video saved as: frozenlake_agent.mp4")