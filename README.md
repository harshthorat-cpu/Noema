# Noema

**What if your RL agent never got to see the state, and still learned to solve the task?**

That is exactly what this project does. The environment encrypts every observation with AES before the agent ever touches it. The agent gets a pile of random-looking bytes, no idea what they mean, and it still figures out how to win.

I find this  a lot of fun, so I built it.

---

## Watch it learn

<table>
  <tr>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/fa1e65fb-52dc-4b05-8674-023066e636e3" controls muted width="100%"></video>
    </td>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/30476de3-f6ab-4958-820f-4307a18df057" controls muted width="100%"></video>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/172d96b6-dc63-4e3f-b456-9c9726b5896c" controls muted width="100%"></video>
    </td>
    <td width="50%">
      <!-- Paste your Space Invaders video link in the src below -->
      <video src="PASTE_SPACE_INVADERS_VIDEO_LINK_HERE" controls muted width="100%"></video>
    </td>
  </tr>
</table>

The first three show the agent learning while it only sees encrypted bytes. The last one is Space Invaders, where it did not work (more on that below).

---

## The idea

In normal RL the loop is simple. The environment shows the agent a state, the agent picks an action, and the environment hands back a reward.

Here there is one extra step in the middle:

<img width="566" height="114" alt="Pipeline: environment, preprocess, AES encrypt, agent" src="https://github.com/user-attachments/assets/54bbd1a5-3dcb-4ac4-bba7-c8bc5add1d58" />

So the agent is not looking at a grid, a pixel, or a coordinate. It is looking at ciphertext. If you printed one of its observations you would see noise.

## Why does this even work?

The trick is the cipher mode. AES in **ECB mode is deterministic**: the same input always produces the same output. That means:

- The same state always gives the same ciphertext.
- Different states give different ciphertexts.
- A neural network does not care that the bytes look random. It only needs them to be *consistent*.

So the agent never learns what a state means. It just learns that "this exact blob of bytes is a good place to go left." For small, discrete environments that is plenty.

The replay buffer stores the encrypted states too, so nothing in the training loop ever sees the plaintext.


## Run it yourself

```bash
git clone -b developer https://github.com/harshthorat-cpu/Noema.git
cd Noema
python <python file to train> train
```

## The honest part

**Space Invaders did not work.** I tried the same setup on it and the agent never learned to play. My guess is that the state space is far bigger than the small grids, so the agent almost never sees the exact same encrypted frame twice, and the "consistent bytes" trick stops helping. I have not confirmed that, but it fits the idea.

Also, ECB is a **bad** cipher for real security. Identical plaintext blocks give identical ciphertext blocks, which leaks structure. That leak is the very reason the agent can learn here. So read this project as an experiment about learnability, not as a privacy tool. Only the state is hidden. The reward and the actions are still visible.

If you swap ECB for a non-deterministic mode like CBC, the same state stops mapping to the same bytes, and learning gets much harder.

---

## Where this comes from

Noema is my implementation of **Reinforcement Learning on Encrypted Data** by Alberto Jesu, Victor-Alexandru Darvariu, Alessandro Staffolani, Rebecca Montanari and Mirco Musolesi.

Paper: [arxiv.org/abs/2109.08236](https://arxiv.org/abs/2109.08236)

All credit for the idea and the experimental design goes to the authors.

