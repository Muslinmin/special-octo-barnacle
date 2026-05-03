# 🤖 AI & ML Weekly Assignments — SIT

> Weekly reinforcement learning and machine learning mini-projects completed as part of coursework at the Singapore Institute of Technology (SIT).  
> All implementations are in Python using Jupyter Notebooks, with libraries including PyTorch, OpenAI Gym / Gymnasium, and Stable-Baselines3.

---

## 📁 Assignment Overview

### 1. Multi-Armed Bandit (Exploration vs. Exploitation)
**Files:** `bandit.ipynb`, `2201360_bandit_task1.ipynb`, `2201360_bandit_task2.ipynb`

**Objective:**  
Implement and analyse the epsilon-greedy strategy on a Multi-Armed Bandit (MAB) problem. The goal is to balance exploration (trying new arms) against exploitation (choosing the currently best-known arm) to maximise cumulative reward over time.

**Lessons Learned:**
- Epsilon decay rate has a significant impact on long-term performance — too fast (e.g. 0.99) causes premature convergence to suboptimal arms; slower decay (e.g. 0.999) allows more thorough exploration.
- There is a sweet spot for decay: slower is not always better, as the agent must eventually commit to exploiting what it has learned.
- Cumulative regret is a useful metric for evaluating bandit strategies over time.

---

### 2. Temporal Difference Learning — Q-Learning & SARSA
**Files:** `Task1_QLearning.ipynb`, `Task2_SARSA.ipynb`

**Objective:**  
Apply tabular Q-Learning (off-policy) and SARSA (on-policy) to a grid-based environment (FrozenLake-v1). Compare how each method converges to an optimal policy and how their learning behaviour differs.

**Lessons Learned:**
- Q-Learning tends to find the optimal policy faster due to its off-policy nature (it always bootstraps from the greedy action), while SARSA is more conservative as it updates based on the actual next action taken.
- Terminal state handling is critical: when `terminated=True`, the TD target must be set to the immediate reward only — no future value should be added.
- Epsilon decay tuning is environment-dependent; FrozenLake required a decay of 0.999 to avoid premature exploitation.

---

### 3. Monte Carlo Methods on GridWorld
**Files:** `GridWorld_MC.ipynb`, `GridWorld_MC_Task2.ipynb`

**Objective:**  
Implement Monte Carlo First-Visit prediction and control on a custom GridWorld environment. Evaluate state-value functions and derive optimal policies using sampled episode returns rather than step-by-step bootstrapping.

**Lessons Learned:**
- Monte Carlo methods require complete episodes before updates can be made, making them slower to converge compared to TD methods in some environments.
- First-Visit MC only averages returns from the first time a state is visited in each episode, which reduces bias in value estimation.
- Policy evaluation and policy improvement can be interleaved (Generalised Policy Iteration) to iteratively refine the agent's behaviour.

---

### 4. Policy Gradient — REINFORCE (CartPole)
**Files:** `CartPole_REINFORCE_PyTorch.ipynb`

**Objective:**  
Implement the REINFORCE algorithm (Monte Carlo Policy Gradient) using PyTorch to solve the CartPole-v1 balancing task. Train a neural network policy directly using sampled episode returns.

**Lessons Learned:**
- REINFORCE is high-variance because the gradient estimate is based on full episode returns, which can fluctuate significantly episode to episode.
- Using a baseline (e.g. mean return subtraction) helps reduce variance and stabilise training.
- The algorithm is sensitive to learning rate — too high causes unstable updates, too low leads to very slow convergence.
- CartPole-v1 is considered solved when the agent achieves an average reward of 475+ over 100 consecutive episodes.

---

### 5. Deep Q-Network (DQN) — CartPole
**Files:** `CartPole_DQN_Train.ipynb`

**Objective:**  
Implement a Double DQN with experience replay to solve CartPole-v1. Use a replay buffer and a target network to stabilise training, and compare against the vanilla REINFORCE approach.

**Lessons Learned:**
- Experience replay breaks temporal correlation between consecutive transitions, which is critical for stable neural network training.
- A separate target network (updated periodically rather than every step) prevents the Q-value estimates from chasing a moving target.
- Batching predictions during replay (i.e. running the entire batch through the network in one forward pass) dramatically reduces training time without affecting learning quality.
- Double DQN reduces overestimation bias compared to vanilla DQN by decoupling action selection from value evaluation.

---

### 6. Actor-Critic Methods — A2C & SAC on FetchReach
**Files:** `Fetchreach_a2c.ipynb`, `FetchReach_SAC_2201360_.ipynb`

**Objective:**  
Train a simulated robotic arm (FetchReach-v4) to reach a target position in 3D space using two actor-critic algorithms: Advantage Actor-Critic (A2C, on-policy) and Soft Actor-Critic (SAC, off-policy). Compare their learning stability and sample efficiency.

**Lessons Learned:**
- SAC (off-policy) is significantly more sample-efficient than A2C because it reuses past experience via a replay buffer.
- A2C is more unstable during training due to its on-policy nature — each update is derived from freshly sampled data that is then discarded.
- SAC's entropy regularisation term encourages exploration by keeping the policy stochastic, which is particularly beneficial in continuous action spaces like robotic control.
- Proper environment wrappers are essential: `FetchReach-v4` (Gymnasium) requires a `FlattenObservation` wrapper to convert the dict observation space into a flat vector for the networks.

---

### 7. Proximal Policy Optimisation (PPO)
**Files:** `2201360_PPO_Assignment.ipynb`, `TrainingVerificationScript_for_exercise5.ipynb`

**Objective:**  
Implement and evaluate PPO — a popular on-policy algorithm known for its stability improvements over vanilla policy gradients — on a continuous control task. Compare PPO's performance against A2C and SAC in terms of convergence speed and final reward.

**Lessons Learned:**
- PPO uses a clipped surrogate objective to prevent destructively large policy updates, making it substantially more stable than REINFORCE or A2C.
- Despite its improvements, PPO remains on-policy and is therefore less sample-efficient than SAC, which reuses experience from a replay buffer.
- Across experiments, the general ranking of sample efficiency was: **SAC > PPO > A2C**, consistent with the theoretical trade-offs between off-policy and on-policy methods.
- Hyperparameter choices (clip ratio, number of epochs per update, batch size) have a meaningful effect on PPO's stability and convergence speed.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| PyTorch | Neural network implementation (REINFORCE, DQN) |
| OpenAI Gym / Gymnasium | RL environments (CartPole, FrozenLake, FetchReach) |
| Stable-Baselines3 | Pre-built A2C, SAC, PPO implementations |
| NumPy / Matplotlib | Numerical computation and result visualisation |
| Google Colab | Primary development and training environment |

---

## 📈 Algorithm Comparison Summary

| Algorithm | Type | Environment | Sample Efficiency | Stability |
|-----------|------|-------------|-------------------|-----------|
| Epsilon-Greedy MAB | Bandit | Custom MAB | — | Depends on ε decay |
| Q-Learning | Off-policy TD | FrozenLake | Medium | High |
| SARSA | On-policy TD | FrozenLake | Medium | High |
| Monte Carlo | MC Control | GridWorld | Low | Medium |
| REINFORCE | Policy Gradient | CartPole | Low | Low (high variance) |
| DQN | Off-policy Deep RL | CartPole | High | Medium |
| A2C | On-policy Actor-Critic | FetchReach | Low | Low |
| SAC | Off-policy Actor-Critic | FetchReach | **Highest** | High |
| PPO | On-policy Actor-Critic | FetchReach | Medium | High |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
