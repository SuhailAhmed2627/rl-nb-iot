
import matplotlib.pyplot as plt
from tqdm import tqdm
from multi_env import MultiNBIoT
import  collections, random, numpy as np, torch

nn = torch.nn
F = torch.nn.functional
optim = torch.optim

SEED = 672

def set_all_seeds(seed):
  random.seed(seed)
  # os.environ('PYTHONHASHSEED') = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

class MultiEnv:
  def __init__(self, num_agents=2, vid=False):
    self.n_agents = num_agents
    self.env = MultiNBIoT()
    self.state_dim = 3
    self.action_dim = 9
  
  def reset(self):
    return self.env.reset()
  
  def step(self, a):
    obs_n, reward_n, done_n, info = self.env.step(a)
    return obs_n, reward_n, done_n, info

def record_ep(agents, env, maxT=100):
  obs_n = env.reset()
  for _ in range(maxT):
    if type(agents)==list:
      a = np.array([agent.act(state) for state, agent in zip(obs_n, agents)])
    else:
      a = agents.act(obs_n)
    obs_n, reward_n, done_n, info = env.step(a)
    if np.all(done_n): break
  env.close()

def rollout_centralised(env, agents, train=True, exploration=True, max_episode=10000, log_episode_interval=5, verbose=False):
  history_reward = []
  state_n = env.reset()
  episode_reward = 0
  episode_count = 0
  recorded_episodes = []
  recorded_episode_reward = []

  if train:
    act = lambda o_s, ex: agents.act(o_s, ex)
  else:
    act = lambda o_s, _: [a.act(o) for a, o in zip(agents, o_s)]

  while episode_count < max_episode:
    actions = act(state_n, exploration)
    next_state_n, reward_n, done_n, _ = env.step(actions)

    episode_reward += np.mean(reward_n)
    if train:
      agents.update(state_n, actions, reward_n, next_state_n, done_n)

    state_n = next_state_n
    if np.all(done_n):
      if not train:
        for a in agents: a.reset()
      state_n = env.reset()
      history_reward.append(episode_reward)
      episode_reward = 0
      episode_count += 1
      if episode_count % log_episode_interval == 0:
        recorded_episodes.append(episode_count)
        episodes_mean_reward = np.mean(history_reward)
        recorded_episode_reward.append(episodes_mean_reward)
        history_reward = []
        if verbose:
          print('Episodes {}, Reward {}'.format(episode_count, episodes_mean_reward))
  return recorded_episodes, recorded_episode_reward

def run_experiments(Envir, EnvArgs, Agent, Hyp_, 
                    n_episodes=1000, runs=10, log_interval=10, verbose=False):
  train_log = []
  train_reward_log = []
  test_reward_log = []

  pbar = tqdm(range(runs))
  for i in pbar:
    env = Envir(**EnvArgs)
    agents = Agent(*Hyp_)
    
    train_, train_reward = rollout_centralised(env=env, 
                                              agents=agents, 
                                              train=True,
                                              exploration=True, 
                                              max_episode=n_episodes,
                                              log_episode_interval=log_interval,
                                              verbose=verbose)
    print(train_reward[-1])

    train_log.append(train_)
    train_reward_log.append(train_reward)

    separated_agents = agents.get_agents()
    test_, test_reward = rollout_centralised(env=Envir(**EnvArgs), 
                                             agents=separated_agents, 
                                             train=False,
                                             exploration=False, 
                                             max_episode=10, 
                                             log_episode_interval=1,
                                             verbose=verbose)
    # store result for every run
    test_reward_log.append(np.mean(test_reward))

    pbar.set_description('r train: %.4f, r test: %.5f'%(train_reward[-1], test_reward_log[-1]))
  return train_log, train_reward_log, test_reward_log, agents, separated_agents

def plot_experiments(train_log, train_reward_log, test_reward_log, runs):
  fig = plt.figure(figsize=(9, 7))
  ax = fig.add_subplot(111)

  for i in range(runs):
    ax.plot(train_log[i], train_reward_log[i], label=f'run {i+1}')
  ax.set_title(f"Train learning Curve for {runs} runs")
  ax.set_ylabel("Episodic Reward")
  ax.set_xlabel("Iterations")
  ax.legend(loc="lower right")
  plt.tight_layout()
  plt.show()
  print(f'test reward (average over {runs} runs):', np.mean(test_reward_log))


class Agents:
  def __init__(self, n_actions, n_agents):
    self.act = lambda o: [np.random.randint(0,n_actions) for _ in range(n_agents)]

env = Switch(vid=True)
record_ep(Agents(env.action_dim, env.n_agents), env, maxT=100)

set_all_seeds(SEED)
print(torch.randn(4,3))
print(np.random.uniform(0, 1, (3, 2)))
print(random.randint(0, 1000))

class EpisodeBuffer:
  def __init__(self):
    self.transitions = []

  def put(self, transition):
    self.transitions.append(transition)
  
  def sample(self, lookup_step=None, idx=None):
    tr = self.transitions
    if idx is not None: tr = tr[idx:idx+lookup_step]
    return list(map(np.array, zip(*tr)))# [T, (o,a,r,n,d), A, I] --> [(o,a,r,n,d), T, A, I]
    
  def __len__(self):
    return len(self.transitions)

class EpisodeMemory():
  def __init__(self, n_agents, batch_sample=False, max_epi_num=100, max_epi_len=500,
                      batch_size=1, lookup_step=None):
    self.n_agents = n_agents
    self.max_epi_num = max_epi_num
    self.max_epi_len = max_epi_len
    self.batch_size = batch_size
    self.lookup_step = lookup_step
    self.memory = collections.deque(maxlen=self.max_epi_num)
    self.sample = self.batch_sample if batch_sample else self.sequential_sample

  def put(self, episode):
    self.memory.append(episode)
  
  def convert(self, sample, T, device):
    D = [[[ep[i][:,j] for ep in sample] for j in range(self.n_agents)] for i in range(5)]
    o, a, r, n, d = D
    
    cv = lambda ten,dty: torch.tensor(ten, dtype=dty).reshape(self.n_agents,len(sample),T,-1).to(device)
    o, r, n, d = map(lambda x:cv(x, torch.float), [o, r, n, d])
    a = cv(a, torch.long)
    
    return o, a, r, n, d

  def sequential_sample(self, device):
    idx = np.random.randint(0, len(self.memory))
    episode = self.memory[idx].sample()
    return self.convert([episode], len(episode), device)

  def batch_sample(self, device):
    sampled_episodes = random.sample(self.memory, self.batch_size)

    T_min = min(map(len, sampled_episodes))
    min_step = min(self.max_epi_len, T_min)
    T = self.lookup_step if min_step > self.lookup_step else min_step

    sample_buffer = []
    for episode in sampled_episodes:
      idx = np.random.randint(0, len(episode)-T+1)# random subset of episode
      sample = episode.sample(lookup_step=T, idx=idx)
      sample_buffer.append(sample)
            
    return self.convert(sample_buffer, T, device)

  def __len__(self):
    return len(self.memory)

class DDQN(nn.Module):
  def __init__(self, state_dim, action_dim, h = 32):
    super().__init__()
    self.device = 'cpu'
    if torch.cuda.is_available(): self.device = torch.cuda.current_device()

    self.h_dim = h
    self.act_dim = action_dim

    self.Head = nn.Sequential(nn.Linear(state_dim, h), nn.ReLU())
    self.LSTM = nn.LSTM(h, h, batch_first=True)

    self.V = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    self.A = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, action_dim))
    self.to(self.device)

  def forward(self, obs, hc):
    is_hc_int = type(hc)==int
    if is_hc_int: hc = self.init_hidden_state(hc, True)

    x = self.Head(obs.to(self.device))
    x, hc = self.LSTM(x, hc)
    v = self.V(x)
    a = self.A(x)
    a = a - a.mean(dim=-1, keepdim=True)
    q = v + a
    return q if is_hc_int else (q, hc)

  def sample_action(self, obs, hc, epsilon, exploration):
    output, hc = self(obs, hc)
    best = output.argmax(-1).detach().squeeze()
    if not exploration: return best.tolist(), hc
    A = obs.size(0)
    mask = (torch.rand(A) < epsilon).long()
    a_s = [random.randint(0,self.act_dim-1) if mask[i] else best[i].item() for i in range(A)]
    return a_s, hc

  def init_hidden_state(self, batch_size=1, training=False):
    if training is True:
      return torch.zeros([1, batch_size, self.h_dim]).to(self.device), torch.zeros([1, batch_size, self.h_dim]).to(self.device)
    else:
      return torch.zeros([1, 1, self.h_dim]).to(self.device), torch.zeros([1, 1, self.h_dim]).to(self.device)

class SingleAgent:
  def __init__(self, critic, role):
    self.role, self.critic = role, critic
    self.reset()
  
  def act(self, obs):
    o = torch.cat([self.role, torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(self.critic.device)], dim=-1)
    a, self.hcs = self.critic.sample_action(o, self.hcs, 0, False)
    return a
  
  def reset(self):
    self.hcs = self.critic.init_hidden_state(training=False)

class CentralisedAgents:
  def __init__(self, hyp, state_dim, action_dim, n_agents):
    self.hyp = hyp
    self.batch_size = hyp.batch_size
    self.epsilon = hyp.eps_start
    self.critic_in, self.action_dim = state_dim+n_agents, action_dim
    self.episode_memory = EpisodeMemory(n_agents, 
                                        batch_sample=hyp.batch_sample, 
                                        max_epi_num=hyp.max_epi_num, 
                                        max_epi_len=hyp.max_epi_len, 
                                        batch_size=hyp.batch_size, 
                                        lookup_step=hyp.lookup_step)
    self.agent_indexes = list(range(n_agents)) 
    self.n_agents = n_agents

    # Create Q functions
    self.critic = DDQN(self.critic_in, action_dim)
    self.critic_target = DDQN(self.critic_in, action_dim)
    self.critic_target.load_state_dict(self.critic.state_dict())

    self.optimiser = optim.Adam(self.critic.parameters(), lr=hyp.learning_rate)
    self.t = 0

    # store the batched roles
    self.roles = torch.eye(n_agents).reshape(n_agents, 1, 1, n_agents).to(self.critic.device)
    self.batched_roles = self.roles.repeat(1, self.batch_size, hyp.lookup_step, 1).to(self.critic.device)
    
    self.new_episode()
  
  def get_agents(self):
    agents = []
    for i in self.agent_indexes:
      c = DDQN(self.critic_in, self.action_dim)
      c.load_state_dict(self.critic.state_dict())
      agents.append(SingleAgent(c, self.roles[i]))
    return agents

  def act(self, obs, exploration=True):
    obs = torch.FloatTensor(obs).unsqueeze(1).to(self.critic.device)
    o = torch.cat((self.roles[:,0], obs), dim=-1)
    a, self.hcs = self.critic.sample_action(o, self.hcs, self.epsilon, exploration)
    return a

  def update(self, obs, a, r, obs_prime, done):
    self.t += 1
    done_mask = [float(not d) for d in done]

    self.episode_record.put([obs, a, r, obs_prime, done_mask])

    if len(self.episode_memory) >= self.hyp.min_epi_num and self.t%self.hyp.train_freq==0:
      self.train()
      if self.t % self.hyp.target_update_period == 0:# perform soft update
        for target_param, local_param in zip(self.critic_target.parameters(), self.critic.parameters()):
          target_param.data.copy_(self.hyp.tau*local_param.data + (1.0-self.hyp.tau)*target_param.data)
    
    if all(done):
      self.episode_memory.put(self.episode_record)
      self.new_episode()

  def new_episode(self):
    self.epsilon = max(self.hyp.eps_end, self.epsilon * self.hyp.eps_decay) # Linear annealing
    self.episode_record = EpisodeBuffer()
    self.hcs = self.critic.init_hidden_state(self.n_agents, True)

  def train(self):
    (o_s, a_s, r_s, n_o_s, dones) = self.episode_memory.sample(self.critic.device)
    A,B,T,I = o_s.size()
    
    n_ = torch.cat((self.batched_roles[:,:,:T], n_o_s), dim=-1).reshape(A*B,T,-1)
    with torch.no_grad():
        q_target = self.critic_target(n_, A*B).reshape(A, B, T, -1)

    q_target_max = q_target.max(-1)[0].view(A,B,T,-1).detach()
    targets = r_s + self.hyp.gamma * q_target_max * dones
    
    o_ = torch.cat((self.batched_roles[:,:,:T], o_s), dim=-1)
    q_ = self.critic(o_.reshape(A*B,T,-1), A*B).reshape(A, B, T, -1).contiguous()
    q = q_.gather(-1, a_s).sum(0)

    loss = F.smooth_l1_loss(q, targets.sum(0))
    
    self.optimiser.zero_grad()
    loss.backward()
    self.optimiser.step()

set_all_seeds(SEED)

class DQRN_Hyp():
  batch_size = 16
  gamma = 0.99
  learning_rate = 2e-3
  target_update_period = 20
  train_freq = 5
  eps_start = 0.3
  eps_end = 0.001
  eps_decay = 0.995
  tau = 1e-2
  batch_sample = True
  lookup_step = 40
  min_epi_num = 20
  max_epi_len = 100 
  max_epi_num = 100

n_agents = 9
env = MultiEnv(num_agents=n_agents)

hyp = DQRN_Hyp()
state_dim = env.state_dim
action_dim = env.action_dim

runs = 10
train_log, train_reward_log, test_reward_log, agents, separated_agents = run_experiments(MultiEnv, 
                                                               {'num_agents':n_agents},
                                                               CentralisedAgents, 
                                                               [hyp, state_dim, action_dim, n_agents], 
                                                               n_episodes=1000,
                                                               runs=runs,
                                                               verbose=False)