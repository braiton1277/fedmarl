import collections
import copy
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentMARL:
    def __init__(self, node_id):
        self.node_id = node_id
        self.last_state  = None
        self.last_action = None
        print(f"[Agente] Inicializado agente para o cliente com node_id={node_id}")
      
    def state(self, current_state, qnet, round):
        
        
        state_tensor = torch.tensor([[current_state]], dtype=torch.float32)
        if self.last_state is not None and self.last_action is not None:
            reward = 2 # ← define como quiser
            
        
        with torch.no_grad():
            q_values = qnet(state_tensor)
            action = torch.argmax(q_values).item()

        transition = None

        action = torch.tensor([action], dtype=torch.int64)

        if self.last_state is not None and self.last_action is not None:
            reward = 2
            transition = (
                self.last_state,     # state
                self.last_action,    # action
                reward,              # reward
                state_tensor,       # next_state
                round                 # done flag
            )



        self.last_state = state_tensor
        self.last_action = action

        
        print(f"Q-values do agente: {q_values.tolist()}")
        return transition
    

    def q_value(self, current_state, qnet):
        device = next(qnet.parameters()).device
        s = torch.as_tensor([[current_state]], dtype=torch.float32, device=device)
        was = qnet.training
        qnet.eval()
        with torch.no_grad():
            q = qnet(s).squeeze(0)  # [Q(s,0), Q(s,1)]
        if was: qnet.train()
        

        return q

    # def collect_states(self, client_manager, rnd):
    #     cids = client_manager.available_clients
    #     states = torch.tensor([self.state(cid, rnd) for cid in cids], dtype=torch.float32)
    #     return cids, states


    def vdn_double_dqn_loss(
        q_online,
        q_target,
        s,          # [B, N, d]
        a,          # [B, N] (long)
        r,          # [B]     (float)
        s_next,     # [B, N, d]
        gamma: float,
        mask: torch.Tensor | None = None,  
        use_huber: bool = True,
    ):
        """
        Calcula a loss do VDN com Double DQN (sem 'done' — tarefa contínua).
        - q_online(s)  -> [B, N, A]
        - q_target(s') -> [B, N, A]
        """

        B, N, d = s.shape
        A = 2  # nº de ações

        # # Q(s, a_tomada) com a rede online
        q_all_flat = q_online(s.reshape(B * N, d))          # [B*N, 2]
        q_all = q_all_flat.reshape(B, N, A)                  # [B, N, 2]

        q_taken = q_all.gather(-1, a.unsqueeze(-1)).squeeze(-1)  # [B, N]


        # # VDN: soma sobre agentes
        q_tot = q_taken.sum(dim=1)
        
        # # Alvo Double DQN
        with torch.no_grad():
            q_next_flat_online = q_online(s_next.reshape(B * N, d))   # [B*N, 2]
            a_star = q_next_flat_online.argmax(dim=-1).reshape(B, N)  # [B, N]

            q_next_flat_tgt = q_target(s_next.reshape(B * N, d))      # [B*N, 2]
            q_tgt_taken = q_next_flat_tgt.gather(
                -1, a_star.reshape(-1, 1)
            ).squeeze(-1).reshape(B, N)                                # [B, N]

            q_tot_next = q_tgt_taken.sum(dim=1)                        # [B]
            y = r + gamma * q_tot_next    


        # ---- loss ----
        loss = F.smooth_l1_loss(q_tot, y) if use_huber else F.mse_loss(q_tot, y)


        # b = 0
        # with torch.no_grad():
        #     print("\n[DEBUG VDN]")
        #     print("Q_online(s)[b,:,:]  (dois Q por agente):")
        #     print(q_all[b].cpu())              # [N, 2]

            
        #     print("a* (argmax online em s_next) [b,:]:")
        #     print(a_star[b].cpu())             # [N]

            
        #     q_next_tgt = q_next_flat_tgt.reshape(B, N, A)
        #     print("Q_target(s_next)[b,:,:] (dois Q por agente):")
        #     print(q_next_tgt[b].cpu())         # [N, 2]

        #     print(f"q_tot[b]:     {float(q_tot[b]):.6f}")
        #     print(f"q_tot_next[b]:{float(q_tot_next[b]):.6f}")
        #     print(f"y[b]:         {float(y[b]):.6f}")
        #     print(f"loss:         {float(loss):.6f}")

        return loss


class QNet(nn.Module):
    def __init__(self, observation_space, action_space, recurrent=False):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        self.recurrent = recurrent
        self.hx_size = 32
        n_obs = observation_space[0].shape[0]
        #self.id_emb = nn.Embedding(num_agents, 8) 
        self.feature = nn.Sequential(
            nn.Linear(n_obs, 64), nn.ReLU(),
            nn.Linear(64, self.hx_size), nn.ReLU()
        )
        
        if recurrent:
            self.gru = nn.GRUCell(self.hx_size, self.hx_size)
        self.q_head = nn.Linear(self.hx_size, action_space[0].n)
        

        #ver oq tem no action space e no obsdim do original
        #ver como eh o vetor actions


        
        # for agent_i in range(self.num_agents):
        #     n_obs = observation_space[agent_i].shape[0]
        #     setattr(self, 'agent_feature_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, 64),
        #                                                                     nn.ReLU(),
        #                                                                     nn.Linear(64, self.hx_size),
        #                                                                     nn.ReLU()))
        #     if recurrent:
        #         setattr(self, 'agent_gru_{}'.format(agent_i), nn.GRUCell(self.hx_size, self.hx_size))
        #     setattr(self, 'agent_q_{}'.format(agent_i), nn.Linear(self.hx_size, action_space[agent_i].n))

    def forward(self, obs, hidden):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        next_hidden = [torch.empty(obs.shape[0], 1, self.hx_size)] * self.num_agents
        for agent_i in range(self.num_agents):
            # ids = torch.full((obs.shape[0], 1), agent_i, device=obs.device, dtype=torch.long)
            # id_vec = self.id_emb(ids)                          # embedding do agente
            # x = torch.cat([obs[:, agent_i, :], id_vec], dim=-1)
            #x = self.feature(x)
            x = self.feature(obs[:, agent_i, :])                                # mesma rede para todos
            if self.recurrent:
                x = self.gru(x, hidden[:, agent_i, :])
                next_hidden[agent_i] = x.unsqueeze(1)
            q_values[agent_i] = self.q_head(x).unsqueeze(1)
            
            
            
            
            
            # x = getattr(self, 'agent_feature_{}'.format(agent_i))(obs[:, agent_i, :])
            # if self.recurrent:
            #     x = getattr(self, 'agent_gru_{}'.format(agent_i))(x, hidden[:, agent_i, :])
            #     next_hidden[agent_i] = x.unsqueeze(1)
            # q_values[agent_i] = getattr(self, 'agent_q_{}'.format(agent_i))(x).unsqueeze(1)

        return torch.cat(q_values, dim=1), torch.cat(next_hidden, dim=1)

    def sample_action(self, obs, hidden, epsilon):
        out, hidden = self.forward(obs, hidden)
        mask = (torch.rand((out.shape[0],)) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1],))
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).float()
        action[~mask] = out[~mask].argmax(dim=2).float()
        return action, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.hx_size))
    




def treino(q, q_target, memory, optimizer, gamma, batch_size, update_iter=10, chunk_size=10, grad_clip_norm=5):
     _chunk_size = chunk_size if q.recurrent else 1 
     for _ in range(update_iter):
         s, a, r, s_prime, done = memory.sample_chunk(batch_size, _chunk_size) 
         hidden = q.init_hidden(batch_size) 
         target_hidden = q_target.init_hidden(batch_size) 
         loss = 0 
         for step_i in range(_chunk_size): 
            q_out, hidden = q(s[:, step_i, :, :], hidden) 
            q_a = q_out.gather(2, a[:, step_i, :].unsqueeze(-1).long()).squeeze(-1) 
            sum_q = q_a.sum(dim=1, keepdims=True) 
            max_q_prime, target_hidden = q_target(s_prime[:, step_i, :, :], target_hidden.detach()) 
            max_q_prime = max_q_prime.max(dim=2)[0].squeeze(-1) 
            target_q = r[:, step_i, :].sum(dim=1, keepdims=True) 
            target_q += gamma * max_q_prime.sum(dim=1, keepdims=True) * (1 - done[:, step_i]) 
            
            loss += F.smooth_l1_loss(sum_q, target_q.detach()) 
            
            done_mask = done[:, step_i].squeeze(-1).bool() 
            hidden[done_mask] = q.init_hidden(len(hidden[done_mask])) 
            target_hidden[done_mask] = q_target.init_hidden(len(target_hidden[done_mask])) 
            
            
            optimizer.zero_grad() 
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(q.parameters(), grad_clip_norm, norm_type=2) 
            optimizer.step() 
            
            
            
            
def test(env, num_episodes, q): 
    score = 0 
    for episode_i in range(num_episodes): 
        state = env.reset() 
        done = [False for _ in range(env.n_agents)] 
        with torch.no_grad(): 
            hidden = q.init_hidden() 
            while not all(done): 
                action, hidden = q.sample_action(torch.Tensor(state).unsqueeze(0), hidden, epsilon=0) 
                next_state, reward, done, info = env.step(action[0].data.cpu().numpy().tolist()) 
                score += sum(reward) 
                state = next_state 
        
        
    return score / num_episodes




