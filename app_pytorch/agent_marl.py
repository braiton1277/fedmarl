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
        mask: torch.Tensor | None = None,   # [B, N] bool/0-1 (opcional)
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
    """Rede MLP para calcular Q-valores de duas ações."""
    def __init__(self, obs_dim: int, hidden: int = 128):
        """
        Args
        ----
        obs_dim : dimensão do vetor-estado s_i
        hidden  : largura das camadas ocultas  (128)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, 2)              # 2 ações: a=0 (não seleciona) e a=1 (seleciona)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor (batch, obs_dim)   →   retorna (batch, 2)
        """
        return self.net(x)
    



