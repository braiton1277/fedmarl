import random
from collections import deque

import torch as T


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.storage = deque(maxlen=capacity)

    def append(self, transition):            # (s, a, r, s', t)
        self.storage.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.storage, batch_size)

        return map(list, zip(*batch))        # states, actions, ...

    def __len__(self):
        return len(self.storage)
    


    def _collate(self, batch, device=None):
            s_list, a_list, r_list, s2_list = [], [], [], []
            for item in batch:
                s, a, r, s2 = item[:4]  # ignora qualquer campo extra (ex.: t)

                # shapes/tipos
                if s.ndim  > 2: s  = s.flatten(1)    # [N,d]
                if s2.ndim > 2: s2 = s2.flatten(1)   # [N,d]
                a = a.view(-1).long()                # [N]
                if not isinstance(r, T.Tensor):
                    r = T.tensor(float(r))
                r = r.to(dtype=T.float32).view(())   # escalar float32

                s_list.append(s); a_list.append(a); r_list.append(r); s2_list.append(s2)

            s  = T.stack(s_list)    # [B,N,d]
            a  = T.stack(a_list)    # [B,N]
            r  = T.stack(r_list)    # [B]
            s2 = T.stack(s2_list)   # [B,N,d]

            if device is not None:
                s, a, r, s2 = s.to(device), a.to(device), r.to(device), s2.to(device)
            return s, a, r, s2

        # ---------- NOVO: amostragem uniforme ----------
    def sample_uniform(self, batch_size: int, device=None):
        assert len(self.storage) >= batch_size, "replay insuficiente"
        batch = random.sample(self.storage, batch_size)
        return self._collate(batch, device)

        # ---------- NOVO: amostragem híbrida (recentes + uniforme) ----------
    def sample_hybrid(self, batch_size: int, recent_k=512, m_recent=8, device=None):
        N = len(self.storage); assert N >= batch_size, "replay insuficiente"
        R = min(recent_k, N)                   # janela de recência
        m = min(m_recent, batch_size, R)       # quantas recentes vão no batch

        recent = list(self.storage)[-R:]
        fresh  = random.sample(recent, m) if m > 0 else []

        pool = list(self.storage)[:-R] if N > R else list(self.storage)
        need = batch_size - len(fresh)
        old  = random.sample(pool if len(pool) >= need else list(self.storage), need)

        batch = fresh + old
        return self._collate(batch, device)

        # ---------- NOVO: sem reposição dentro de um "burst" de updates ----------
    def sample_indices_no_replacement(self, total_needed: int):
        N = len(self.storage)
        k = min(N, total_needed)
        return T.randperm(N)[:k].tolist()

    def collate_by_indices(self, indices, device=None):
        batch = [self.storage[i] for i in indices]
        return self._collate(batch, device)