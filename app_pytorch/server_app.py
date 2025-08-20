import math
import random
import shutil
from copy import deepcopy
from logging import INFO, WARN
from pathlib import Path
from time import sleep

import torch
from flwr.common import (ArrayRecord, ConfigRecord, Context, Message,
                         MessageType, RecordDict)
from flwr.common.logger import log
from flwr.server import Grid, ServerApp

from app_pytorch.agent_marl import AgentMARL, QNet
from app_pytorch.replay_buffer import ReplayBuffer
from app_pytorch.task import Net, load_global_testloader
from app_pytorch.task import test as test_fn

# from reply_buffer import ReplayBuffer


def cleanup_stats_dir(dir_path="clients"):
    p = Path(dir_path).resolve()
    shutil.rmtree(p, ignore_errors=True)  # remove tudo (inclusive subpastas)
    p.mkdir(parents=True, exist_ok=True)  # recria vazia
    print(f"[server] diretório limpo: {p}")

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = context.run_config["num-server-rounds"]
    min_nodes = 2
    fraction_sample = context.run_config["fraction-sample"]
    qnet = QNet(obs_dim=1, hidden=128)
    target_qnet = QNet(obs_dim=1, hidden=128)
    buffer = ReplayBuffer(capacity=50_000)
    joint_states, joint_actions = [], []
    joint_next_states = []
    # Inicializa o modelo global
    global_model = Net()
    global_model_key = "model"
    teste_global = load_global_testloader()
    acc_prev = None          # ainda não existe recompensa
    U_prev   = None

    log(INFO, "\n==== Iniciando rodada única de probing ====")

    # Espera clientes suficientes
    all_node_ids: list[int] = []
    while len(all_node_ids) < min_nodes:
        all_node_ids = list(grid.get_node_ids())
        node_ids_probe = all_node_ids[:]   
        if len(all_node_ids) >= min_nodes:
            node_ids = node_ids_probe  # Usa todos os nós disponíveis
            break
        log(INFO, "Aguardando conexões de clientes...")
        sleep(2)

    agents = {node_id: AgentMARL(node_id) for node_id in node_ids}
    t =0

    while t < 4:
        if t == 0:
            cleanup_stats_dir("clients")
        t+=1
        log(INFO,t)
        log(INFO, "Clientes conectados para probing: %s", node_ids)

        # Cria mensagem com modelo global e configs
        gmodel_record = ArrayRecord(global_model.state_dict())
        recorddict = RecordDict(
            {
                global_model_key: gmodel_record,
                "train-config": ConfigRecord({"lr": 0.01, "mode": "probing"}),
            }
        )
        
        messages = construct_messages(t,
            node_ids, recorddict, MessageType.TRAIN, server_round=0
        )

        # Envia e recebe respostas
        replies = grid.send_and_receive(messages)
        log(INFO, "Recebido %s/%s resultados de probing", len(replies), len(messages))

        # Coleta probing losses
        probing_losses = []
        q_list = []
        for msg in replies:
            if msg.has_content():
                node_id = msg.metadata.src_node_id
                print(f"[Servidor] Mensagem recebida do node_id={node_id}")
                probing_loss_state = msg.content["train_metrics"]["train_loss"]
                transition = agents[node_id].state(probing_loss_state, qnet, t)
                q = agents[node_id].q_value(probing_loss_state,qnet)
                q0 = q[0].item()
                q1 = q[1].item()
                delta = q1 - q0
                q_msg = f"[Q] node_id={node_id}  Q(s,0)={q0:.4f}  Q(s,1)={q1:.4f}  Δ={delta:.4f}"
                print(q_msg)
                q_list.append((node_id, q0, q1, delta))
                if transition is not None:
                    s, a, r, s_next, t = transition
                    joint_states.append(s)
                    joint_actions.append(a)
                    joint_next_states.append(s_next)
                    print(transition)
                probing_losses.append(msg.content["train_metrics"]["train_loss"])
            else:
                log(WARN, f"Erro na mensagem {msg.metadata.message_id}")

        # Log da probing loss média
        if probing_losses:
            log(INFO, f"Probing loss média: {sum(probing_losses)/len(probing_losses):.3f}")
        else:
            log(WARN, "Nenhuma probing loss recebida!")
        
        

        log(INFO, "\n==== Iniciando rodada de treino ====")
        log(INFO,t)
        

        # Loop and wait until enough nodes are available.
        all_node_ids: list[int] = []
        while len(all_node_ids) < min_nodes:
            all_node_ids = list(grid.get_node_ids())
            if len(all_node_ids) >= min_nodes:
                # Sample nodes
                #num_to_sample = int(len(all_node_ids) * fraction_sample)
                k_select = int(len(all_node_ids) * fraction_sample)
                q_score_by_node = {}
                q_score_by_node = {nid: delta for (nid, q0, q1, delta) in q_list}
                eps = epsilon_by_round(t)
                do_explore = (random.random() < eps) or (len(q_score_by_node) < k_select)
                if do_explore:
                    node_ids_explore = random.sample(all_node_ids, k_select)
                    log(INFO, f"[Seleção] Exploração (ε={eps:.3f}) → {k_select} nós: {sorted(node_ids_explore)}")
                else:
                    # Exploitation: top-k pelos maiores scores em q_list
                    scored = [(nid, q_score_by_node.get(nid, float('-inf')))           # FIX: cria 'scored'
                            for nid in all_node_ids]
                    scored.sort(key=lambda x: x[1], reverse=True)

                    node_ids_explore = [nid for (nid, score) in scored[:k_select]]
                    top_str = ", ".join(f"{nid}:{score:.4f}" for nid, score in scored[:k_select])
                    log(INFO, f"[Seleção][EXPLOIT] ε={eps:.3f} → Top-{k_select} por ΔQ: [{top_str}]")
                    # lista final de selecionados 
                    log(INFO, f"[Seleção][EXPLOIT] Selecionados ({k_select}/{len(all_node_ids)}): {sorted(node_ids)}")
                #node_ids = random.sample(all_node_ids, num_to_sample)
                break
            log(INFO, "Waiting for nodes to connect...")
            sleep(2)

        log(INFO, "Sampled %s nodes (out of %s)", len(node_ids_explore), len(all_node_ids))
        for server_round in range(num_rounds):
            log(INFO, "")  
            log(INFO, "Starting round %s/%s", server_round + 1, num_rounds)

            # Create messages
            gmodel_record = ArrayRecord(global_model.state_dict())
            recorddict = RecordDict(
                {
                    global_model_key: gmodel_record,
                    "train-config": ConfigRecord({"lr": 0.01, "mode": "train"}),
                }
            )
            messages = construct_messages(t,
                node_ids_explore, recorddict, MessageType.TRAIN, server_round
            )

            # Send messages and wait for all results
            replies = grid.send_and_receive(messages)
            log(INFO, "Received %s/%s results", len(replies), len(messages))

            # Convert ArrayRecords in messages back to PyTorch's state_dicts
            state_dicts = []
            avg_train_losses = []
            for msg in replies:
                if msg.has_content():
                    state_dicts.append(msg.content[global_model_key].to_torch_state_dict())
                    avg_train_losses.append(msg.content["train_metrics"]["train_loss"])
                else:
                    log(WARN, f"message {msg.metadata.message_id} as an error.")

            # Compute average state dict
            avg_statedict = average_state_dicts(state_dicts)
            # Materialize global model
            global_model.load_state_dict(avg_statedict)

            # Log average train loss
            log(INFO, f"Avg train loss: {sum(avg_train_losses)/len(avg_train_losses):.3f}")




        log(INFO, "\n==== Iniciando rodada de evaluate global ====")

        # Local evaluation
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        global_model.to("cpu")

        _, eval_acc = test_fn(
        global_model,
        teste_global,
        device,
        )

        log(INFO, f"Acurácia global = {eval_acc:.4%}")

    
        U_curr = 10.0 - 20.0 / (1.0 + math.exp(0.35 * (1.0 - eval_acc)))

        if acc_prev is not None:                     # a partir da 2ª rodada
            r_t = U_curr - U_prev
        
            U_prev = U_curr

                # Pré-processa e garante tipos/shapes antes de salvar no replay
            s_t   = torch.stack(joint_states).detach().cpu().float()        # [N, ...]
            a_t = torch.stack(joint_actions).detach().cpu().squeeze(-1).long()        # [N]
            s_tp1 = torch.stack(joint_next_states).detach().cpu().float()   # [N, ...]
            r_t   = torch.tensor(float(r_t), dtype=torch.float32)           # []

            #
            if s_t.ndim   > 2: s_t   = s_t.flatten(1)
            if s_tp1.ndim > 2: s_tp1 = s_tp1.flatten(1)

            
            N, d = s_t.shape
            assert a_t.shape   == (N,),      f"a_t shape {a_t.shape} != {(N,)}"
            assert s_tp1.shape == (N, d),    f"s_tp1 shape {s_tp1.shape} != {(N, d)}"

           
            buffer.append((s_t, a_t, r_t, s_tp1))
            print(f"# amostras no replay: {len(buffer)}")
            
        U_prev = U_curr 
        acc_prev = eval_acc 

        joint_states.clear()
        joint_actions.clear()
        joint_next_states.clear()

        print(buffer.storage) 

        batch_size    = 32
        learn_start   = 128          # mínimo no buffer para começar a aprender
        gradient_steps = 16          # G updates por rodada (aumente depois p/ 32)
        gamma         = 0.99
        tau           = 0.01         
        clip_grad     = 10.0
        opt = torch.optim.Adam(qnet.parameters(), lr=3e-4)
        if len(buffer) >= 2:
            for _ in range(gradient_steps):
                # if len(buffer) < batch_size:
                #     break
                # 1) Amostrar minibatch [B,N,d],
                # uniforme:
                s, a, r, s_next = buffer.sample_uniform(batch_size=2, device=device)

                # print(">>> SHAPES / DTYPES")
                # print("s      :", s.shape,  s.dtype)      # [2, N, d]
                # print("a      :", a.shape,  a.dtype)      # [2, N], torch.int64
                # print("r      :", r.shape,  r.dtype)      # [2],   torch.float32
                # print("s_next :", s_next.shape, s_next.dtype)

                # checagens rápidas
                B, N, d = s.shape
                assert a.shape == (B, N)
                assert r.shape == (B,)
                assert s_next.shape == (B, N, d)

            #     # conteúdo das duas amostras
            #     print("\n>>> AMOSTRA 0")
            #     print("s[0]:\n", s[0])               # [N, d]
            #     print("a[0]:", a[0].tolist())        # [N]
            #     print("r[0]:", float(r[0]))          # escalar
            #     print("s_next[0]:\n", s_next[0])     # [N, d]

            #     print("\n>>> AMOSTRA 1")
            #     print("s[1]:\n", s[1])
            #     print("a[1]:", a[1].tolist())
            #     print("r[1]:", float(r[1]))
            #     print("s_next[1]:\n", s_next[1])
            # else:
            #     print(f"replay insuficiente: {len(buffer)} < {batch_size}")

                # OU híbrido (recomendado p/ novas transiçoes entrarem rapido):
                #s, a, r, s_next = buffer.sample_hybrid(batch_size=32, recent_k=512, m_recent=8, device=device)

                loss = AgentMARL.vdn_double_dqn_loss(qnet, target_qnet, s, a, r, s_next, gamma)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(qnet.parameters(), 10.0)
                opt.step()

                # Polyak update da target
                with torch.no_grad():
                    for p, pt in zip(qnet.parameters(), target_qnet.parameters()):
                        pt.data.mul_(1 - tau).add_(tau * p.data)


def epsilon_by_round(t, eps_start=0.2, eps_end=0.02, decay=200):
    return eps_end + (eps_start - eps_end) * math.exp(-t/decay)

def construct_messages(t,
    node_ids: list[int],
    record: RecordDict,
    message_type: MessageType,
    server_round: int,
) -> list[Message]:

    messages = []
    for node_id in node_ids:  # one message for each node

        rec = RecordDict(deepcopy(dict(record)))  # copia rala do payload base
        rec["node-info"] = ConfigRecord({
        "server_round": t,
        "node_id": node_id,
        # "partition_id": pid_map.get(node_id, None),  # se quiser também
        })


        message = Message(
            content=rec,
            message_type=message_type,  # target method in ClientApp
            dst_node_id=node_id,
            group_id=str(server_round),
        )
        messages.append(message)
    return messages


def average_state_dicts(state_dicts):
    """Return average state_dict."""
    # Initialize the averaged state dict
    avg_state_dict = {}

    # Iterate over keys in the first state dict
    for key in state_dicts[0]:
        # Stack all the tensors for this parameter across state dicts
        stacked_tensors = torch.stack([sd[key] for sd in state_dicts])
        # Compute the mean across the 0th dimension
        avg_state_dict[key] = torch.mean(stacked_tensors, dim=0)

    return avg_state_dict



