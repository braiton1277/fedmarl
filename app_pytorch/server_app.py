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

#começar com buffer cheio talvez 100 

# escalar recompensa
# r_scale = 100.0
# r_t = torch.tensor(float(r_scale * (U_curr - U_prev)), dtype=torch.float32)

#fazer 1000 rodadas com 256 min

#ver como fazer o evaluate

def cleanup_stats_dir(dir_path="clients"):
    p = Path(dir_path).resolve()
    shutil.rmtree(p, ignore_errors=True)  # remove tudo (inclusive subpastas)
    p.mkdir(parents=True, exist_ok=True)  # recria vazia
    print(f"[server] diretório limpo: {p}")

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    # num_rounds = context.run_config["num-server-rounds"]
    num_rounds = 4
    num_rounds_total = 100
    min_nodes = 2
    fraction_sample = context.run_config["fraction-sample"]
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    qnet = QNet(obs_dim=1, hidden=128).to(device)
    target_qnet = QNet(obs_dim=1, hidden=128).to(device)
    target_qnet.load_state_dict(qnet.state_dict())
    opt = torch.optim.Adam(qnet.parameters(), lr=3e-4)
    buffer = ReplayBuffer(capacity=50_000)
    


    # Inicializa o modelo global
    global_model = Net()
    global_model_key = "model"
    teste_global = load_global_testloader()        
    U_prev   = None
    pending_S, pending_A, pending_r = None, None, None
    log(INFO, "\n==== Conectado clientes ====")


    # Espera clientes suficientes
    all_node_ids: list[int] = []
    all_node_ids = list(grid.get_node_ids())
    while len(all_node_ids) < min_nodes:
            all_node_ids = list(grid.get_node_ids())

    node_ids_agents = all_node_ids[:]
    agents = {node_id: AgentMARL(node_id) for node_id in node_ids_agents}
    t =0
    log(INFO, "%d Clientes conectados com agente: %s",len(node_ids_agents), node_ids_agents)

    log(INFO, "\n==== Iniciando rodada de probing ====")
    while t < num_rounds_total:
        if t == 0:
            cleanup_stats_dir("clients")
        t+=1
        log(INFO, "Rodada de numero: %d",t)
        
        
        
        if len(all_node_ids) >= min_nodes:
            num_to_sample = int(len(all_node_ids) * 0.6)
            node_ids = random.sample(all_node_ids, num_to_sample)
        else:
            log(INFO, "Não há nós suficiente")
            break

        log(INFO, "Waiting for nodes to connect...")
        sleep(2)
        log(INFO, "%d conectados para probing: %s", len(node_ids), node_ids)
        
       
       
       

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
                with torch.no_grad():
                    s_probe = torch.tensor([[float(probing_loss_state)]],
                                        dtype=torch.float32, device=device)  # [1,1]
                    q = agents[node_id].q_value(s_probe,qnet)
                q0 = q[0].item()
                q1 = q[1].item()
                delta = q1 - q0
                q_msg = f"[Q] node_id={node_id}  Q(s,0)={q0:.4f}  Q(s,1)={q1:.4f}  Δ={delta:.4f}  Ploss={probing_loss_state:.4f}"
                print(q_msg)
                q_list.append((node_id, q0, q1, delta, probing_loss_state))
                probing_losses.append(msg.content["train_metrics"]["train_loss"])
            else:
                log(WARN, f"Erro na mensagem {msg.metadata.message_id}")

        # Log da probing loss média
        if probing_losses:
            log(INFO, f"Probing loss média: {sum(probing_losses)/len(probing_losses):.3f}")
        else:
            log(WARN, "Nenhuma probing loss recebida!")
        
        

        log(INFO, "\n==== Iniciando rodada de treino ====")
        log(INFO, "Treino da rodada %d",t)
        q_by_id = {
            nid: {"q0": q0, "q1": q1, "delta": delta, "loss": probing_loss_state}
            for (nid, q0, q1, delta, probing_loss_state) in q_list
            }
        slate_ids = [nid for nid in node_ids if nid in q_by_id]
        if not slate_ids:
            log(WARN, "Ninguém respondeu no probing; pulando rodada %d", t)
            continue
        
        k_select = max(1, int(len(node_ids)*0.6))
        eps = epsilon_by_round(t, num_rounds_total, start=1.0, end=0.05, scheme="exp", warmup=20)
       

        if (random.random() < eps) or (not q_by_id):
            train_node_id = random.sample(node_ids, k_select)
            reason = f"EXPLORAÇÃO (eps={eps:.3f})"
            

        else:
            #Sample nodes
            improved = [(nid, q_by_id[nid]["q1"], q_by_id[nid]["delta"])
                for nid in slate_ids if q_by_id[nid]["q1"] > q_by_id[nid]["q0"]]
            
            if not improved:
                train_node_id = random.sample(slate_ids, k_select)
                reason = "Fallback (sem q1>q0)"
            else:
                improved.sort(key=lambda x: (x[2], x[1]), reverse=True)
                train_node_id = [nid for (nid, _, _) in improved[:k_select]]

                reason = "EXPLOTAÇÃO (top-Δ)"
           
            


        non_train_node_id = [nid for nid in slate_ids if nid not in train_node_id]

        train_clients = [(nid, q_by_id[nid]["q1"], q_by_id[nid]["loss"]) for nid in train_node_id]
        non_train_clients = [(nid, q_by_id[nid]["q1"], q_by_id[nid]["loss"]) for nid in non_train_node_id]

        log(INFO, "%s → selecionados: %s", reason, train_node_id)
        log(INFO, "Treinam (%d): %s", len(train_clients), train_clients)
        log(INFO, "Não treinam (%d): %s", len(non_train_clients), non_train_clients)

        
        
                         
        for server_round in range(num_rounds):
            if not train_node_id:
                log(WARN, "train_node_id vazio na rodada %d — encerrando o loop.", t)
                break
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
                train_node_id, recorddict, MessageType.TRAIN, server_round
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
        global_model.to("cpu")

        _, eval_acc = test_fn(
        global_model,
        teste_global,
        device,
        )

        log(INFO, f"Acurácia global = {eval_acc:.4%}")

        s_t = torch.tensor([[q_by_id[nid]["loss"]] for nid in slate_ids], dtype=torch.float32)     # [N, 1]
        a_t = torch.tensor([1 if nid in train_node_id else 0 for nid in slate_ids], dtype=torch.long)  # [N]
        assert s_t.ndim == 2 and a_t.ndim == 1 and s_t.shape[0] == a_t.shape[0]
        U_curr = 20.0 / (1.0 + math.exp(0.35 * (1.0 - eval_acc))) - 10.0
        #10.0 - 20.0 / (1.0 + math.exp(0.35 * (1.0 - eval_acc)))

        if U_prev is not None:                     # a partir da 2ª rodada
            r_float = U_curr - U_prev
            r_t = torch.tensor(float(r_float), dtype=torch.float32)

            log(INFO, "Recompensa da rodada: ΔU = U_curr - U_prev = %.6f - %.6f = %.6f",
            U_curr, U_prev, r_float)
            if pending_S is not None:
                buffer.append((pending_S, pending_A, pending_r, s_t))         # ([N,d],[N],[],[N,d])
                # Atualiza pendência com a transição da rodada atual
                log(INFO, "Transição adicionada ao replay: S=%s A=%s r=[] S'=%s | total=%d",
                    tuple(pending_S.shape), tuple(pending_A.shape), tuple(s_t.shape), len(buffer))
                log(INFO, "S_t (N=%d): %s", s_t.shape[0], fmt_vec(s_t))
                log(INFO, "A_t (N=%d): %s", a_t.shape[0], fmt_vec(a_t))
           
            pending_S, pending_A, pending_r = s_t, a_t, r_t
            log(INFO, "Pendência atualizada: S=%s A=%s r=%.6f",
            tuple(pending_S.shape), tuple(pending_A.shape), float(pending_r.item()))
            
        else:
            # Primeira rodada: não há U_prev para delta -> só arma a pendência
            pending_S, pending_A = s_t, a_t
            pending_r = torch.tensor(0.0, dtype=torch.float32)  # ou pule a 1ª transição
            log(INFO, "Primeira rodada: pendência iniciada (sem S_{t+1} ainda).")
            
            
        U_prev = U_curr 
        


        #print(buffer.storage) 

        # batch_size = 64
        # learn_start = 256
        # updates_per_round = 8
        # gamma = 0.99
        # tau = 0.01
        # clip_grad = 10.0



        batch_size    = 32    
        gradient_steps = min(32, max(1, len(buffer) // (8*batch_size)))        
        gamma         = 0.99
        tau           = 0.01         
        clip_grad     = 10.0
        target_update_every  = None
        if len(buffer) >= 50:
            log(INFO, "\n==== Iniciando Treinamento DQN e agregação VDN ====")
            log(INFO, "VDN: replay=%d, batch=%d ⇒ gradient_steps=%d",
            len(buffer), batch_size, gradient_steps)
            

            for _ in range(gradient_steps):
                
                s, a, r, s_next = buffer.sample_uniform(batch_size=32, device=device)

                # print(">>> SHAPES / DTYPES")
                # print("s      :", s.shape,  s.dtype)      # [2, N, d]
                # print("a      :", a.shape,  a.dtype)      # [2, N], torch.int64
                # print("r      :", r.shape,  r.dtype)      # [2],   torch.float32
                # print("s_next :", s_next.shape, s_next.dtype)

                B, N, d = s.shape
                assert a.shape == (B, N),      f"a {a.shape} != {(B,N)}"
                assert r.shape == (B,),        f"r {r.shape} != {(B,)}"
                assert s_next.shape == (B,N,d),f"s_next {s_next.shape} != {(B,N,d)}"
                assert a.dtype == torch.int64, "Ações devem ser long (int64)"

            
                loss = AgentMARL.vdn_double_dqn_loss(qnet, target_qnet, s, a, r, s_next, gamma)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(qnet.parameters(), 10.0)
                opt.step()

                # Polyak update da target
                with torch.no_grad():
                    for p, pt in zip(qnet.parameters(), target_qnet.parameters()):
                        pt.data.mul_(1 - tau).add_(tau * p.data)




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



def epsilon_by_round(t, total_rounds, start=1.0, end=0.05, scheme="exp", warmup=3):
    # warm-up (100% aleatório nas primeiras rodadas)
    if t < warmup:
        return 1.0
    t_eff = max(0, t - warmup)

    if scheme == "exp":     # exponencial 
        decay = max(1, int(0.3 * total_rounds))  # meia-vida ~30% das rodadas
        return end + (start - end) * math.exp(-t_eff / decay)
    elif scheme == "linear":  # linear
        decay = max(1, int(0.6 * total_rounds))
        return max(end, start - (start - end) * (t_eff / decay))
    else:
        return max(end, start)  # fallback
    

def fmt_vec(t, k=15):
    """Converte tensor para lista (com truncagem opcional)."""
    v = t.detach().cpu()
    if v.ndim == 2 and v.shape[1] == 1:  # [N,1] -> [N]
        v = v.squeeze(1)
    lst = v.tolist()
    n = len(lst)
    if n <= k:
        return str(lst)
    return f"{lst[:k]} ... (+{n-k} itens)"