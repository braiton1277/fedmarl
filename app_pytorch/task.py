import json
import os
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset
_global_test_ds = None


CLASSES_CIFAR10 = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]

def load_global_testloader(
    partition_id: int = 0,
    num_partitions: int = 10,
    *,
    alpha: float = 0.2,
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
    debug: bool = True,
):
    global _global_test_ds

    if _global_test_ds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=alpha,
            seed=seed,
        )
        _fds_test = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"test": partitioner},
        )

    part = _fds_test.load_partition(partition_id)
    base = part["test"] if (hasattr(part, "keys") and "test" in part) else part

    tfm = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    def apply_transforms(batch):
        batch["img"] = [tfm(img) for img in batch["img"]]
        return batch
    base.set_transform(apply_transforms)

    bs = min(batch_size, max(1, len(base)))
    testloader = DataLoader(base, batch_size=bs, shuffle=False, drop_last=False, num_workers=num_workers)

    if debug:
        n = len(base)
        print(f"[global-test] parts={num_partitions} alpha={alpha} pid={partition_id} n_part={n} (bs={bs})")

        # === Contagem de classes iterando pelo DataLoader (sem mexer na transform) ===
        try:
            dist = Counter()
            seen = 0
            for batch in testloader:
                ys = batch["label"]
                # HF DataLoader costuma entregar lista de ints
                if not isinstance(ys, list):
                    ys = list(ys)
                dist.update(int(y) for y in ys)
                seen += len(ys)
            for c in sorted(dist):
                frac = 100.0 * dist[c] / max(1, seen)
                name = CLASSES_CIFAR10[c] if 0 <= c < len(CLASSES_CIFAR10) else str(c)
                print(f"classe {c:>2} ({name:>10}): {dist[c]:>4}  ({frac:5.1f}%)")
        except Exception as e:
            print(f"[warn] contagem via DataLoader falhou: {e}")

        # Espiar um batch (formas/rótulos)
        try:
            batch = next(iter(testloader))
            imgs, ys = batch["img"], batch["label"]
            print("batch_size_real:", len(imgs))
            print("shape_da_primeira_img:", getattr(imgs[0], "shape", "desconhecido"))
            print("rotulos_do_batch:", ys[:16])
            try:
                names = [CLASSES_CIFAR10[int(t)] for t in ys[:10]]
                print("rotulos_nomes (10):", names)
            except Exception:
                pass
        except StopIteration:
            print("[warn] DataLoader vazio (sem batches).")

    return testloader




# def load_global_testloader(batch_size: int = 128, num_workers: int = 0):
#     """
#     Carrega o conjunto de TESTE do CIFAR‑10 para avaliação global no servidor.

#     Retorna
#     -------
#     testloader : torch.utils.data.DataLoader
#         DataLoader que percorre todo o split de teste (10 000 amostras).
#     """
#     global _global_test_ds

#     if _global_test_ds is None:
#         # 1) Baixa apenas o split 'test' (não faz partição)
#         _global_test_ds = load_dataset("uoft-cs/cifar10", split="test")

#         # 2) Mesmas transformações usadas pelos clientes
#         tfm = Compose([
#             ToTensor(),
#             Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ])

#         def apply_transforms(batch):
#             batch["img"] = [tfm(img) for img in batch["img"]]
#             return batch

#         _global_test_ds = _global_test_ds.with_transform(apply_transforms)

#     # 3) Constrói o DataLoader
#     testloader = DataLoader(
#         _global_test_ds,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#     )
#     return testloader


def _counts_as_dict(counts: Counter, total: int):
    """Converte Counter -> dict {classe: {count, pct, name}}"""
    out = {}
    for c in sorted(counts.keys()):
        k = int(c)
        out[str(k)] = {
            "count": int(counts[k]),
            "pct": (100.0 * counts[k] / total) if total else 0.0,
            "name": CLASSES_CIFAR10[k] if 0 <= k < len(CLASSES_CIFAR10) else str(k),
        }
    return out

def load_data(t,
    node_id,
    partition_id: int,
    num_partitions: int,
    *,
    alpha: float = 0.2,             # mesmo alpha de antes
    seed: int = 42,
    dump_stats_dir= "clients",  # <--- passe um diretório para salvar JSONs
):
    """Load partition CIFAR10 data (Dirichlet non-IID) e (opcionalmente) salva stats por cliente."""
    global fds

    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=alpha,     # ↓ => mais não-IID; ↑ => mais IID
            seed=seed,
        )
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )

    # --- carrega a partição deste cliente ---
    partition = fds.load_partition(partition_id)
    has_keys = callable(getattr(partition, "keys", None))
    base = partition["train"] if (has_keys and "train" in partition) else partition

    # --- split local train/test (antes das transforms, para podermos ler labels direto) ---
    n_total = len(base)
    if n_total <= 1:
        train_ds = base
        test_ds = base.select([])  # vazio
    else:
        test_sz = max(1, min(n_total - 1, int(round(0.2 * n_total))))
        split = base.train_test_split(test_size=test_sz, seed=seed)
        train_ds, test_ds = split["train"], split["test"]

    save_stats = False
    try:
        save_stats = dump_stats_dir is not None and int(t) == 1
    except Exception:
        save_stats = False

    if save_stats:
        os.makedirs(dump_stats_dir, exist_ok=True)
        try:
            train_labels = list(train_ds["label"])
            test_labels  = list(test_ds["label"]) if len(test_ds) > 0 else []
        except Exception as e:
            # fallback (mais lento): itera itens
            train_labels = [int(train_ds[i]["label"]) for i in range(len(train_ds))]
            test_labels  = [int(test_ds[i]["label"])  for i in range(len(test_ds))]

        train_cnt = Counter(train_labels)
        test_cnt  = Counter(test_labels)

        

        stats = {
            "t": int(t),
            "node_id": int(node_id) if node_id is not None else None,
            "client_id": int(partition_id),
            "num_partitions": int(num_partitions),
            "alpha": float(alpha),
            "n_total": int(n_total),
            "n_train": int(len(train_ds)),
            "n_test":  int(len(test_ds)),
            "train_per_class": _counts_as_dict(train_cnt, len(train_ds)),
            "test_per_class":  _counts_as_dict(test_cnt,  len(test_ds)),
        }


        # usar node_id no nome do arquivo (fallback para partition_id se vier None)
        fname = f"node_{node_id}.json" if node_id is not None else f"client_{partition_id:03d}.json"
        out_path = os.path.join(dump_stats_dir, fname)
        try:
            os.remove(out_path)
        except FileNotFoundError:
            pass

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"[cid={partition_id} node={node_id} t={t}] estatísticas salvas em: {out_path}")
    pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    train_ds.set_transform(apply_transforms)
    test_ds.set_transform(apply_transforms)

    # --- loaders ---
    n_train = len(train_ds)
    n_test  = len(test_ds)
    bs_train = min(32, max(1, n_train))
    bs_test  = min(32, max(1, n_test if n_test > 0 else 1))

    trainloader = DataLoader(train_ds, batch_size=bs_train, shuffle=True,  drop_last=False, num_workers=0)
    testloader  = DataLoader(test_ds,  batch_size=bs_test,  shuffle=False, drop_last=False, num_workers=0)

    # log mínimo (pode remover se quiser 100% silencioso)
    print(f"[cid={partition_id}] n_total={n_total} n_train={n_train} n_test={n_test} bs_train={bs_train}")

    return trainloader, testloader




# def load_data(partition_id: int, num_partitions: int):
#     """Load partition CIFAR10 data (Dirichlet non-IID) com robustez de split/transform/loader."""
#     global fds
#     if fds is None:
#         partitioner = DirichletPartitioner(
#             num_partitions=num_partitions,
#             partition_by="label",  # 
#             alpha=0.2,             # ↓ => mais não-IID; ↑ => mais IID
#             seed=42,
#         )
#         fds = FederatedDataset(
#             dataset="uoft-cs/cifar10",
#             partitioners={"train": partitioner},
#         )

    
#     partition = fds.load_partition(partition_id)

    
#     has_keys = callable(getattr(partition, "keys", None))
#     base = partition["train"] if (has_keys and "train" in partition) else partition

    
#     n_total = len(base)
#     if n_total <= 1:
#         # tudo em train, test vazio
#         train_ds = base
#         test_ds = base.select([])  # dataset vazio
#     else:
#         test_sz = max(1, min(n_total - 1, int(round(0.2 * n_total))))
#         split = base.train_test_split(test_size=test_sz, seed=42)
#         train_ds, test_ds = split["train"], split["test"]

    
#     pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#     def apply_transforms(batch):
#         batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
#         return batch

    
#     train_ds.set_transform(apply_transforms)
#     test_ds.set_transform(apply_transforms)

    
#     n_train = len(train_ds)
#     n_test = len(test_ds)
#     bs_train = min(32, max(1, n_train))
#     bs_test = min(32, max(1, n_test if n_test > 0 else 1))

#     trainloader = DataLoader(train_ds, batch_size=bs_train, shuffle=True, drop_last=False, num_workers=0)
#     testloader  = DataLoader(test_ds,  batch_size=bs_test,  shuffle=False, drop_last=False, num_workers=0)

    
#     print(f"[cid={partition_id}] n_total={n_total} n_train={n_train} n_test={n_test} bs_train={bs_train}")

#     return trainloader, testloader

# def load_data(partition_id: int, num_partitions: int):
#     """Load partition CIFAR10 data."""
#     # Only initialize `FederatedDataset` once
#     global fds
#     if fds is None:
#         partitioner = IidPartitioner(num_partitions=num_partitions)
#         fds = FederatedDataset(
#             dataset="uoft-cs/cifar10",
#             partitioners={"train": partitioner},
#         )
#     partition = fds.load_partition(partition_id)
#     # Divide data on each node: 80% train, 20% test
#     partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
#     pytorch_transforms = Compose(
#         [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     )

#     def apply_transforms(batch):
#         """Apply transforms to the partition from FederatedDataset."""
#         batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
#         return batch

#     partition_train_test = partition_train_test.with_transform(apply_transforms)
#     trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
#     testloader = DataLoader(partition_train_test["test"], batch_size=32)
#     return trainloader, testloader


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy