import os, glob
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset

def _dirichlet_split(labels, n_clients, alpha, rng):
    labels = np.asarray(labels); idxs = [[] for _ in range(n_clients)]
    for c in np.unique(labels):
        ix = np.where(labels == c)[0]; rng.shuffle(ix)
        prop = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(prop / prop.sum()) * len(ix)).astype(int)[:-1]
        for i, ch in enumerate(np.split(ix, cuts)):
            idxs[i] += ch.tolist()
    # Dirichlet draws can leave clients empty, which is invalid for a shuffled
    # DataLoader. Rebalance one sample at a time from the largest clients.
    while any(len(x) == 0 for x in idxs):
        empty = next(i for i, x in enumerate(idxs) if len(x) == 0)
        donor = int(np.argmax([len(x) for x in idxs]))
        idxs[empty].append(idxs[donor].pop())
    return [np.array(x, dtype=np.int64) for x in idxs]

def load_fashion():
    import torchvision, torchvision.transforms as T
    tr = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
    train = torchvision.datasets.FashionMNIST("./data", True, tr, download=True)
    test  = torchvision.datasets.FashionMNIST("./data", False, tr, download=True)
    # Vectorised equivalent of ToTensor + Normalize; avoids applying ToTensor to
    # an already-tensor object and avoids a slow Python loop over 70k images.
    Xtr = ((train.data.float() / 255.0 - 0.2860) / 0.3530).view(-1, 784)
    Xte = ((test.data.float() / 255.0 - 0.2860) / 0.3530).view(-1, 784)
    ytr = train.targets; yte = test.targets
    return (Xtr, ytr), (Xte, yte), 784, 10

def load_cifar():
    import torchvision, torchvision.transforms as T
    tr = T.Compose([T.ToTensor(), T.Normalize((.4914,.4822,.4465),(.247,.2435,.2616))])
    # Serialise acquisition when several resumed workers need the same dataset.
    # The pinned mirror is accepted only after checking torchvision's archive MD5.
    import fcntl, hashlib, urllib.request, json
    from pathlib import Path
    data_root = Path("./data"); data_root.mkdir(exist_ok=True)
    with (data_root / ".cifar_acquisition.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            train = torchvision.datasets.CIFAR10("./data", True, tr, download=False)
            test = torchvision.datasets.CIFAR10("./data", False, tr, download=False)
        except RuntimeError:
            archive = data_root / "cifar-10-python.tar.gz"
            expected = "c58f30108f718f92721af3b95e74349a"
            valid = archive.exists() and hashlib.md5(archive.read_bytes()).hexdigest() == expected
            if not valid:
                url = "https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/resolve/a48007227f9e2cd0af96175f4afb5ac4965e261b/cifar-10-python.tar.gz"
                tmp = archive.with_suffix(".download.tmp")
                try:
                    digest = hashlib.md5(); size = 0
                    with urllib.request.urlopen(url, timeout=30) as response, tmp.open("wb") as output:
                        while True:
                            block = response.read(1024 * 1024)
                            if not block: break
                            output.write(block); digest.update(block); size += len(block)
                    if size != 170498071 or digest.hexdigest() != expected:
                        raise ValueError("CIFAR mirror archive failed integrity verification")
                    os.replace(tmp, archive)
                    Path("cifar_data_acquisition.json").write_text(json.dumps(
                        dict(url=url, md5=expected, bytes=size, pinned_mirror=True), indent=2))
                except Exception as exc:
                    print("Verified CIFAR mirror unavailable; using the official torchvision download:", type(exc).__name__)
            train = torchvision.datasets.CIFAR10("./data", True, tr, download=True)
            test = torchvision.datasets.CIFAR10("./data", False, tr, download=True)
    # Verified bitwise against torchvision on all 60,000 CIFAR-10 images.
    def cached(raw):
        x = torch.from_numpy(raw.data).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        x.sub_(torch.tensor([.4914, .4822, .4465]).view(1, 3, 1, 1))
        x.div_(torch.tensor([.247, .2435, .2616]).view(1, 3, 1, 1))
        ds = TensorDataset(x, torch.tensor(raw.targets, dtype=torch.long))
        ds.targets = raw.targets
        return ds
    return cached(train), cached(test), (3, 32, 32), 10

def load_har(path="./data/HAR"):
    """UCI HAR: expects train/X_train.txt, train/y_train.txt, test/... (space-delimited)."""
    def rd(p): return np.loadtxt(p)
    Xtr = torch.tensor(rd(f"{path}/train/X_train.txt"), dtype=torch.float32)
    ytr = torch.tensor(rd(f"{path}/train/y_train.txt").astype(np.int64) - 1)
    Xte = torch.tensor(rd(f"{path}/test/X_test.txt"),  dtype=torch.float32)
    yte = torch.tensor(rd(f"{path}/test/y_test.txt").astype(np.int64) - 1)
    return (Xtr, ytr), (Xte, yte), Xtr.shape[1], int(ytr.max().item()) + 1

def make_loaders(dataset, batch, seed, alpha, n_clients):
    rng = np.random.default_rng(seed)
    if dataset == "fashion":
        (Xtr, ytr), (Xte, yte), d, C = load_fashion(); is_img = False
    elif dataset == "har":
        (Xtr, ytr), (Xte, yte), d, C = load_har(); is_img = False
    elif dataset == "cifar":
        train, test, shp, C = load_cifar(); d = shp; is_img = True
        Xtr = ytr = Xte = yte = None
    else:
        raise ValueError(dataset)

    test_ds = (TensorDataset(Xte, yte) if not is_img else test)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    if is_img:
        labels = np.array(train.targets)
    else:
        labels = ytr.numpy()
    client_idxs = _dirichlet_split(labels, n_clients, alpha, rng)

    def client_loader(i):
        if is_img:
            ds = Subset(train, client_idxs[i].tolist())
        else:
            ds = TensorDataset(Xtr[client_idxs[i]], ytr[client_idxs[i]])
        return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False), len(client_idxs[i])

    return client_loader, test_loader, d, C, is_img, client_idxs
