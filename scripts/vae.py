# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# %%
JSONL_PATH = "group_embeddings.jsonl"
OUT_JSONL = "vae_latent_vectors.jsonl"

LATENT_DIM = 12
HIDDEN1 = 512
HIDDEN2 = 128

BATCH_SIZE = 32
EPOCHS = 500
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BETA = 1.0
SEED = 42


# %%
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_embedding_dataframe(jsonl_path: str) -> pd.DataFrame:
    jsonl_path = Path(jsonl_path)

    folders = []
    vectors = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            folders.append(obj["folder"])
            vectors.append(obj["embedding"])

    if not vectors:
        raise ValueError("No embeddings found in file.")

    dim = len(vectors[0])
    columns = [f"emb_{i+1}" for i in range(dim)]
    df = pd.DataFrame(vectors, index=folders, columns=columns, dtype=float)
    df.index.name = "folder"
    return df


class StandardScalerNP:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, x: np.ndarray):
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted.")
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted.")
        return x * self.scale_ + self.mean_


# %%
class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden1: int, hidden2: int, latent_dim: int):
        super().__init__()

        self.enc_fc1 = nn.Linear(input_dim, hidden1)
        self.enc_fc2 = nn.Linear(hidden1, hidden2)
        self.mu_layer = nn.Linear(hidden2, latent_dim)
        self.logvar_layer = nn.Linear(hidden2, latent_dim)

        self.dec_fc1 = nn.Linear(latent_dim, hidden2)
        self.dec_fc2 = nn.Linear(hidden2, hidden1)
        self.out_layer = nn.Linear(hidden1, input_dim)

    def encode(self, x: torch.Tensor):
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        return self.out_layer(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(
    x: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
):
    recon_loss = F.mse_loss(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl
    return total, recon_loss, kl


# %%
def train_vae(
    x_np: np.ndarray,
    latent_dim: int = 20,
    hidden1: int = 512,
    hidden2: int = 128,
    batch_size: int = 32,
    epochs: int = 300,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    beta: float = 1.0,
    seed: int = 42,
):
    set_seed(seed)
    device = get_device()
    print("device:", device)

    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    dataset = TensorDataset(x_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VAE(
        input_dim=x_np.shape[1],
        hidden1=hidden1,
        hidden2=hidden2,
        latent_dim=latent_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    model.train()
    n_samples = x_np.shape[0]

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(batch_x)
            loss, recon_loss, kl = vae_loss(batch_x, recon, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl.item()

        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(
                f"epoch={epoch:4d} "
                f"loss={total_loss / n_samples:.6f} "
                f"recon={total_recon / n_samples:.6f} "
                f"kl={total_kl / n_samples:.6f}"
            )

    return model, device


# %%
def extract_latent_means(
    model: VAE,
    x_np: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_np, dtype=torch.float32, device=device)
        mu, _ = model.encode(x_tensor)
        z = mu.detach().cpu().numpy()
    return z


def save_latent_jsonl(df_latent: pd.DataFrame, out_jsonl: str):
    out_path = Path(out_jsonl)
    with out_path.open("w", encoding="utf-8") as f:
        for folder, row in df_latent.iterrows():
            obj = {
                "folder": folder,
                "embedding": [float(v) for v in row.values],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# %%
# load embeddings
df = load_embedding_dataframe(JSONL_PATH)
print("n modules:", df.shape[0])
print("embedding dim:", df.shape[1])

# standardize
scaler = StandardScalerNP().fit(df.values.astype(np.float32))
x_std = scaler.transform(df.values.astype(np.float32))

# train VAE
model, device = train_vae(
    x_std,
    latent_dim=LATENT_DIM,
    hidden1=HIDDEN1,
    hidden2=HIDDEN2,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    beta=BETA,
    seed=SEED,
)

# extract latent means
z = extract_latent_means(model, x_std, device=device)

df_latent = pd.DataFrame(
    z,
    index=df.index,
    columns=[f"z{i+1}" for i in range(z.shape[1])],
)
df_latent.index.name = "folder"

print("latent dim:", df_latent.shape[1])
print(df_latent.head())

# save
save_latent_jsonl(df_latent, OUT_JSONL)
print("saved:", OUT_JSONL)