import os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from .network import PolicyValueNet
from .selfplay import ReplayBuffer, play_selfplay_games

def make_model(C=18, width=128, blocks=10, action_size=4096, device="cuda"):
    model = PolicyValueNet(C, width, blocks, action_size).to(device)
    return model

@torch.no_grad()
def model_eval_fn(model, device="cuda"):
    def _eval(batch_np):
        x = torch.from_numpy(batch_np).to(device)
        model.eval()
        p, v = model(x)
        return p.detach().cpu().numpy(), v.detach().cpu().numpy()
    return _eval

def train_iteration(model, buffer, steps=20_000, batch_size=2048, lr=3e-4, wd=1e-4, device="cuda"):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for _ in trange(steps, desc="Train"):
        if len(buffer) < batch_size:
            continue
        S, P, Z = buffer.sample(batch_size)
        S = torch.from_numpy(S).to(device)               # [B,C,8,8]
        P = torch.from_numpy(P).to(device)               # [B,4096]
        Z = torch.from_numpy(Z).to(device)               # [B]

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits, v = model(S)                         # logits [B,4096], v [B]
            # Cross-entropy with soft labels (visit dist)
            logp = torch.log_softmax(logits, dim=1)
            policy_loss = -(P * logp).sum(dim=1).mean()
            value_loss  = F.mse_loss(v, Z)
            loss = policy_loss + value_loss + 1e-4 * sum((p**2).sum() for p in model.parameters())*0.0

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(device=device)

    buffer = ReplayBuffer(capacity=300_000)

    for it in range(3):   # run a few iterations for v0
        print(f"\n=== Iteration {it} ===")
        eval_fn = model_eval_fn(model, device=device)
        # Self-play on current net
        sp_data = play_selfplay_games(eval_fn, n_games=50, sims=400, temp_moves=30, dirichlet=(0.25,0.3))
        buffer.add_many(sp_data)
        print("Buffer size:", len(buffer))

        # Train
        train_iteration(model, buffer, steps=10_000, batch_size=1024, lr=3e-4, wd=1e-4, device=device)

        # Save
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/az_v0_it{it}.pt")

    # Final save
    torch.save(model.state_dict(), "checkpoints/az_v0_final.pt")
    print("Saved to checkpoints/az_v0_final.pt")

if __name__ == "__main__":
    main()
