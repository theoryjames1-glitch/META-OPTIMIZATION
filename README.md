# META-OPTIMIZATION


[![Video Title](https://img.youtube.com/vi/qf9mJ9Yr6Jo/0.jpg)](https://www.youtube.com/watch?v=qf9mJ9Yr6Jo)

BEYOND THE WALL OF SLEEP

```python
# AdaptiveSGDTapeMeta_stable.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Differentiable Tape (bounded, leaky)
# -----------------------------

class DifferentiableTape(nn.Module):
    def __init__(self, N=32, D=16, decay=0.95):
        super().__init__()
        self.N, self.D, self.decay = N, D, decay
        # Keep as a buffer; we'll REASSIGN it (no in-place) each write
        self.register_buffer("memory", torch.zeros(N, D))
        self.write_head = nn.Linear(D, N)
        self.read_head  = nn.Linear(D, N)

    @torch.no_grad()
    def reset(self):
        # Safe because called BEFORE any inner-loop graph is built
        self.memory = torch.zeros_like(self.memory)

    def write(self, f_t):
        """
        Out-of-place update to avoid version counter issues.
        We detach writes for stability (no gradient through write head).
        """
        f_t_bounded = torch.tanh(f_t.detach())               # [1, D]
        a_w = torch.softmax(self.write_head(f_t_bounded), -1)  # [1, N]  (no grad)
        # new_memory = decay * old + outer_product(a_w, f_t_bounded)
        add = a_w.transpose(0, 1) @ f_t_bounded             # [N, D]
        new_memory = self.decay * self.memory + add         # OUT-OF-PLACE
        # Reassign reference (no in-place on the tensor used in the graph)
        self.memory = new_memory

    def read(self, f_t):
        """
        Read stays differentiable w.r.t. read_head params.
        We treat memory as a constant snapshot (no grad needed on memory).
        """
        f_tb = torch.tanh(f_t)                               # [1, D]
        a_r  = torch.softmax(self.read_head(f_tb), -1)       # [1, N] (requires grad)
        mem  = self.memory                                   # [N, D] (constant snapshot)
        mem_t = a_r @ mem                                    # [1, D]
        return mem_t


# -----------------------------
# Trainable Optimizer (meta-learned)
# -----------------------------
class AdaptiveSGDWithTape(nn.Module):
    def __init__(self, mem_dim=16):
        super().__init__()
        self.tape = DifferentiableTape(D=mem_dim, N=32, decay=0.95)
        self.coeff_proj = nn.Linear(mem_dim, 3)

    def forward(self, p, g_t, loss_t, step_t,
                base_lr=0.1, base_mom=0.0, base_wd=0.0, prev_upd=None):
        device = p.device

        # Detached, normalized features (stable)
        loss_feat = torch.log1p(loss_t.detach().abs())
        grad_feat = torch.log1p(g_t.detach().norm())
        step_feat = torch.tensor(float(step_t), device=device) / 100.0
        f_t = torch.stack([loss_feat, grad_feat, step_feat]).unsqueeze(0)
        f_t = F.pad(f_t, (0, self.tape.D - f_t.shape[-1]))

        # Stateful write (no grad through write), differentiable read
        self.tape.write(f_t)
        mem_t = self.tape.read(f_t)                          # [1, D]

        # Bounded coeffs
        raw = self.coeff_proj(mem_t)[0]
        lr_eff  = base_lr * (0.05 + 1.95 * torch.sigmoid(raw[0]))
        mom_eff = torch.clamp(base_mom + 0.5 * torch.tanh(raw[1]), 0.0, 0.99)
        wd_eff  = torch.clamp(base_wd + 1e-4 * F.softplus(raw[2]), 0.0, 1e-2)

        if prev_upd is None:
            prev_upd = torch.zeros_like(p)

        upd   = - lr_eff * (g_t + wd_eff * p) + mom_eff * prev_upd
        new_p = p + upd                                       # OUT-OF-PLACE
        return new_p, upd


# -----------------------------
# Meta-training loop (toy task)
# -----------------------------
def meta_train(
    outer_steps=50, inner_steps=10, device="cpu",
    base_lr=0.1, base_mom=0.0, base_wd=0.0, seed=0
):
    torch.manual_seed(seed)
    opt_model = AdaptiveSGDWithTape(mem_dim=16).to(device)
    meta_opt  = torch.optim.Adam(opt_model.parameters(), lr=1e-3)

    for outer in range(outer_steps):
        # Reset tape state each episode
        opt_model.tape.reset()

        # Initialize a fresh parameter p for the inner problem
        p = torch.tensor([0.0], device=device, requires_grad=True)
        prev_upd = None
        losses = []

        for inner in range(inner_steps):
            # Define a toy objective (can randomize target to improve generalization)
            target = torch.tensor([3.0], device=device)
            loss = (p - target).pow(2).sum()
            g_t  = torch.autograd.grad(loss, p, create_graph=True)[0]

            p, prev_upd = opt_model(
                p, g_t, loss, inner,
                base_lr=base_lr, base_mom=base_mom, base_wd=base_wd,
                prev_upd=prev_upd
            )
            losses.append(loss)

        meta_loss = losses[-1]

        meta_opt.zero_grad()
        meta_loss.backward()
        # Stabilize meta-updates
        torch.nn.utils.clip_grad_norm_(opt_model.parameters(), 1.0)
        meta_opt.step()

        if outer % 10 == 0:
            print(f"Outer {outer:02d} | meta_loss = {meta_loss.item():.4f}")

    return opt_model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _ = meta_train(outer_steps=50, inner_steps=10, device=device)
```
