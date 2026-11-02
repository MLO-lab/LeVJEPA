import torch
import torch.distributed as dist

class SIGReg(torch.nn.Module):
    """Sketched Isotropic Gaussian Regularizer (LeJEPA): an Epps-Pulley test of the embeddings' 1-D projections along
    `num_proj` random directions against a standard Gaussian, integrated over `knots` points of [0, 3]."""

    def __init__(self, knots=17, num_proj=1024, normalize_by_n=False):
        super().__init__()
        self.num_proj = num_proj
        self.normalize_by_n = normalize_by_n
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """proj (V, B, D): V views of B embeddings."""
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        distributed = dist.is_available() and dist.is_initialized()
        if distributed:
            # ranks are seeded differently; the ECFs averaged below must be measured along the same directions
            dist.broadcast(A, src=0)
        # empirical characteristic function (cos/sin means over the batch dim)
        x_t = (proj @ A).unsqueeze(-1) * self.t
        ecf = torch.stack((x_t.cos().mean(-3), x_t.sin().mean(-3)))
        world_size = 1
        if distributed:
            dist.all_reduce(ecf, op=dist.ReduceOp.AVG)
            world_size = dist.get_world_size()
        # epps-pulley statistic on the global-averaged ECF
        err = (ecf[0] - self.phi).square() + ecf[1].square()
        if self.normalize_by_n:
            statistic = err @ self.weights
        else:
            statistic = (err @ self.weights) * proj.size(-2) * world_size
        return statistic.mean()  # average over projections and views
