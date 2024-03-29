import torch
import torch.nn.functional as F
import torch.distributed as dist

from core.utils import dist as link

from torch.nn.modules.loss import _Loss


class InfoNCE(_Loss):
    def __init__(self, temperature):
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    @staticmethod
    def cosine_similarity(p, z):
        # [N, E]

        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)
        # [N E] [N E] -> [N] -> [1]
        return (p * z).sum(dim=1).mean()  # dot product & batch coeff normalization

    def loss(self, p, z_gather, weights=None):
        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)

        offset = link.get_rank() * p.shape[0]
        labels = torch.arange(offset, offset + p.shape[0], dtype=torch.long).cuda()
        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]
        if weights is not None:
            return F.cross_entropy(p_z_m, labels, weight=weights), p_z_m
        else:
            return F.cross_entropy(p_z_m, labels), p_z_m

    def forward(self, p1, z1, p2, z2, weights=None):
        p1 = p1.split(z2.size(0), dim=0)
        p2 = p2.split(z1.size(0), dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())
        if weights is not None:
            weights = weights.cuda()
            weights = concat_all_gather(weights)
            
        loss = 0
        p_z_m = torch.zeros(z1.shape[0], z2_gather.shape[0]).cuda()
        for p in p1:
            # with torch.no_grad():
            loss_, p_z_m_ = self.loss(p, z2_gather, weights)
            loss = loss + loss_
            with torch.no_grad():
                p_z_m = p_z_m + p_z_m_
        for p in p2:
            loss_, p_z_m_ = self.loss(p, z1_gather, weights)
            loss = loss + loss_
            with torch.no_grad():
                p_z_m = p_z_m + p_z_m_
            # self.loss(p, z1_gather)
        with torch.no_grad():
            p_z_m = p_z_m / (len(p1) + len(p2))
        return loss / (len(p1) + len(p2)), p_z_m


@torch.no_grad()
def concat_all_gather(tensor):
    """gather the given tensor"""
    tensors_gather = [torch.ones_like(tensor) for _ in range(link.get_world_size())]
    dist.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output
