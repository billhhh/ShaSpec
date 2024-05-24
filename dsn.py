"""
Bill Wang
22/07/2021
"""

from torch import nn
import torch


class _DomainSpecificNorm3d(nn.Module):
    _version = 2

    def __init__(self, norm_cfg, num_features, num_domains=2, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_DomainSpecificNorm3d, self).__init__()

        if norm_cfg == 'BN':
            self.dsns = nn.ModuleList(
                [nn.BatchNorm3d(num_features) for _ in range(num_domains)])
        elif norm_cfg == 'SyncBN':
            self.dsns = nn.ModuleList(
                [nn.SyncBatchNorm(num_features) for _ in range(num_domains)])
        elif norm_cfg == 'GN':
            self.dsns = nn.ModuleList(
                [nn.GroupNorm(16, num_features) for _ in range(num_domains)])
        elif norm_cfg == 'IN':
            self.dsns = nn.ModuleList(
                [nn.InstanceNorm3d(num_features, affine=True) for _ in range(num_domains)])

    def reset_running_stats(self):
        for dsn in self.dsns:
            dsn.reset_running_stats()

    def reset_parameters(self):
        for dsn in self.dsns:
            dsn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, domain_ids):
        self._check_input_dim(x)
        if len(set(domain_ids)) == 1:  # list has same values
            dsn = self.dsns[domain_ids[0]]
            return dsn(x)
        else:
            rst = []
            for batch_id in range(len(domain_ids)):
                dsn = self.dsns[domain_ids[batch_id]]
                rst.append(dsn(x[batch_id].unsqueeze(0)))
            return torch.cat(rst, dim=0)


class DomainSpecificNorm3d(_DomainSpecificNorm3d):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
