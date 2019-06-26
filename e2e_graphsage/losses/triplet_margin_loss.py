"""
Custom triplet margin loss
which is more flexible than the one provided by the base library
"""
import torch

VALID_DISTANCE_FNS = {
    'cos', 'cosine',
    'dot',
    'l2', 'euclidean',
    'l1',
    'huber', 'smooth_l1'
}
VALID_REDUCTIONS = {'mean', 'sum', 'none'}


class TripletMarginLoss(torch.nn.Module):
    """
    Refer to
    https://pytorch.org/docs/stable/nn.html?highlight=tripletmarginloss#torch.nn.TripletMarginLoss
    for a guide on the parameters to this function

    The differences are the follow
    - this function accepts a kwarg distance_fn
      so the distance function to use is not limited to only L_p distances
    - this function also expects negative embedding tensor provided is either
      a 2d tensor or a 3d tensor since in many cases we expect there to be more
      than just 1 negative samples
    """
    def __init__(
        self,
        margin=1.0,
        distance_fn='cosine',
        smooth_l1_sigma=1.0,
        eps=1e-6,
        swap=False,
        reduction='mean'
    ):
        super(TripletMarginLoss, self).__init__()

        assert distance_fn in VALID_DISTANCE_FNS, \
            'distance_fn should be one of {}, got {}'.format(
                VALID_DISTANCE_FNS, distance_fn)
        assert reduction in VALID_REDUCTIONS, \
            'reduction should be one of {}'.format(VALID_REDUCTIONS)

        self.margin = margin
        self.distance_fn = distance_fn

        if distance_fn in {'huber', 'smooth_l1'}:
            raise NotImplementedError('Huber loss not yet implemented')
            self.smooth_l1_sigma = smooth_l1_sigma

        self.eps = eps
        self.swap = bool(swap)
        self.reduction = reduction

    def forward(self, A, P, N, loss_only=True):
        """
        Expects
        A_shape ~ (batch_size, embedding_size)
        P_shape ~ (batch_size, embedding_size)
        N_shape ~ (batch_size, embedding_size) or
            (num_negative_samples, batch_size, embedding_size)
        """
        if len(N.shape) == 2:
            N = N.unsqueeze(0)

        if self.distance_fn in {'cos', 'cosine'}:
            A = torch.nn.functional.normalize(A, p=2, dim=-1)
            P = torch.nn.functional.normalize(P, p=2, dim=-1)
            N = torch.nn.functional.normalize(N, p=2, dim=-1)

        if self.distance_fn in {'cos', 'cosine', 'dot'}:
            pos_aff = (A * P).sum(dim=-1)
            neg_aff = (A * N).sum(dim=-1)

            if self.swap:
                neg_aff = torch.max(neg_aff, (P * N).sum(dim=-1))

            loss = neg_aff - (pos_aff - self.margin)

            if not loss_only:
                all_aff = torch.cat([pos_aff.unsqueeze(0), neg_aff], dim=0)
                indices_of_ranks = all_aff.argsort(dim=0, descending=True)

        elif self.distance_fn in {'l2', 'euclidean'}:
            pos_dist = ((A - P) ** 2).sum(dim=-1)
            neg_dist = ((A - N) ** 2).sum(dim=-1)

            if self.swap:
                neg_dist = torch.min(neg_dist, ((P - N) ** 2).sum(dim=-1))

            loss = (pos_dist + self.margin) - neg_dist

            if not loss_only:
                all_dist = torch.cat([pos_dist.unsqueeze(0), neg_dist], dim=0)
                indices_of_ranks = all_dist.argsort(dim=0, descending=False)

        elif self.distance_fn in {'l1'}:
            pos_dist = (A - P).abs().sum(dim=-1)
            neg_dist = (A - N).abs().sum(dim=-1)

            if self.swap:
                neg_dist = torch.min(neg_dist, (P - N).abs().sum(dim=-1))

            loss = (pos_dist + self.margin) - neg_dist

            if not loss_only:
                all_dist = torch.cat([pos_dist.unsqueeze(0), neg_dist], dim=0)
                indices_of_ranks = all_dist.argsort(dim=0, descending=False)

        else:
            # distance_fn in {'huber', 'smooth_l1'}
            raise NotImplementedError('Huber loss not yet implemented')

        loss = torch.nn.functional.relu(loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        if loss_only:
            return loss

        else:
            ranks = indices_of_ranks.argsort(dim=0)[0] + 1
            ranks = ranks.float()
            rr = 1 / ranks
            return loss, ranks.mean(), rr.mean()  # MRR is mean of all RR
