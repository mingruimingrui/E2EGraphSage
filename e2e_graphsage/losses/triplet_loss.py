"""
Custom triplet margin loss
which is more flexible than the one provided by the base library
"""
import torch

VALID_LOSS_TYPES = {
    'hinge', 'triplet', 'margin',
    'xent', 'cross_entropy',
    'skipgram'
}
VALID_DISTANCE_FNS = {
    'cos', 'cosine',
    'dot',
    'l2', 'euclidean',
    'l1',
    'huber', 'smooth_l1'
}
VALID_REDUCTIONS = {'mean', 'sum', 'none'}


class TripletLoss(torch.nn.Module):
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
        loss_type='hinge',
        distance_fn='cosine',
        smooth_l1_sigma=1.0,
        eps=1e-6,
        swap=False,
        reduction='mean'
    ):
        super(TripletLoss, self).__init__()

        assert loss_type in VALID_LOSS_TYPES, \
            'loss_type should be one of {}, got {}'.format(
                VALID_LOSS_TYPES, loss_type)
        assert distance_fn in VALID_DISTANCE_FNS, \
            'distance_fn should be one of {}, got {}'.format(
                VALID_DISTANCE_FNS, distance_fn)
        assert reduction in VALID_REDUCTIONS, \
            'reduction should be one of {}'.format(VALID_REDUCTIONS)

        self.margin = margin
        self.distance_fn = distance_fn
        self.use_aff = self.distance_fn in {'cos', 'cosine', 'dot'}

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

        # Compute affinity / distance
        if self.distance_fn in {'cos', 'cosine', 'dot'}:
            pos_aff = (A * P).sum(dim=-1)
            neg_aff = (A * N).sum(dim=-1)
            if self.swap:
                neg_aff = torch.max(neg_aff, (P * N).sum(dim=-1))

        elif self.distance_fn in {'l2', 'euclidean'}:
            pos_dist = ((A - P) ** 2).sum(dim=-1)
            neg_dist = ((A - N) ** 2).sum(dim=-1)
            if self.swap:
                neg_dist = torch.min(neg_dist, ((P - N) ** 2).sum(dim=-1))

        elif self.distance_fn in {'l1'}:
            pos_dist = (A - P).abs().sum(dim=-1)
            neg_dist = (A - N).abs().sum(dim=-1)
            if self.swap:
                neg_dist = torch.min(neg_dist, (P - N).abs().sum(dim=-1))

        else:
            # distance_fn in {'huber', 'smooth_l1'}
            raise NotImplementedError('Huber loss not yet implemented')

        # Compute loss
        if self.loss_type in {'hinge', 'triplet', 'margin'}:
            if self.use_aff:
                loss = neg_aff - (pos_aff - self.margin)
            else:
                loss = (pos_dist + self.margin) - neg_dist
            loss = torch.nn.functional.relu(loss)

        elif self.loss_type in {'xent', 'cross_entropy'}:
            if self.use_aff:
                pos_xent = \
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        input=pos_aff,
                        target=torch.ones_like(pos_aff),
                        reduction='sum'
                    )
                neg_xent = \
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        input=neg_aff,
                        target=torch.zeros_like(neg_aff),
                        reduction='sum'
                    )
            else:
                raise Exception('{} not implemented for {}'.format(
                    self.distance_fn, self.loss_type))
            loss = pos_xent + self.negative_weights * neg_xent

        elif self.loss_type in {'skipgram'}:
            if self.use_aff:
                # Use LogSumExp to get a smooth max
                smooth_max = neg_aff.exp().sum(dim=0).log()
                loss = (smooth_max - pos_aff).sum()
            else:
                # Use LogSumExp to get a smooth max
                smooth_max = neg_dist.exp().sum(dim=0).log()
                loss = (pos_dist - smooth_max).sum()

        else:
            raise ValueError('{} is not an expected loss_type'.format(
                self.loss_type))

        # Do reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        if loss_only:
            return loss

        else:
            if self.use_aff:
                all_aff = torch.cat([pos_aff.unsqueeze(0), neg_aff], dim=0)
                indices_of_ranks = all_aff.argsort(dim=0, descending=True)
            else:
                all_dist = torch.cat([pos_dist.unsqueeze(0), neg_dist], dim=0)
                indices_of_ranks = all_dist.argsort(dim=0, descending=False)
            ranks = indices_of_ranks.argsort(dim=0)[0] + 1
            ranks = ranks.float()
            rr = 1 / ranks
            return loss, ranks.mean(), rr.mean()  # MRR is mean of all RR
