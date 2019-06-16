import torch
from six import integer_types


class RollingNegativeSampler(torch.nn.Module):
    """
    Rolling negative sampler
    Takes in an embeddings tensor, and rolls along the first dimension
    to perform negative sampling
    """
    def __init__(
        self,
        num_negative_samples=20,
    ):
        super(RollingNegativeSampler, self).__init__()
        assert isinstance(num_negative_samples, integer_types) \
            and num_negative_samples > 0, \
            'Expecting num negative samples to be a positive integer'
        self.num_negative_samples = num_negative_samples

    def forward(self, embeddings):
        negative_samples = []
        for i in range(1, self.num_negative_samples + 1):
            negative_samples.append(embeddings.roll(i))
        negative_samples = torch.stack(negative_samples, dim=0)
        return negative_samples
