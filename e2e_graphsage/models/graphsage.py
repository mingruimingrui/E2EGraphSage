"""
graphsage modules
"""

import torch
import numpy as np
from six import integer_types
from collections import Callable, Iterable

VALID_POOLING_METHODS = {'max', 'min', 'mean'}
DEFAULT_ACIVATION_NEIGH = torch.nn.modules.activation.ReLU(inplace=True)

__all__ = ['SageLayer', 'GraphSage']


class SageLayer(torch.nn.Module):
    def __init__(
        self,
        input_feature_size,
        inner_feature_size,
        pooling_method='max',
        use_bias_neigh=True,
        activation_neigh=DEFAULT_ACIVATION_NEIGH,
        use_bias_out=False,
        activation_out=None
    ):
        super(SageLayer, self).__init__()

        assert isinstance(input_feature_size, integer_types) \
            and input_feature_size > 0, \
            'input_feature_size must be a positive integer'
        assert isinstance(inner_feature_size, integer_types) \
            and inner_feature_size > 0, \
            'inner_feature_size must be a positive integer'

        assert pooling_method in VALID_POOLING_METHODS, \
            'pooling method {} is invalid, expecting one of {}'.format(
                pooling_method, VALID_POOLING_METHODS)

        if activation_neigh is not None:
            assert isinstance(activation_neigh, Callable), \
                'activation_neigh should be callable'
        if activation_out is not None:
            assert isinstance(activation_out, Callable), \
                'activation_out should be callable'

        self.pooling_method = pooling_method

        self.activation_neigh = activation_neigh
        self.activation_out = activation_out

        self.fc_self = torch.nn.Linear(
            input_feature_size,
            inner_feature_size,
            bias=bool(use_bias_out)
        )
        self.fc_neigh = torch.nn.Linear(
            input_feature_size,
            inner_feature_size,
            bias=bool(use_bias_neigh)
        )
        self.fc_pool = torch.nn.Linear(
            inner_feature_size,
            inner_feature_size,
            bias=bool(use_bias_out)
        )

    def forward(self, x_self, x_neigh, mask=None):
        """
        x_self_shape ~
            (batch_size, prior_sample_size, feature_size)
        x_neigh_shape ~
            (batch_size, prior_sample_size, sample_size, feature_size)
        mask_shape ~
            (batch_size, prior_sample_size, sample_size, 1)
        """
        # Apply linear layer to all nodes
        h_self = self.fc_self(x_self)
        h_neigh = self.fc_neigh(x_neigh)

        if self.activation_neigh is not None:
            h_neigh = self.activation_neigh(h_neigh)

        # Apply neighbor mask if needed
        if mask is not None:
            h_neigh = h_neigh * mask

        # Aggregate neighbors
        if self.pooling_method == 'max':
            h_pool, _ = h_neigh.max(dim=-2)

        elif self.pooling_method == 'min':
            h_pool, _ = h_neigh.min(dim=-2)

        else:
            # pooling_method == 'mean'
            h_pool = h_neigh.mean(dim=-2)

        # Apply linear layer to aggregated neighbors
        h_pool = self.fc_pool(h_pool)

        # Concat and return
        h_out = torch.cat([h_self, h_pool], dim=-1)

        if self.activation_out is not None:
            h_out = self.activation_out(h_out)

        return h_out


class GraphSage(torch.nn.Module):
    def __init__(
        self,
        expansion_rates,
        input_feature_size,
        inner_feature_size,
        output_feature_size,
        pooling_method='max',
        use_bias_neigh=False,
        activation_neigh=DEFAULT_ACIVATION_NEIGH,
        use_bias_conv=False,
        activation_conv=None,
        use_bias_out=False,
        activation_out=None
    ):
        super(GraphSage, self).__init__()

        assert isinstance(expansion_rates, Iterable), \
            'expansion_rates should be a list of integers'
        assert len(expansion_rates) > 0, \
            'expansion_rates should not be empty'
        for r in expansion_rates:
            assert isinstance(r, integer_types) and r > 0, \
                'expansion_rates should be a list of positive integers'

        self.expansion_rates = expansion_rates
        self.depth = len(expansion_rates)
        self.cum_expansion_rates = \
            [1] + np.cumprod(expansion_rates).tolist()[:-1]

        layer_input_sizes = []
        conv_layers = []
        for i in range(self.depth):
            if i == 0:
                layer_input_size = input_feature_size
            else:
                layer_input_size = inner_feature_size * 2

            layer_input_sizes.append(layer_input_size)
            conv_layers.append(SageLayer(
                input_feature_size=layer_input_size,
                inner_feature_size=inner_feature_size,
                pooling_method=pooling_method,
                use_bias_neigh=bool(use_bias_neigh),
                activation_neigh=activation_neigh,
                use_bias_out=bool(use_bias_conv),
                activation_out=activation_conv
            ))
        self.layer_input_sizes = layer_input_sizes
        self.conv_layers = torch.nn.ModuleList(conv_layers)

        self.fc = torch.nn.Linear(
            inner_feature_size * 2,
            output_feature_size,
            bias=bool(use_bias_out)
        )

        if activation_out is not None:
            assert isinstance(activation_out, Callable), \
                'activation_out should be callable'
        self.activation_out = activation_out

    def forward(self, hierarchical_input_embeddings, masks):
        assert len(hierarchical_input_embeddings) == self.depth + 1, (
            'GraphSage built with depth of {}, '
            'received hierarchical input embeddings with len {}. '
            'hierarchical input embeddings should have size of depth + 1'
        ).format(self.depth, len(hierarchical_input_embeddings))

        if masks is not None:
            assert len(masks) == self.depth, (
                'GAT built with depth of {}, '
                'received masks with len {}. '
                'masks should have size of depth'
            ).format(self.depth, len(hierarchical_input_embeddings))

        # Initial state is equal to input embeddings
        current_hidden_states = list(hierarchical_input_embeddings)

        # Prod function will be used to compute cummulative expansion_rates
        def prod(values):
            res = 1
            for v in values:
                res *= v
            return res

        # Apply each conv layer
        for i, layer in enumerate(self.conv_layers):

            # Apply for all relevant nodes
            for j in range(self.depth - i):
                x_self = current_hidden_states[j]
                x_neigh = current_hidden_states[j + 1]

                x_neigh = x_neigh.reshape(
                    -1,
                    self.cum_expansion_rates[j],
                    self.expansion_rates[j],
                    self.layer_input_sizes[i]
                )

                if masks is None:
                    current_hidden_states[j] = layer(x_self, x_neigh)
                else:
                    mask = masks[j]
                    mask = mask.reshape(
                        -1,
                        self.cum_expansion_rates[j],
                        self.expansion_rates[j],
                        1
                    )
                    current_hidden_states[j] = layer(x_self, x_neigh, mask)

        h_out = self.fc(current_hidden_states[0][:, 0].contiguous())

        if self.activation_out is not None:
            h_out = self.activation_out(h_out)

        return h_out
