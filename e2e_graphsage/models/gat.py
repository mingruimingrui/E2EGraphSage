"""
GAT modules
"""

import torch
import numpy as np
from six import integer_types
from collections import Callable, Iterable

VALID_AGGREGATE_METHODS = {'concat', 'cat', 'average', 'avg'}
DEFAULT_ACTIVATION_ATTN = torch.nn.LeakyReLU(0.2, inplace=True)

__all__ = ['GATLayer', 'GAT']


class GATLayer(torch.nn.Module):
    def __init__(
        self,
        input_feature_size,
        inner_feature_size,
        num_attn_heads=1,
        aggregate_method='concat',
        use_bias_attn=False,
        activation_attn=DEFAULT_ACTIVATION_ATTN,
        use_bias_out=False,
        activation_out=None
    ):
        super(GATLayer, self).__init__()

        assert isinstance(input_feature_size, integer_types) \
            and input_feature_size > 0, \
            'input_feature_size must be a positive integer'
        assert isinstance(inner_feature_size, integer_types) \
            and inner_feature_size > 0, \
            'inner_feature_size must be a positive integer'
        assert isinstance(num_attn_heads, integer_types) \
            and num_attn_heads > 0, \
            'num_attn_heads must be a positive integer'

        assert aggregate_method in VALID_AGGREGATE_METHODS, \
            'aggregate method {} is invalid, expecting one of {}'.format(
                aggregate_method, VALID_AGGREGATE_METHODS)

        if activation_attn is not None:
            assert isinstance(activation_attn, Callable), \
                'activation_attn should be callable'
        if activation_out is not None:
            assert isinstance(activation_out, Callable), \
                'activation_out should be callable'

        self.num_attn_heads = num_attn_heads
        self.aggregate_method = aggregate_method

        self.activation_attn = activation_attn
        self.activation_out = activation_out

        fc = []
        fc_attn_self = []
        fc_attn_neigh = []

        for i in range(self.num_attn_heads):
            fc.append(torch.nn.Linear(
                input_feature_size,
                inner_feature_size,
                bias=bool(use_bias_out)
            ))
            fc_attn_self.append(torch.nn.Linear(
                input_feature_size, 1, bias=bool(use_bias_attn)))
            fc_attn_neigh.append(torch.nn.Linear(
                input_feature_size, 1, bias=bool(use_bias_attn)))

        self.fc = torch.nn.ModuleList(fc)
        self.fc_attn_self = torch.nn.ModuleList(fc_attn_self)
        self.fc_attn_neigh = torch.nn.ModuleList(fc_attn_neigh)

    def forward(self, x_self, x_neigh, mask=None):
        """
        x_self_shape ~
            (batch_size, prior_sample_size, feature_size)
        x_neigh_shape ~
            (batch_size, prior_sample_size, sample_size, feature_size)
        mask_shape ~
            (batch_size, prior_sample_size, sample_size, 1)
        """
        # Allign self and neighbor axis
        x_self = x_self.unsqueeze(dim=-2)

        # To allow self loop, simply appen self to neighbors
        x_all = torch.cat([x_self, x_neigh], dim=-2)

        # If mask should be applied, then we get the mask for x_all
        if mask is not None:
            mask_all = torch.cat([
                torch.ones_like(mask[:, :, :1]),
                mask
            ], dim=2)

        # Apply head attn head
        h_out = []
        for i in range(self.num_attn_heads):
            # Get attn logits
            attn_logits_self = self.fc_attn_self[i](x_self)
            attn_logits_neigh = self.fc_attn_neigh[i](x_all)
            attn_logits_all = attn_logits_self + attn_logits_neigh
            if self.activation_attn is not None:
                attn_logits_all = self.activation_attn(attn_logits_all)

            # Apply neighbor mask if needed
            if mask is not None:
                attn_weights = attn_logits_all * mask_all

            # Get attn weights
            attn_weights = torch.nn.functional.softmax(
                attn_logits_all, dim=-2)

            # Get hidden_state
            xw_all = self.fc[i](x_all)
            h_out.append((xw_all * attn_weights).sum(dim=-2))

        if self.aggregate_method in {'concat', 'cat'}:
            h_out = torch.cat(h_out, dim=-1)
        else:
            # aggregate_method in {'average', 'avg'}
            h_out = torch.stack(h_out, dim=-1).mean(dim=-1)

        if self.activation_out is not None:
            h_out = self.activation_out(h_out)

        return h_out


class GAT(torch.nn.Module):
    def __init__(
        self,
        expansion_rates,
        input_feature_size,
        inner_feature_size,
        output_feature_size,
        num_attn_heads=1,
        use_bias_attn=False,
        activation_attn=DEFAULT_ACTIVATION_ATTN,
        use_bias_conv=False,
        activation_conv=None,
        use_bias_out=False,
        activation_out=None
    ):
        super(GAT, self).__init__()

        assert isinstance(expansion_rates, Iterable), \
            'expansion_rates should be a list of integers'
        assert len(expansion_rates) > 0, \
            'expansion_rates should not be empty'
        for r in expansion_rates:
            assert isinstance(r, integer_types) and r > 0, \
                'expansion_rates should be a list of positive integers'

        self.expansion_rates = expansion_rates
        self.depth = (len(expansion_rates))
        self.cum_expansion_rates = \
            [1] + np.cumprod(expansion_rates).tolist()[:-1]

        layer_input_sizes = []
        conv_layers = []
        for i in range(self.depth):
            if i == 0:
                layer_input_size = input_feature_size
            else:
                layer_input_size = inner_feature_size * num_attn_heads

            layer_input_sizes.append(layer_input_size)
            conv_layers.append(GATLayer(
                input_feature_size=layer_input_size,
                inner_feature_size=inner_feature_size,
                num_attn_heads=num_attn_heads,
                aggregate_method='concat',
                use_bias_attn=bool(use_bias_attn),
                activation_attn=activation_attn,
                use_bias_out=bool(use_bias_conv),
                activation_out=activation_conv
            ))
        self.layer_input_sizes = layer_input_sizes
        self.conv_layers = torch.nn.ModuleList(conv_layers)

        self.fc = torch.nn.Linear(
            inner_feature_size * num_attn_heads,
            output_feature_size,
            bias=bool(use_bias_out)
        )

        if activation_out is not None:
            assert isinstance(activation_out, Callable), \
                'activation_out should be callable'
        self.activation_out = activation_out

    def forward(self, hierarchical_input_embeddings, masks=None):
        assert len(hierarchical_input_embeddings) == self.depth + 1, (
            'GAT built with depth of {}, '
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
