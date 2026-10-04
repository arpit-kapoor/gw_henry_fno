import torch
import torch.nn as nn
import torch.nn.functional as F


def pointwise_conv(conv, x):
    """Apply a kernel-size-1 ``nn.ConvNd`` to channel-first ``x``.

    On Apple MPS the 1x1 convolution kernels are several times slower (and use
    more memory) than the equivalent channel matmul, so compute it as
    ``W @ x`` there. Uses the conv's own parameters, so checkpoints are
    unaffected.
    """
    if x.device.type != "mps":
        return conv(x)
    y = torch.matmul(conv.weight.flatten(1), x.flatten(2))
    if conv.bias is not None:
        y = y + conv.bias[:, None]
    return y.view(x.shape[0], -1, *x.shape[2:])


class MLP(nn.Module):
    """A Multi-Layer Perceptron, with arbitrary number of layers

    Parameters
    ----------
    in_channels : int
    out_channels : int, default is None
        if None, same is in_channels
    hidden_channels : int, default is None
        if None, same is in_channels
    n_layers : int, default is 2
        number of linear layers in the MLP
    non_linearity : default is F.gelu
    dropout : float, default is 0
        if > 0, dropout probability
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        hidden_channels=None,
        n_layers=2,
        n_dim=2,
        non_linearity=F.gelu,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = (
            in_channels if hidden_channels is None else hidden_channels
        )
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
            if dropout > 0.0
            else None
        )

        Conv = getattr(nn, f"Conv{n_dim}d")
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0 and i == (n_layers - 1):
                self.fcs.append(Conv(self.in_channels, self.out_channels, 1))
            elif i == 0:
                self.fcs.append(Conv(self.in_channels, self.hidden_channels, 1))
            elif i == (n_layers - 1):
                self.fcs.append(Conv(self.hidden_channels, self.out_channels, 1))
            else:
                self.fcs.append(Conv(self.hidden_channels, self.hidden_channels, 1))

    def forward(self, x):
        if x.device.type == "mps":
            return self._forward_channels_last(x)
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        return x

    def _forward_channels_last(self, x):
        """Same as :meth:`forward`, computed in channels-last layout.

        On MPS a fused ``F.linear`` over (B, *S, C) is much faster than a
        channel-first matmul plus a separate bias add, and the wide hidden
        activations are never permuted -- only the narrow input and output.
        """
        x = x.movedim(1, -1)
        for i, fc in enumerate(self.fcs):
            x = F.linear(x, fc.weight.flatten(1), fc.bias)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)
        return x.movedim(-1, 1).contiguous()
