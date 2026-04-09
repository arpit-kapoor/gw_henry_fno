import torch
from torch import nn
from abc import ABC, abstractmethod

class Embedding(nn.Module, ABC):
    """
    Abstract base class for embedding layers.
    """
    def __init__(self):
        super().__init__()
    
    @property
    @abstractmethod
    def out_channels(self):
        pass

class SinusoidalEmbedding(Embedding):
    """
    SinusoidalEmbedding provides a unified sinusoidal positional embedding
    in the styles of Transformers [1]_ and Neural Radiance Fields (NERFs) [2]_.

    Expects inputs of shape ``(batch, n_in, in_channels)`` or ``(n_in, in_channels)``

    Parameters
    ----------
    in_channels : int
        Number of input channels to embed
    num_frequencies : int, optional
        Number of frequencies in positional embedding.
        By default, set to the number of input channels
    embedding_type : {'transformer', 'nerf'}
        Type of embedding to apply. For a function with N input channels, 
        each channel value p is embedded via a function g with 2L channels 
        such that g(p) is a 2L-dim vector. For 0 <= k < L:

        * 'transformer' for transformer-style encoding.

            g(p)_k = sin((p / max_positions) ^ {k / N})
            g(p)_{k+1} = cos((p / max_positions) ^ {k / N})

        * 'nerf' : NERF-style encoding.  

            g(p)_k = sin(2^(k) * Pi * p)
            g(p)_{k+1} = cos(2^(k) * Pi * p)

    max_positions : int, optional
        Maximum number of positions for the encoding, default 10000
        Only used if `embedding_type == 'transformer'`.

    References
    ----------
    .. [1] Vaswani, A. et al (2017)
        "Attention Is All You Need". 
        NeurIPS 2017, https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf. 

    .. [2] Mildenhall, B. et al (2020)
        "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis".
        ArXiv, https://arxiv.org/pdf/2003.08934. 
    """
    def __init__(self, 
                 in_channels: int,
                 num_frequencies: int = None, 
                 embedding_type: str = 'transformer', 
                 max_positions: int = 10000):
        super().__init__()
        self.in_channels = in_channels
        self.num_frequencies = num_frequencies
        
        # verify embedding type
        allowed_embeddings = ['nerf', 'transformer']
        assert embedding_type in allowed_embeddings, \
            f"Error: embedding_type expected one of {allowed_embeddings}, received {embedding_type}"
        self.embedding_type = embedding_type
        if self.embedding_type == "transformer":
            assert max_positions is not None, "Error: max_positions must have an int value for transformer embedding."
        self.max_positions = max_positions
    
    @property
    def out_channels(self):
        """
        Returns the number of output channels after embedding.
        """
        return 2 * self.num_frequencies * self.in_channels

    def forward(self, x):
        """
        Parameters 
        ----------
        x : torch.Tensor
            shape (n_in, self.in_channels) or (batch, n_in, self.in_channels)
        Returns
        -------
        torch.Tensor
            Embedded tensor with sinusoidal positional encoding.
        """
        assert x.ndim in [2, 3], (
            f"Error: expected inputs of shape (batch, n_in, {self.in_channels}) "
            f"or (n_in, channels), got inputs with ndim={x.ndim}, shape={x.shape}"
        )
        if x.ndim == 2:
            batched = False
            x = x.unsqueeze(0)
        else:
            batched = True
        batch_size, n_in, _ = x.shape
        
        if self.embedding_type == 'nerf':
            # NeRF-style: frequencies are powers of 2 times pi
            freqs = 2 ** torch.arange(0, self.num_frequencies, device=x.device) * torch.pi
        elif self.embedding_type == 'transformer':
            # Transformer-style: frequencies are spaced logarithmically
            freqs = torch.arange(0, self.num_frequencies, device=x.device) / (self.num_frequencies * 2)
            freqs = (1 / self.max_positions) ** freqs
        
        # Compute the outer product of input positions and frequencies
        # Resulting shape: (batch_size, n_in, in_channels, num_frequencies)
        freqs = torch.einsum('bij, k -> bijk', x, freqs)

        # Stack sin and cos along a new last dimension, then flatten
        # Shape after stack: (batch_size, n_in, in_channels, num_frequencies, 2)
        freqs = torch.stack((freqs.sin(), freqs.cos()), dim=-1)

        # Flatten the last two dimensions to interleave sin and cos
        freqs = freqs.view(batch_size, n_in, -1)
        
        if not batched:
            freqs = freqs.squeeze(0)
        return freqs
