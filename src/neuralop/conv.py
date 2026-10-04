from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


import tensorly as tl
from tensorly.plugins import use_opt_einsum
from tltorch.factorized_tensors.core import FactorizedTensor

tl.set_backend("pytorch")
use_opt_einsum("optimal")
einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _contract_dense(x, weight, separable=False):
    order = tl.ndim(x)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:])  # no batch-size

    # batch-size, out_channels, x, y...
    if separable:
        out_syms = [x_syms[0]] + list(weight_syms)
    else:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]

    eq = f'{"".join(x_syms)},{"".join(weight_syms)}->{"".join(out_syms)}'

    if not torch.is_tensor(weight):
        weight = weight.to_tensor()

    return tl.einsum(eq, x, weight)




class SpectralConv(nn.Module):
    """Generic N-Dimensional Fourier Neural Operator

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels
    out_channels : int, optional
        Number of output channels
    max_n_modes : None or int tuple, default is None
        Number of modes to use for contraction in Fourier domain during training.
 
        .. warning::
            
            We take care of the redundancy in the Fourier modes, therefore, for an input 
            of size I_1, ..., I_N, please provide modes M_K that are I_1 < M_K <= I_N
            We will automatically keep the right amount of modes: specifically, for the 
            last mode only, if you specify M_N modes we will use M_N // 2 + 1 modes 
            as the real FFT is redundant along that last dimension.

            
        .. note::

            Provided modes should be even integers. odd numbers will be rounded to the closest even number.  

        This can be updated dynamically during training.

    max_n_modes : int tuple or None, default is None
        * If not None, **maximum** number of modes to keep in Fourier Layer, along each dim
            The number of modes (`n_modes`) cannot be increased beyond that.
        * If None, all the n_modes are used.

    n_layers : int, optional
        Number of Fourier Layers, by default 4
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        max_n_modes=None,
        bias=True,
        n_layers=1,
        init_std="auto",
        fft_norm="backward",
        rank=0.5
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # n_modes is the total number of modes kept along each dimension
        self.n_modes = n_modes
        self.order = len(self.n_modes)

        if max_n_modes is None:
            max_n_modes = self.n_modes
        elif isinstance(max_n_modes, int):
            max_n_modes = [max_n_modes]
        self.max_n_modes = max_n_modes

        self.n_layers = n_layers

        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels))**0.5
        else:
            init_std = init_std

        self.fft_norm = fft_norm
        factorization = "ComplexDense"
        fixed_rank_modes = None
        self.rank = rank

        weight_shape = (in_channels, out_channels, *max_n_modes)


        self.weight = nn.ModuleList(
            [
                FactorizedTensor.new(
                    weight_shape,
                    rank=self.rank,
                    factorization=factorization,
                    fixed_rank_modes=fixed_rank_modes
                )
                for _ in range(n_layers)
            ]
        )
        for w in self.weight:
            w.normal_(0, init_std)
        self._contract = _contract_dense

        if bias:
            self.bias = nn.Parameter(
                init_std
                * torch.randn(*((n_layers, self.out_channels) + (1,) * self.order))
            )
        else:
            self.bias = None

    def _get_weight(self, index):
        return self.weight[index]
    
    @property
    def n_modes(self):
        return self._n_modes
    
    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int): # Should happen for 1D FNO only
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        # The last mode has a redundacy as we use real FFT
        # As a design choice we do the operation here to avoid users dealing with the +1
        n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    def forward(
        self, x: torch.Tensor, indices=0, output_shape: Optional[Tuple[int]] = None
    ):
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)
        indices : int, default is 0
            if joint_factorization, index of the layers for n_layers > 1

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        if x.device.type == "mps" and output_shape is None:
            return self._forward_dft(x, indices)

        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient
        fft_dims = list(range(-self.order, 0))

        x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
        if self.order > 1:
            x = torch.fft.fftshift(x, dim=fft_dims[:-1])

        out_dtype = torch.cfloat
        out_fft = torch.zeros([batchsize, self.out_channels, *fft_size],
                              device=x.device, dtype=out_dtype)
        starts = [(max_modes - min(size, n_mode)) for (size, n_mode, max_modes) in zip(fft_size, self.n_modes, self.max_n_modes)]
        slices_w =  [slice(None), slice(None)] # Batch_size, channels
        slices_w += [slice(start//2, -start//2) if start else slice(start, None) for start in starts[:-1]]
        slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)] # The last mode already has redundant half removed
        weight = self._get_weight(indices)[slices_w]

        starts = [(size - min(size, n_mode)) for (size, n_mode) in zip(list(x.shape[2:]), list(weight.shape[2:]))]
        slices_x =  [slice(None), slice(None)] # Batch_size, channels
        slices_x += [slice(start//2, -start//2) if start else slice(start, None) for start in starts[:-1]]
        slices_x += [slice(None, -starts[-1]) if starts[-1] else slice(None)] # The last mode already has redundant half removed
        out_fft[slices_x] = self._contract(x[slices_x], weight, separable=False)


        if output_shape is not None:
            mode_sizes = output_shape
        
        if self.order > 1:
            out_fft = torch.fft.fftshift(out_fft, dim=fft_dims[:-1])
            
        x = torch.fft.irfftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x

    def _kept_mode_bins(self, mode_sizes, weight_modes):
        """FFT bin indices kept by the FFT path, per spatial dim.

        Returns ``[(in_bins, out_bins), ...]``: ``in_bins[j]`` is the frequency
        bin of ``rfftn(x)`` feeding mode slot ``j`` of the weight, and
        ``out_bins[j]`` is the bin of ``irfftn``'s input that slot ``j`` is
        written to. They are derived by replaying the fftshift/slice/fftshift
        sequence of :meth:`forward` on index arrays, so both paths stay
        identical (including for odd sizes, where the two fftshifts do not
        cancel).
        """
        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1] // 2 + 1
        bins = []
        for dim, (size, n_mode) in enumerate(zip(fft_size, weight_modes)):
            start = size - min(size, n_mode)
            idx = torch.arange(size)
            if dim == self.order - 1:
                kept = slice(None, -start) if start else slice(None)
                bins.append((idx[kept], idx[kept]))
                continue
            kept = slice(start // 2, -start // 2) if start else slice(start, None)
            in_bins = torch.fft.fftshift(idx)[kept]
            slot = torch.full((size,), -1, dtype=torch.long)
            slot[kept] = torch.arange(len(in_bins))
            slot = torch.fft.fftshift(slot)
            out_bins = torch.empty_like(in_bins)
            out_bins[slot[slot >= 0]] = idx[slot >= 0]
            bins.append((in_bins, out_bins))
        return bins

    def _dft_matrices(self, mode_sizes, weight_modes, device, dtype):
        """Real-valued truncated DFT matrices for :meth:`_forward_dft` (cached).

        The first spatial dim (when there are several) is transformed on its
        own: ``fwd_lead`` is ``(2m, n)`` and ``inv_lead`` is ``(2n, m)``, the
        real and imaginary parts of a complex matrix stacked along rows and
        applied from the left. The remaining dims are flattened and handled by
        one Kronecker-product matrix applied from the right, ``fwd_trail`` as
        ``(re, im)`` of shape ``(N, M)`` and ``inv_trail`` as ``(re, -im)`` of
        shape ``(M, N)``, with the irfft Hermitian weights and the FFT
        normalisation folded in. Both are large contiguous matmuls, which MPS
        handles far better than many tiny batched ones.
        """
        key = (tuple(mode_sizes), tuple(weight_modes), device, dtype)
        cache = self.__dict__.setdefault("_dft_cache", {})
        if key in cache:
            return cache[key]

        n_total = 1
        for n in mode_sizes:
            n_total *= n
        fwd_scale, inv_scale = {
            "backward": (1.0, 1.0 / n_total),
            "forward": (1.0 / n_total, 1.0),
            "ortho": (n_total ** -0.5, n_total ** -0.5),
        }[self.fft_norm]

        def phase(n, k):
            # exp(i * 2*pi * k * t / n) of shape (n, len(k)), reducing k*t
            # mod n exactly before going to floating point.
            t = torch.arange(n)
            angle = 2 * torch.pi * ((t[:, None] * k[None, :]) % n).double() / n
            return torch.polar(torch.ones_like(angle), angle)

        fwd, inv = [], []
        bins = self._kept_mode_bins(mode_sizes, weight_modes)
        for dim, (n, (in_bins, out_bins)) in enumerate(zip(mode_sizes, bins)):
            fwd.append(phase(n, in_bins).conj())  # (n, m), kernel exp(-i...)
            inv.append(phase(n, out_bins).T)  # (m, n), kernel exp(+i...)
        # irfft along the last dim: Hermitian weights, imag of DC/Nyquist dropped.
        n = mode_sizes[-1]
        herm = torch.full((len(bins[-1][1]), 1), 2.0, dtype=torch.float64)
        herm[bins[-1][1] == 0] = 1.0
        if n % 2 == 0:
            herm[bins[-1][1] == n // 2] = 1.0
        inv[-1] = inv[-1] * herm

        n_lead = 1 if self.order > 1 else 0
        fwd_trail, inv_trail = fwd[n_lead], inv[n_lead]
        for f, e in zip(fwd[n_lead + 1 :], inv[n_lead + 1 :]):
            fwd_trail = torch.kron(fwd_trail, f)
            inv_trail = torch.kron(inv_trail, e)
        fwd_trail = fwd_trail * fwd_scale
        inv_trail = inv_trail * inv_scale

        def to(m):
            return m.to(device=device, dtype=dtype)

        mats = {
            "fwd_trail": (to(fwd_trail.real), to(fwd_trail.imag)),
            "inv_trail": (to(inv_trail.real), to(-inv_trail.imag)),
        }
        if n_lead:
            mats["fwd_lead"] = to(torch.cat([fwd[0].T.real, fwd[0].T.imag]))
            mats["inv_lead"] = to(torch.cat([inv[0].T.real, inv[0].T.imag]))
        cache[key] = mats
        return mats

    def _forward_dft(self, x: torch.Tensor, indices=0):
        """Same result as :meth:`forward`, using only real-valued matmuls.

        Used on Apple MPS, where complex tensors are not supported by the
        gather/scatter kernels. Because only a handful of Fourier modes are
        kept, explicit truncated DFTs are cheap and map onto fast matmuls.
        Real and imaginary parts are kept as separate contiguous tensors so
        every transform is a matmul on a view (no permute copies, which are
        slow on MPS).
        """
        batchsize, channels, *mode_sizes = x.shape
        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1] // 2 + 1

        # Same weight slicing as forward(); the raw parameter is (in, out, *modes, 2).
        starts = [(max_modes - min(size, n_mode)) for (size, n_mode, max_modes) in zip(fft_size, self.n_modes, self.max_n_modes)]
        slices_w =  [slice(None), slice(None)]
        slices_w += [slice(start//2, -start//2) if start else slice(start, None) for start in starts[:-1]]
        slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]
        weight = self._get_weight(indices)._parameters["tensor"][tuple(slices_w)]
        weight_modes = weight.shape[2:-1]

        mats = self._dft_matrices(mode_sizes, weight_modes, x.device, x.dtype)
        has_lead = self.order > 1

        def along_lead(z_re, z_im, mat):
            # Left-multiply the first spatial dim (axis 2) of z by ``mat``.
            b, c, n, rest = z_re.shape
            p_re = mat @ z_re.reshape(b * c, n, rest)
            p_im = mat @ z_im.reshape(b * c, n, rest)
            m = mat.shape[0] // 2
            return (
                (p_re[:, :m] - p_im[:, m:]).view(b, c, m, rest),
                (p_re[:, m:] + p_im[:, :m]).view(b, c, m, rest),
            )

        # Forward truncated DFT: trailing dims (real -> complex), then the first.
        if has_lead:
            x = x.reshape(batchsize, channels, mode_sizes[0], -1)
        k_re, k_im = mats["fwd_trail"]
        z_re, z_im = x @ k_re, x @ k_im
        if has_lead:
            z_re, z_im = along_lead(z_re, z_im, mats["fwd_lead"])

        # Complex channel contraction with the (in, out, *modes, 2) weight.
        w_re, w_im = weight[..., 0], weight[..., 1]
        w_block = torch.cat(
            [torch.cat([w_re, w_im], 1), torch.cat([-w_im, w_re], 1)], 0
        ).flatten(2)  # (2C_in, 2C_out, K)
        z = torch.cat([z_re, z_im], 1).flatten(2)  # (B, 2C_in, K)
        z = torch.einsum("bik,iok->bok", z, w_block)
        out_shape = (batchsize, self.out_channels, *z_re.shape[2:])
        z_re = z[:, : self.out_channels].reshape(out_shape)
        z_im = z[:, self.out_channels :].reshape(out_shape)

        # Inverse truncated DFT: first dim, then trailing dims (complex -> real).
        if has_lead:
            z_re, z_im = along_lead(z_re, z_im, mats["inv_lead"])
        q_re, q_im = mats["inv_trail"]
        x = (z_re @ q_re + z_im @ q_im).reshape(batchsize, self.out_channels, *mode_sizes)

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x

    def get_conv(self, indices):
        """Returns a sub-convolutional layer from the joint parametrize main-convolution

        The parametrization of sub-convolutional layers is shared with the main one.
        """
        if self.n_layers == 1:
            Warning("A single convolution is parametrized, directly use the main class.")

        return SubConv(self, indices)

    def __getitem__(self, indices):
        return self.get_conv(indices)


class SubConv(nn.Module):
    """Class representing one of the convolutions from the mother joint
    factorized convolution.

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules, they all point to
    the same data, which is shared.
    """

    def __init__(self, main_conv, indices):
        super().__init__()
        self.main_conv = main_conv
        self.indices = indices

    def forward(self, x, **kwargs):
        return self.main_conv.forward(x, self.indices, **kwargs)

    def transform(self, x, **kwargs):
        return self.main_conv.transform(x, self.indices, **kwargs)

    @property
    def weight(self):
        return self.main_conv.get_weight(indices=self.indices)