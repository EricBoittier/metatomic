"""Three implementations of an anisotropic quadratic CV; real backend arrays/AD.

q = 1/2 sum_k w_k (r_2k-r_0k)^2, w=(1,2,3).
Anisotropy deliberately gives a non-symmetric strain derivative so transposition
errors cannot hide behind the symmetry of a distance or radius-of-gyration CV.
"""

import numpy as np
from metatensor import Labels, TensorBlock, TensorMap


class BackendModel:
    def __init__(self, backend):
        self.backend = backend
        self.calls = []
        self.unloads = 0
        self.fail = False
        if backend == "torch":
            import torch

            self.module = torch
        elif backend == "jax":
            import jax
            import jax.numpy as jnp

            jax.config.update("jax_enable_x64", True)
            self.module = jnp

            def function(p, strain):
                d = (p[2] - p[0]) @ strain
                return 0.5 * jnp.sum(jnp.array([1.0, 2.0, 3.0]) * d * d)

            self.differentiate = jax.jit(jax.value_and_grad(function, argnums=(0, 1)))
        else:
            self.module = np

    def evaluate(self, system, requested):
        system.arrays_backend = self.backend
        positions = system.positions
        self.calls.append(
            (len(system), type(positions).__module__, str(positions.dtype), requested)
        )
        if self.fail and len(system):
            raise RuntimeError(f"intentional {self.backend} model failure")
        if len(system) == 0:
            value = np.zeros((0, 1))
            pg = np.zeros((0, 3, 1))
            sg = np.zeros((0, 3, 3, 1))
            samples = Labels("system", np.zeros((0, 1), dtype=np.int32))
            gsamples = Labels(
                ["sample", "system", "atom"], np.zeros((0, 3), dtype=np.int32)
            )
            ssamples = Labels("sample", np.zeros((0, 1), dtype=np.int32))
        else:
            assert len(system) == 3
            if self.backend == "numpy":
                d = positions[2] - positions[0]
                g = d * np.array([1.0, 2.0, 3.0])
                q = 0.5 * np.dot(d, g)
                grad = np.zeros_like(positions)
                grad[0] = -g
                grad[2] = g
                strain = np.outer(d, g)
            elif self.backend == "torch":
                torch = self.module
                p = positions.detach().clone().requires_grad_(True)
                f = torch.eye(3, dtype=torch.float64, requires_grad=True)
                d = (p[2] - p[0]) @ f
                q = (
                    0.5
                    * (torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64) * d * d).sum()
                )
                grad, strain = torch.autograd.grad(q, (p, f))
                q = q.detach().numpy()
                grad = grad.detach().numpy()
                strain = strain.detach().numpy()
            else:
                q, (grad, strain) = self.differentiate(
                    positions, self.module.eye(3, dtype=self.module.float64)
                )
                q, grad, strain = np.asarray(q), np.asarray(grad), np.asarray(strain)
            value = np.asarray(q, dtype=np.float64).reshape(1, 1)
            pg = np.asarray(grad, dtype=np.float64)[:, :, None]
            sg = np.asarray(strain, dtype=np.float64)[None, :, :, None]
            samples = Labels("system", np.array([[0]], dtype=np.int32))
            gsamples = Labels(
                ["sample", "system", "atom"],
                np.array([[0, 0, i] for i in range(3)], dtype=np.int32),
            )
            ssamples = Labels("sample", np.array([[0]], dtype=np.int32))
        props = Labels("q", np.array([[0]], dtype=np.int32))
        block = TensorBlock(value, samples, [], props)
        if "positions" in requested:
            block.add_gradient(
                "positions", TensorBlock(pg, gsamples, [Labels.range("xyz", 3)], props)
            )
        if "strain" in requested:
            block.add_gradient(
                "strain",
                TensorBlock(
                    sg,
                    ssamples,
                    [Labels.range("xyz_1", 3), Labels.range("xyz_2", 3)],
                    props,
                ),
            )
        return TensorMap(Labels("_", np.array([[0]], dtype=np.int32)), [block])
