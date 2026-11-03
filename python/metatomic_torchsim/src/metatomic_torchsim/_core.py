"""TorchSim wrapper for backend-independent metatomic-core models.

Any :py:class:`metatomic.BaseModel` (NumpyModel, TorchModel, JaxModel, models
loaded from native or TorchScript plugins, ...) runs through
:py:func:`metatomic.execute_model`. Forces and stress come from the explicit
``positions`` and ``strain`` gradients the model declares for its energy output, so
the model does not need to be differentiable with torch autograd, and data crosses
array libraries through DLPack.
"""

import os
import warnings
from typing import Dict, Literal, Optional, Union

import numpy as np
import torch
import vesin
from metatensor import Labels, TensorBlock, TensorMap

from metatomic import (
    BaseModel,
    Quantity,
    System,
    execute_model,
    load_model,
    unit_conversion_factor,
)


try:
    import torch_sim as ts
    from torch_sim.models.interface import ModelInterface
except ImportError as e:
    raise ImportError(
        "the torchsim package is required for metatomic-torchsim: "
        "pip install torch-sim-atomistic"
    ) from e


def _register_external_arrays():
    # Outputs of native (C/C++) plugins and of TorchScript models loaded through the
    # metatomic-torch plugin live in arrays metatensor-python does not know about.
    # Both export DLPack, so they can be read as CPU arrays.
    import metatensor

    for origin in ("metatensor::SimpleDataArray", "metatensor_torch::TorchDataArray"):
        try:
            metatensor.register_external_data_wrapper(
                origin, metatensor.ExternalCpuArray
            )
        except Exception:  # already registered, or not supported by this metatensor
            pass


_register_external_arrays()

_PAIR_NAMES = [
    "first_atom",
    "second_atom",
    "cell_shift_a",
    "cell_shift_b",
    "cell_shift_c",
]


def _as_torch(array, device, dtype=None) -> torch.Tensor:
    """Wrap any array (numpy, torch, JAX, external) as a torch tensor, through DLPack
    when possible."""
    if isinstance(array, torch.Tensor):
        tensor = array.detach()
    elif isinstance(array, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(array))
    else:
        try:
            tensor = torch.from_dlpack(array)
        except Exception:
            tensor = torch.from_numpy(np.array(array))
    return tensor.to(device=device, dtype=dtype or tensor.dtype)


class CoreModel(ModelInterface):
    """TorchSim wrapper for metatomic-core models (:py:class:`metatomic.BaseModel`).

    Positions and energies use TorchSim's metal units (angstrom, eV). Pair lists are
    computed on the CPU with vesin, with a Verlet skin. The model runs with the dtype
    it declares in its capabilities; positions are converted to that dtype and the
    results back to the dtype of the simulation state.
    """

    def __init__(
        self,
        model: Union[BaseModel, str, os.PathLike],
        *,
        plugin_name: Optional[str] = None,
        model_options: Optional[Dict[str, str]] = None,
        device: Optional[Union[torch.device, str]] = None,
        check_consistency: bool = False,
        compute_forces: bool = True,
        compute_stress: Optional[bool] = None,
        variants: Optional[Dict[str, Optional[str]]] = None,
        non_conservative: Union[bool, Literal["forces", "stress"]] = False,
        uncertainty_threshold: Optional[float] = 0.1,
        additional_outputs: Optional[Dict[str, Quantity]] = None,
        neighbor_skin: float = 2.0,
    ) -> None:
        """
        :param model: a :py:class:`metatomic.BaseModel`, or a source string given to
            :py:func:`metatomic.load_model` (together with ``plugin_name`` and
            ``model_options``).
        :param device: device to run the model on. Defaults to the first device in
            the model's ``supported_devices`` available here.
        :param compute_stress: compute the stress (from ``strain`` gradients, or the
            ``non_conservative_stress`` output). ``None`` enables it when the model
            supports it, only for states where all systems are periodic with a
            non-zero volume.
        :param additional_outputs: maps aliases to extra :py:class:`metatomic.Quantity`
            to compute, stored in :py:attr:`additional_outputs` (as metatensor-core
            ``TensorMap``) after each call.

        The other parameters behave as in :py:class:`MetatomicModel`.
        """
        super().__init__()
        if non_conservative not in (True, False, "forces", "stress"):
            raise ValueError(
                "non_conservative must be False, True, 'forces', or 'stress'"
            )
        if neighbor_skin < 0:
            raise ValueError("neighbor_skin must be non-negative")
        if uncertainty_threshold is not None and uncertainty_threshold <= 0:
            raise ValueError("uncertainty_threshold must be positive or None")

        if not isinstance(model, BaseModel):
            model = load_model(
                os.fspath(model), options=model_options, plugin_name=plugin_name
            )
        self._model = model
        self._capabilities = model.capabilities()
        self._dtype = {"float32": torch.float32, "float64": torch.float64}[
            self._capabilities.dtype
        ]
        self._device = self._select_device(device)
        self._check_consistency = check_consistency
        self._variants = variants or {}
        self._nc_forces = non_conservative in (True, "forces")
        self._nc_stress = non_conservative in (True, "stress")
        self._skin = float(neighbor_skin)
        self._uncertainty_threshold = uncertainty_threshold

        energy = self._pick("energy", "system")
        self._auto_stress = compute_stress is None
        if compute_stress is None:
            compute_stress = self._nc_stress or "strain" in energy.gradients
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress

        gradients = []
        for gradient, needed, nc in [
            ("positions", compute_forces, self._nc_forces),
            ("strain", compute_stress, self._nc_stress),
        ]:
            if needed and not nc:
                if gradient not in energy.gradients:
                    raise ValueError(
                        f"model does not declare {gradient} gradients for "
                        f"'{energy.name}', use non_conservative or disable them"
                    )
                gradients.append(gradient)
        self._requests = {
            "energy": Quantity(
                name=energy.name, unit="eV", sample_kind="system", gradients=gradients
            )
        }
        if self._nc_forces and compute_forces:
            q = self._pick("non_conservative_force", "atom")
            self._requests["forces"] = Quantity(
                name=q.name, unit="eV/angstrom", sample_kind="atom"
            )
        if self._nc_stress and compute_stress:
            q = self._pick("non_conservative_stress", "system")
            self._requests["stress"] = Quantity(
                name=q.name, unit="eV/angstrom^3", sample_kind="system"
            )
        uq = self._pick("energy_uncertainty", "atom", required=False)
        if uq is not None and uncertainty_threshold is not None:
            self._requests["uncertainty"] = Quantity(
                name=uq.name, unit="eV", sample_kind="atom"
            )
        # used for non-periodic states when compute_stress is automatic
        self._requests_without_stress = {
            alias: q for alias, q in self._requests.items() if alias != "stress"
        }
        self._requests_without_stress["energy"] = Quantity(
            name=energy.name,
            unit="eV",
            sample_kind="system",
            gradients=[g for g in gradients if g != "strain"],
        )
        self._additional_output_requests = dict(additional_outputs or {})
        for alias, q in self._additional_output_requests.items():
            if not isinstance(alias, str) or not isinstance(q, Quantity):
                raise TypeError(
                    "additional_outputs must map aliases to Quantity objects"
                )

        self._all_requests = list(self._requests.values()) + list(
            self._additional_output_requests.values()
        )
        if model.requested_inputs(self._all_requests):
            names = [q.name for q in model.requested_inputs(self._all_requests)]
            raise ValueError(
                f"this model requests extra inputs ({', '.join(names)}), which are "
                "not implemented in metatomic-torchsim"
            )
        self._pair_lists = model.requested_pair_lists(self._all_requests)
        self._to_angstrom = unit_conversion_factor(
            self._capabilities.length_unit, "angstrom"
        )
        self._neighbors = {}

        self.additional_outputs: Dict[str, TensorMap] = {}
        """Additional outputs from the last :py:meth:`forward` call, by alias"""

    def _select_device(self, requested):
        supported = self._capabilities.supported_devices
        candidates = [str(requested)] if requested is not None else supported
        for candidate in candidates:
            kind = {"mps": "metal"}.get(
                candidate.split(":")[0], candidate.split(":")[0]
            )
            if kind not in supported:
                continue
            if kind in ("cuda", "rocm") and not torch.cuda.is_available():
                continue
            if kind == "metal" and not torch.backends.mps.is_available():
                continue
            device = torch.device(
                candidate.replace("metal", "mps").replace("rocm", "cuda")
            )
            if isinstance(self._model, torch.nn.Module):
                self._model.to(device=device)
            return device
        raise ValueError(
            f"no available device among {candidates}, model supports {supported}"
        )

    def _pick(self, name, sample_kind, required=True):
        variant = self._variants.get(name, self._variants.get("energy"))
        candidates = [
            q
            for q in self._capabilities.outputs
            if q.sample_kind == sample_kind and q.name.split("/")[0] == name
        ]
        target = name if variant is None else f"{name}/{variant}"
        matches = [q for q in candidates if q.name == target]
        if matches:
            return matches[0]
        if variant is None and len(candidates) == 1:
            return candidates[0]
        if required:
            raise ValueError(
                f"model does not unambiguously support {target!r} with "
                f"{sample_kind!r} samples"
            )
        return None

    def model(self) -> BaseModel:
        """The wrapped metatomic-core model"""
        return self._model

    def _arrays(self, array: np.ndarray):
        return (
            array
            if self._device.type == "cpu"
            else torch.as_tensor(array, device=self._device)
        )

    def _system(self, index, types, positions, cell, pbc) -> System:
        np_dtype = np.float32 if self._dtype == torch.float32 else np.float64
        positions = positions.astype(np_dtype)
        cell = np.where(pbc[:, None], cell, 0.0).astype(np_dtype)
        system = System(
            "angstrom",
            self._arrays(types.astype(np.int32)),
            self._arrays(positions),
            self._arrays(cell),
            self._arrays(pbc),
        )
        for options in self._pair_lists:
            cutoff = options.cutoff * self._to_angstrom
            key = (index, cutoff, options.full_list)
            if key not in self._neighbors:
                self._neighbors[key] = vesin.NeighborList(
                    cutoff=cutoff, full_list=options.full_list, skin=self._skin
                )
            i, j, shifts, vectors = self._neighbors[key].compute(
                points=positions.astype(np.float64),
                box=cell.astype(np.float64),
                periodic=pbc,
                quantities="ijSD",
                copy=True,
            )
            pairs = TensorBlock(
                vectors.astype(np_dtype).reshape(-1, 3, 1),
                Labels(_PAIR_NAMES, np.column_stack([i, j, shifts]).astype(np.int32)),
                [Labels.range("xyz", 3)],
                Labels.range("distance", 1),
            )
            if self._device.type != "cpu":
                pairs = pairs.to(device=self._device, arrays="torch")
            system.add_pairs(options, pairs)
        return system

    def forward(self, state: "ts.SimState", **kwargs) -> Dict[str, torch.Tensor]:
        """Compute energies, forces, and stresses for the given simulation state.

        :returns: ``"energy"`` (shape ``[n_systems]``), ``"forces"`` (shape
            ``[n_atoms, 3]``, if ``compute_forces``), and ``"stress"`` (shape
            ``[n_systems, 3, 3]``, if ``compute_stress``), with the device and
            dtype of ``state``.
        """
        device, dtype = state.positions.device, state.positions.dtype
        positions = state.positions.detach().cpu().numpy()
        cells = state.row_vector_cell.detach().cpu().numpy()
        types = state.atomic_numbers.detach().cpu().numpy()
        system_idx = state.system_idx.detach().cpu().numpy()
        pbc = np.broadcast_to(
            np.asarray(
                state.pbc.cpu() if torch.is_tensor(state.pbc) else state.pbc, dtype=bool
            ),
            (3,),
        ).copy()
        n_systems = len(cells)
        masks = [system_idx == i for i in range(n_systems)]
        offsets = torch.as_tensor(
            np.concatenate([[0], np.cumsum([m.sum() for m in masks])[:-1]]),
            device=device,
        )
        # atoms of each system are contiguous in TorchSim states
        order = np.concatenate([np.nonzero(m)[0] for m in masks])

        systems = [
            self._system(i, types[m], positions[m], cells[i], pbc)
            for i, m in enumerate(masks)
        ]
        requests = self._requests
        if self._compute_stress and not (
            pbc.all() and np.all(np.abs(np.linalg.det(cells)) > 0)
        ):
            if not self._auto_stress:
                raise ValueError(
                    "stress requires periodic systems with a non-zero volume"
                )
            requests = self._requests_without_stress
        tensors = execute_model(
            self._model,
            systems,
            list(requests.values()) + list(self._additional_output_requests.values()),
            check_consistency=self._check_consistency,
        )
        predictions = dict(zip(requests, tensors[: len(requests)], strict=True))
        self.additional_outputs = dict(
            zip(self._additional_output_requests, tensors[len(requests) :], strict=True)
        )

        def column(labels, name):
            return torch.as_tensor(
                np.asarray(labels.column(name)).astype(np.int64), device=device
            )

        def per_atom(block_values, samples):
            values = _as_torch(block_values, device, dtype).reshape(len(samples), -1)
            index = torch.as_tensor(order, device=device)[
                offsets[column(samples, "system")] + column(samples, "atom")
            ]
            out = torch.zeros(
                len(positions), values.shape[1], device=device, dtype=dtype
            )
            return out.index_add_(0, index, values)

        results = {}
        block = predictions["energy"].block()
        energy_system = column(block.samples, "system")
        values = _as_torch(block.values, device, dtype).reshape(-1)
        results["energy"] = torch.zeros(
            n_systems, device=device, dtype=dtype
        ).index_add_(0, energy_system, values)
        gradients = block.gradients_list()
        if "positions" in gradients:
            gradient = block.gradient("positions")
            results["forces"] = -per_atom(gradient.values, gradient.samples)
        if "strain" in gradients:
            gradient = block.gradient("strain")
            values = _as_torch(gradient.values, device, dtype).reshape(-1, 3, 3)
            system = energy_system[column(gradient.samples, "sample")]
            virial = torch.zeros(n_systems, 3, 3, device=device, dtype=dtype)
            virial.index_add_(0, system, values)
            volumes = torch.as_tensor(
                np.abs(np.linalg.det(cells)), device=device, dtype=dtype
            )
            results["stress"] = virial / volumes[:, None, None]

        if "forces" in predictions:
            block = predictions["forces"].block()
            forces = per_atom(block.values, block.samples)
            # remove the spurious net force of each system
            sys_idx = torch.as_tensor(system_idx, device=device)
            net = torch.zeros(n_systems, 3, device=device, dtype=dtype).index_add_(
                0, sys_idx, forces
            )
            counts = torch.as_tensor(
                [m.sum() for m in masks], device=device, dtype=dtype
            )
            results["forces"] = forces - (net / counts[:, None])[sys_idx]
        if "stress" in predictions:
            block = predictions["stress"].block()
            stress = torch.zeros(n_systems, 9, device=device, dtype=dtype)
            stress.index_add_(
                0,
                column(block.samples, "system"),
                _as_torch(block.values, device, dtype).reshape(-1, 9),
            )
            results["stress"] = stress.reshape(n_systems, 3, 3)

        if "uncertainty" in predictions:
            block = predictions["uncertainty"].block()
            uncertainty = per_atom(block.values, block.samples)[:, 0]
            exceeded = torch.nonzero(uncertainty > self._uncertainty_threshold)[
                :, 0
            ].tolist()
            if exceeded:
                suffix = (
                    f" (and {len(exceeded) - 20} more)" if len(exceeded) > 20 else ""
                )
                warnings.warn(
                    "Some of the atomic energy uncertainties are larger than the "
                    f"threshold of {self._uncertainty_threshold} eV. The prediction is "
                    f"above the threshold for atoms {exceeded[:20]}{suffix}.",
                    stacklevel=2,
                )
        return results


def _is_core_model(model, kwargs) -> bool:
    return (
        isinstance(model, BaseModel)
        or kwargs.get("plugin_name") is not None
        or kwargs.get("model_options") is not None
    )
