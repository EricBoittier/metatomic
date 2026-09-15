import ctypes
import json
import math
import re
import struct
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from ctypes_dlpack import DLManagedTensorVersioned, DLPackArray, array_as_dlpack
from metatensor import TensorBlock, TensorMap

from ._c_api import (
    c_uintptr_t,
    mta_string_t,
    mta_system_data_kind,
    mta_system_t,
    mts_block_t,
    mts_tensormap_t,
)
from ._c_lib import _get_library
from ._status import check_pointer


_HEX_NUMBER = re.compile(r"(0[xX])?[0-9a-fA-F]+")


def _hex_from_cutoff(value: float) -> str:
    """
    Get the hexadecimal representation of the bit pattern of ``value``.

    Storing floating point values as their bit pattern makes the JSON
    round-trip exact, without relying on the precision of the decimal
    representation.
    """
    bits = struct.unpack("<Q", struct.pack("<d", value))[0]
    return hex(bits)


def _cutoff_from_hex(value: str) -> float:
    """
    Inverse of :py:func:`_hex_from_cutoff`, reading a ``f64`` from the hexadecimal
    representation of its bit pattern.

    ``context`` is used to build the error message if ``value`` is not a valid
    hexadecimal string.
    """
    if isinstance(value, str) and _HEX_NUMBER.fullmatch(value) is not None:
        bits = int(value, 16)
    else:
        bits = 2**64

    if bits >= 2**64:
        raise ValueError(
            "'cutoff' in JSON for PairListOptions must be a hex-encoded string, "
            f"got '{value}'"
        )

    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _format_metadata(metadata: dict) -> str:
    """
    Call ``mta_format_metadata`` to render a JSON-serialized
    :py:class:`ModelMetadata` as human-readable text.

    The formatting is done by the shared library to make sure all the languages
    supported by metatomic produce exactly the same output.
    """
    from ._c_lib import _get_library

    lib = _get_library()

    printed = mta_string_t()
    lib.mta_format_metadata(json.dumps(metadata).encode("utf8"), ctypes.byref(printed))
    try:
        return lib.mta_string_view(printed).decode("utf8")
    finally:
        lib.mta_string_free(printed)


def _check_string_list(values, context: str) -> list[str]:
    """
    Check that ``values`` is a list of strings, and return it as a new
    :py:class:`list`. ``context`` is used to build the error messages.
    """
    if not isinstance(values, list):
        raise ValueError(f"{context} must be an array")

    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{context} must be an array of strings")

    return list(values)


### ================================================================================ ###


class PairListOptions:
    """
    Options for the calculation of a pair list (also known as a neighbor list).

    A model declares the pair lists it needs with these options, and the engine running
    the model is then responsible for computing matching pair lists and attaching them
    to the systems given to the model.
    """

    def __init__(
        self,
        *,
        cutoff: float,
        full_list: bool,
        strict: bool = True,
        requestors: Optional[Sequence[str]] = None,
    ):
        """
        :param cutoff: spherical cutoff radius for this pair list, in the length unit of
            the model
        :param full_list: should the pair list be a full list (containing both the pair
            ``i -> j`` and ``j -> i``) or a half list (containing only ``i -> j``)
        :param strict: does the list only contain pairs within the cutoff (``True``) or
            can it also contain pairs slightly beyond the cutoff (``False``)
        :param requestors: list of strings describing who requested this pair list. More
            requestors can be added later with :py:meth:`add_requestor`.
        """
        self.cutoff = cutoff
        self.full_list = full_list
        self.strict = strict
        self.requestors = [] if requestors is None else requestors

    @property
    def cutoff(self) -> float:
        """
        Spherical cutoff radius for this pair list, in the length unit of the model.
        """
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value: float):
        cutoff = float(value)

        if not math.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("cutoff must be a finite positive number")

        self._cutoff = cutoff

    @property
    def full_list(self) -> bool:
        """
        Should the pair list be a full list (containing both the pair ``i -> j`` and ``j
        -> i``) or a half list (containing only ``i -> j``)?
        """
        return self._full_list

    @full_list.setter
    def full_list(self, value: bool):
        self._full_list = bool(value)

    @property
    def strict(self) -> bool:
        """
        Does this list only contain pairs within the cutoff (``True``), or can it also
        contain pairs slightly beyond the cutoff (``False``)?
        """
        return self._strict

    @strict.setter
    def strict(self, value: bool):
        self._strict = bool(value)

    @property
    def requestors(self) -> list[str]:
        """
        List of strings describing who requested this pair list.

        The returned list is a copy, use :py:meth:`add_requestor` to register a new
        requestor.
        """
        return list(self._requestors)

    @requestors.setter
    def requestors(self, value: Sequence[str]):
        self._requestors = []
        for requestor in value:
            self.add_requestor(requestor)

    def add_requestor(self, requestor: str):
        """
        Add ``requestor`` to the list of entities requesting this pair list. Empty
        strings and duplicates are ignored.

        :param requestor: string describing who is requesting this pair list
        """
        requestor = str(requestor)
        if requestor != "" and requestor not in self._requestors:
            self._requestors.append(requestor)

    def to_dict(self) -> dict:
        """
        Convert this object to a JSON-compatible dictionary, following the
        :ref:`documented format <core-json-pair-options>`.
        """
        return {
            "type": "metatomic_pair_list_options",
            # store the bit pattern so the float round-trips exactly
            "cutoff": _hex_from_cutoff(self.cutoff),
            "full_list": self.full_list,
            "strict": self.strict,
            "requestors": self.requestors,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PairListOptions":
        """
        Create a :py:class:`PairListOptions` from a JSON-compatible dictionary,
        following the :ref:`documented format <core-json-pair-options>`.

        :param data: dictionary containing the data, typically obtained by parsing JSON
            with :py:func:`json.loads`
        :raises ValueError: if the data does not match the expected format
        """
        if not isinstance(data, dict):
            raise ValueError(
                "invalid JSON data for PairListOptions, expected an object"
            )

        valid_keys = set(["type", "cutoff", "full_list", "strict", "requestors"])
        for key in data.keys():
            if key not in valid_keys:
                raise ValueError(f"unexpected key '{key}' in JSON for PairListOptions")

        if data.get("type") != "metatomic_pair_list_options":
            raise ValueError(
                "'type' in JSON for PairListOptions must be "
                "'metatomic_pair_list_options'"
            )

        cutoff = _cutoff_from_hex(data.get("cutoff"))
        if not math.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError(
                "'cutoff' in JSON for PairListOptions must be a finite positive number"
            )

        if not isinstance(data.get("full_list"), bool):
            raise ValueError(
                "'full_list' in JSON for PairListOptions must be a boolean"
            )

        if not isinstance(data.get("strict"), bool):
            raise ValueError("'strict' in JSON for PairListOptions must be a boolean")

        requestors = []
        if "requestors" in data:
            requestors = _check_string_list(
                data["requestors"], "'requestors' in JSON for PairListOptions"
            )

        return cls(
            cutoff=cutoff,
            full_list=data["full_list"],
            strict=data["strict"],
            requestors=requestors,
        )

    def __repr__(self) -> str:
        return (
            f"PairListOptions(cutoff={self._cutoff}, full_list={self._full_list}, "
            f"strict={self._strict})"
        )

    def _comparison_key(self):
        # the list of requestors is intentionally left out: two requests with
        # the same parameters can be fulfilled by the same pair list, whoever
        # asked for them
        return (self._cutoff, self._full_list, self._strict)

    # `PairListOptions` are compared by cutoff first, then `full_list` and
    # finally `strict`; the list of requestors is ignored everywhere.

    def __eq__(self, other) -> bool:
        """
        Check if two :py:class:`PairListOptions` are equal.

        The list of requestors is ignored when checking for equality.
        """
        if not isinstance(other, PairListOptions):
            return NotImplemented
        return self._comparison_key() == other._comparison_key()

    def __ne__(self, other) -> bool:
        """
        Check if two :py:class:`PairListOptions` are different.

        The list of requestors is ignored when checking for equality.
        """
        if not isinstance(other, PairListOptions):
            return NotImplemented
        return self._comparison_key() != other._comparison_key()

    def __lt__(self, other) -> bool:
        """Check if this pair list sorts before ``other``"""
        if not isinstance(other, PairListOptions):
            return NotImplemented
        return self._comparison_key() < other._comparison_key()

    def __le__(self, other) -> bool:
        """Check if this pair list sorts before ``other`` or is equal to it"""
        if not isinstance(other, PairListOptions):
            return NotImplemented
        return self._comparison_key() <= other._comparison_key()

    def __gt__(self, other) -> bool:
        """Check if this pair list sorts after ``other``"""
        if not isinstance(other, PairListOptions):
            return NotImplemented
        return self._comparison_key() > other._comparison_key()

    def __ge__(self, other) -> bool:
        """Check if this pair list sorts after ``other`` or is equal to it"""
        if not isinstance(other, PairListOptions):
            return NotImplemented
        return self._comparison_key() >= other._comparison_key()

    def __hash__(self) -> int:
        return hash(self._comparison_key())


def _as_dlpack(array) -> ctypes.POINTER(DLManagedTensorVersioned):
    """Convert a numpy array to a DLPack pointer. The array must outlive the call."""
    return array_as_dlpack(np.ascontiguousarray(array))


def _from_dlpack(tensor) -> np.ndarray:
    """Take a borrowed DLPack tensor from the C API and wrap it as numpy."""
    return np.from_dlpack(DLPackArray(tensor))


def _string_from_mta(printed) -> str:
    lib = _get_library()
    try:
        return lib.mta_string_view(printed).decode("utf8")
    finally:
        lib.mta_string_free(printed)


class System:
    """
    An atomistic system used as input to metatomic models.

    This is a RAII wrapper around the ``mta_system_t`` type from the C API. It
    can either own the underlying system (in which case it is freed with the
    :py:class:`System`), or be a non-owning view into a system owned elsewhere
    (for example a system passed to a model by the runtime).

    All C API errors are raised as :py:class:`metatomic.MetatomicError`.
    """

    def __init__(self, length_unit, types, positions, cell, pbc):
        """
        Create a new :py:class:`System` from numpy arrays.

        Ownership of the four arrays is transferred to the new system through
        DLPack. The arrays must already have the dtype and layout expected by
        :c:func:`mta_system_create`.

        :param length_unit: unit of length used by ``positions`` and ``cell``
        :param types: array with shape ``(n_atoms,)`` of atomic types (``int32``)
        :param positions: array with shape ``(n_atoms, 3)`` of atomic positions
        :param cell: array with shape ``(3, 3)`` of unit cell vectors
        :param pbc: array with shape ``(3,)`` of periodic boundary conditions
        """
        self._lib = _get_library()
        self._is_view = False

        ptr = ctypes.POINTER(mta_system_t)()
        self._lib.mta_system_create(
            str(length_unit).encode("utf8"),
            _as_dlpack(types),
            _as_dlpack(positions),
            _as_dlpack(cell),
            _as_dlpack(pbc),
            ctypes.byref(ptr),
        )
        check_pointer(ptr)
        self._ptr = ptr

    def _check_ptr(self):
        if not self._ptr:
            raise ValueError("this System has been released and can no longer be used")

    def _check_not_view(self, method_name: str):
        if self._is_view:
            raise ValueError(
                f"can not call System.{method_name} on this system since it is "
                "a view of a system owned elsewhere."
            )

    @staticmethod
    def unsafe_from_ptr(system):
        """
        Create an owning :py:class:`System` from a raw ``mta_system_t`` pointer.

        The :py:class:`System` takes ownership of the pointer and will free it
        when garbage-collected.
        """
        check_pointer(system)
        obj = System.__new__(System)
        obj._lib = _get_library()
        obj._ptr = system
        obj._is_view = False
        return obj

    @staticmethod
    def unsafe_view_from_ptr(system):
        """
        Create a non-owning :py:class:`System` view from a raw ``mta_system_t``
        pointer. The system will *not* be freed when the :py:class:`System` is
        destroyed, and must outlive it.
        """
        check_pointer(system)
        obj = System.__new__(System)
        obj._lib = _get_library()
        obj._ptr = system
        obj._is_view = True
        return obj

    def as_mta_system_t(self):
        """
        Get the underlying C pointer for this :py:class:`System`.

        This class still manages the system memory after the call. Use
        :py:meth:`System.release` to take ownership of the pointer.
        """
        self._check_ptr()
        return self._ptr

    def release(self):
        """
        Release the underlying C pointer of this :py:class:`System`.

        This class is no longer managing the system memory after the call.
        """
        self._check_not_view("release")
        ptr = self.as_mta_system_t()
        self._ptr = None
        self._is_view = True
        return ptr

    def __del__(self):
        if (
            getattr(self, "_lib", None) is not None
            and getattr(self, "_ptr", None)
            and not getattr(self, "_is_view", True)
        ):
            self._lib.mta_system_free(self._ptr)

    def __len__(self) -> int:
        return self.size

    @property
    def size(self) -> int:
        """Number of atoms in this system."""
        self._check_ptr()
        size = c_uintptr_t()
        self._lib.mta_system_size(self._ptr, ctypes.byref(size))
        return size.value

    @property
    def length_unit(self) -> str:
        """Unit of length used by the positions and cell of this system."""
        self._check_ptr()
        unit = mta_string_t()
        self._lib.mta_system_get_length_unit(self._ptr, ctypes.byref(unit))
        return _string_from_mta(unit)

    def _data(self, kind) -> np.ndarray:
        self._check_ptr()
        tensor = ctypes.POINTER(DLManagedTensorVersioned)()
        self._lib.mta_system_get_data(self._ptr, kind, ctypes.byref(tensor))
        check_pointer(tensor)
        return _from_dlpack(tensor)

    @property
    def types(self) -> np.ndarray:
        """Atomic types of all atoms, as an array with shape ``(n_atoms,)``."""
        return self._data(mta_system_data_kind.MTA_SYSTEM_DATA_TYPES)

    @property
    def positions(self) -> np.ndarray:
        """Positions of all atoms, as an array with shape ``(n_atoms, 3)``."""
        return self._data(mta_system_data_kind.MTA_SYSTEM_DATA_POSITIONS)

    @property
    def cell(self) -> np.ndarray:
        """Unit cell, as an array with shape ``(3, 3)``."""
        return self._data(mta_system_data_kind.MTA_SYSTEM_DATA_CELL)

    @property
    def pbc(self) -> np.ndarray:
        """Periodic boundary conditions, as an array with shape ``(3,)``."""
        return self._data(mta_system_data_kind.MTA_SYSTEM_DATA_PBC)

    def _options_json(self, options: Union[PairListOptions, str]) -> bytes:
        if isinstance(options, PairListOptions):
            return json.dumps(options.to_dict()).encode("utf8")
        return str(options).encode("utf8")

    def add_pairs(self, options: Union[PairListOptions, str], pairs: TensorBlock):
        """
        Add a pair list (neighbor list) to this system.

        Ownership of ``pairs`` is transferred to this :py:class:`System`.

        :param options: :py:class:`PairListOptions` or a JSON string describing
            the pair list
        :param pairs: pair data, stored as a metatensor block
        """
        self._check_ptr()
        if not isinstance(pairs, TensorBlock):
            raise TypeError(
                f"`pairs` must be a metatensor TensorBlock, not {type(pairs)}"
            )
        self._lib.mta_system_add_pairs(
            self._ptr, self._options_json(options), pairs.release()
        )

    def pairs(self, options: Union[PairListOptions, str]) -> TensorBlock:
        """
        Get a previously stored pair list matching ``options``.

        The returned block is a non-owning view into data owned by this
        :py:class:`System`.
        """
        self._check_ptr()
        block = ctypes.POINTER(mts_block_t)()
        self._lib.mta_system_get_pairs(
            self._ptr, self._options_json(options), ctypes.byref(block)
        )
        check_pointer(block)
        return TensorBlock.unsafe_view_from_ptr(block, parent=self)

    def known_pairs(self) -> list[PairListOptions]:
        """Options of all pair lists registered with this system."""
        self._check_ptr()
        options = mta_string_t()
        self._lib.mta_system_known_pairs(self._ptr, ctypes.byref(options))
        data = json.loads(_string_from_mta(options))
        return [PairListOptions.from_dict(item) for item in data]

    def add_custom_data(self, name: str, data: TensorMap):
        """
        Add custom data to this system, stored under ``name``.

        Ownership of ``data`` is transferred to this :py:class:`System`.
        """
        self._check_ptr()
        if not isinstance(data, TensorMap):
            raise TypeError(f"`data` must be a metatensor TensorMap, not {type(data)}")
        self._lib.mta_system_add_custom_data(
            self._ptr, str(name).encode("utf8"), data.release()
        )

    def custom_data(self, name: str) -> TensorMap:
        """
        Get the custom data previously stored under ``name``.

        The returned tensor map is a non-owning view into data owned by this
        :py:class:`System`.
        """
        self._check_ptr()
        data = ctypes.POINTER(mts_tensormap_t)()
        self._lib.mta_system_get_custom_data(
            self._ptr, str(name).encode("utf8"), ctypes.byref(data)
        )
        check_pointer(data)
        return TensorMap.unsafe_view_from_ptr(data, parent=self)

    def known_custom_data(self) -> list[str]:
        """Names of all custom data registered with this system."""
        self._check_ptr()
        names = mta_string_t()
        self._lib.mta_system_known_custom_data(self._ptr, ctypes.byref(names))
        return list(json.loads(_string_from_mta(names)))

    def __repr__(self) -> str:
        if not getattr(self, "_ptr", None):
            return "System(<released>)"
        return f"System({self.size} atoms, length_unit={self.length_unit!r})"
