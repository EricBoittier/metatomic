import ctypes
import itertools

import metatensor as mts
import numpy as np
import pytest
from metatensor import Labels, TensorBlock, TensorMap

from metatomic import (
    ModelCapabilities,
    ModelMetadata,
    PairListOptions,
    Quantity,
    System,
    execute_model,
)


jax = pytest.importorskip("jax")
jnp = jax.numpy

from metatomic import (  # noqa: E402
    JaxModel,
    JaxPairs,
    JaxSystem,
    unit_conversion_factor,
)


PAIRS_OPTIONS = PairListOptions(cutoff=5.0, full_list=False, strict=True)


@jax.jit
def harmonic(k, vectors):
    return 0.5 * k * (vectors**2).sum()


class HarmonicPairs:
    """E = 1/2 * k * sum_ij |r_ij|^2, over all pairs in the list"""

    def __init__(self):
        self.k = jnp.asarray(2.0)

    def requested_pair_lists(self, requested_outputs):
        return [PAIRS_OPTIONS]

    def requested_inputs(self, requested_outputs):
        return [Quantity(name="test::input", unit="eV", sample_kind="atom")]

    def __call__(self, system):
        return harmonic(self.k, system.pairs(PAIRS_OPTIONS).values.reshape(-1, 3))


class NestedModel:
    def requested_pair_lists(self, requested_outputs):
        return [PairListOptions(cutoff=3.0, full_list=True, strict=True)]


class PairModel(JaxModel):
    def __init__(self):
        self.pairs = HarmonicPairs()

    def capabilities(self):
        return ModelCapabilities(
            atomic_types=[1, 6, 8],
            interaction_range=5.0,
            length_unit="angstrom",
            supported_devices=["cpu"],
            dtype="float32",
            outputs=[
                Quantity(name="energy", unit="eV", sample_kind="system"),
                Quantity(name="test::vector", unit="", sample_kind="system"),
                Quantity(name="test::per_atom", unit="", sample_kind="atom"),
                Quantity(name="charge", unit="e", sample_kind="system"),
            ],
        )

    def metadata(self):
        return ModelMetadata(name="pair model")

    def requested_pair_lists(self, requested_outputs):
        return [PAIRS_OPTIONS]

    def forward(self, systems, selected_atoms, outputs):
        assert selected_atoms is None

        results = []
        for request in outputs:
            assert request.gradients == []
            for system in systems:
                assert isinstance(system.positions, jax.Array)

            if request.name == "energy":
                block = TensorBlock(
                    values=jnp.stack([self.pairs(s) for s in systems]).reshape(-1, 1),
                    samples=Labels.range("system", len(systems)),
                    components=[],
                    properties=Labels.range("energy", 1),
                )
            else:
                assert request.name == "test::vector"
                sums = jnp.stack(
                    [
                        s.pairs(PAIRS_OPTIONS).values.reshape(-1, 3).sum(axis=0)
                        for s in systems
                    ]
                ).reshape(-1, 3, 1)
                block = TensorBlock(
                    values=jnp.concatenate([sums, 2 * sums], axis=2),
                    samples=Labels.range("system", len(systems)),
                    components=[Labels.range("direction", 3)],
                    properties=Labels.range("p", 2),
                )

            results.append(TensorMap(Labels.range("_", 1), [block]))

        return results


def make_system(positions, cell, pairs, vectors_offset=0.0, length_unit="angstrom"):
    system = System(
        length_unit,
        np.array([1, 6, 8], dtype=np.int32),
        np.array(positions, dtype=np.float32),
        np.array(cell, dtype=np.float32),
        np.array([np.any(c != 0.0) for c in np.array(cell)]),
    )

    pairs = np.array(pairs, dtype=np.int32)
    positions = np.array(positions, dtype=np.float32)
    cell = np.array(cell, dtype=np.float32)
    vectors = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    vectors += pairs[:, 2:].astype(np.float32) @ cell + vectors_offset

    block = TensorBlock(
        values=vectors.reshape(-1, 3, 1),
        samples=Labels(
            [
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            pairs,
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )
    system.add_pairs(PAIRS_OPTIONS, block)

    custom = TensorBlock(
        values=np.zeros((3, 1), dtype=np.float32),
        samples=Labels(["system", "atom"], np.array([[0, 0], [0, 1], [0, 2]])),
        components=[],
        properties=Labels.range("_", 1),
    )
    system.add_custom_data("test::input", TensorMap(Labels.range("_", 1), [custom]))

    return system, vectors


POSITIONS = [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.2, 1.5, 0.3]]
CELL = [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]]
PAIRS = [[0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [1, 2, 0, 0, 0], [0, 1, 1, 0, 0]]
ENERGY = Quantity(
    name="energy", unit="eV", sample_kind="system", gradients=["positions", "strain"]
)


def expected_position_gradients(pairs, vectors):
    expected = np.zeros((3, 3))
    for (i, j, *_), vector in zip(pairs, vectors, strict=True):
        expected[i] -= 2.0 * vector
        expected[j] += 2.0 * vector
    return expected


def test_requested_pair_lists():
    model = PairModel()
    model.nested = [NestedModel()]

    requested = model.requested_pair_lists(model.capabilities().outputs)
    assert requested == [
        PairListOptions(cutoff=5.0, full_list=False, strict=True),
        PairListOptions(cutoff=3.0, full_list=True, strict=True),
    ]
    assert requested[0].requestors == ["PairModel", "PairModel.pairs"]
    assert requested[1].requestors == ["PairModel.nested.0"]


def test_requested_inputs():
    model = PairModel()
    assert model.requested_inputs(model.capabilities().outputs) == [
        Quantity(name="test::input", unit="eV", sample_kind="atom")
    ]

    class OtherInput:
        def requested_inputs(self, requested_outputs):
            return [Quantity(name="test::input", unit="kcal/mol", sample_kind="atom")]

    model.other = OtherInput()
    message = "requested with different unit"
    with pytest.raises(ValueError, match=message):
        model.requested_inputs(model.capabilities().outputs)


def test_capabilities():
    model = PairModel()
    gradients = [o.gradients for o in model.capabilities().outputs]
    assert gradients == [["positions", "strain"], ["positions", "strain"], [], []]
    assert all(o.gradients == [] for o in model._raw_capabilities().outputs)


def test_autograd():
    system, vectors = make_system(POSITIONS, CELL, PAIRS)
    (result,) = execute_model(PairModel(), [system], [ENERGY], check_consistency=True)

    block = result.block()
    np.testing.assert_allclose(block.values, [[(vectors**2).sum()]], rtol=1e-6)

    gradient = block.gradient("positions")
    assert gradient.samples == Labels(
        ["sample", "system", "atom"], np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])
    )
    np.testing.assert_allclose(
        gradient.values.reshape(3, 3),
        expected_position_gradients(PAIRS, vectors),
        rtol=1e-6,
        atol=1e-6,
    )

    gradient = block.gradient("strain")
    assert gradient.samples == Labels(["sample"], np.array([[0]]))
    np.testing.assert_allclose(
        gradient.values.reshape(3, 3),
        2.0 * np.einsum("pa,pb->ab", vectors, vectors),
        rtol=1e-6,
        atol=1e-6,
    )

    request = Quantity(
        name="energy", unit="eV", sample_kind="system", gradients=["positions"]
    )
    (result,) = execute_model(PairModel(), [system], [request], check_consistency=True)
    assert result.block().gradients_list() == ["positions"]


def test_autograd_components():
    system, vectors = make_system(POSITIONS, CELL, PAIRS)
    request = Quantity(
        name="test::vector",
        unit="",
        sample_kind="system",
        gradients=["positions", "strain"],
    )
    energy, result = execute_model(
        PairModel(), [system], [ENERGY, request], check_consistency=True
    )
    assert energy.block().gradients_list() == ["positions", "strain"]

    block = result.block()
    vector_sum = vectors.sum(axis=0)
    np.testing.assert_allclose(
        block.values, [np.stack([vector_sum, 2 * vector_sum], axis=1)], rtol=1e-6
    )

    expected_positions = np.zeros((3, 3, 3, 2))
    for i, j, *_ in PAIRS:
        for factor, p in [(1.0, 0), (2.0, 1)]:
            expected_positions[i, :, :, p] -= factor * np.eye(3)
            expected_positions[j, :, :, p] += factor * np.eye(3)

    gradient = block.gradient("positions")
    assert gradient.components == [Labels.range("xyz", 3), Labels.range("direction", 3)]
    np.testing.assert_allclose(gradient.values, expected_positions, atol=1e-6)

    expected_strain = np.zeros((1, 3, 3, 3, 2))
    for factor, p in [(1.0, 0), (2.0, 1)]:
        expected_strain[0, ..., p] = factor * np.einsum(
            "a,bc->abc", vector_sum, np.eye(3)
        )
    np.testing.assert_allclose(
        block.gradient("strain").values, expected_strain, rtol=1e-6, atol=1e-6
    )


def test_several_systems():
    system_1, vectors_1 = make_system(POSITIONS, CELL, PAIRS)
    system_2, vectors_2 = make_system(POSITIONS, CELL, PAIRS[:2], vectors_offset=0.1)
    (result,) = execute_model(
        PairModel(), [system_1, system_2], [ENERGY], check_consistency=True
    )

    block = result.block()
    np.testing.assert_allclose(
        block.values.ravel(), [(vectors_1**2).sum(), (vectors_2**2).sum()], rtol=1e-6
    )
    gradient = block.gradient("positions").values.reshape(2, 3, 3)
    np.testing.assert_allclose(
        gradient[1], expected_position_gradients(PAIRS[:2], vectors_2), atol=1e-6
    )


def test_reuse_pair_vectors():
    cell = np.zeros((3, 3))
    pairs = PAIRS[:3]
    system, vectors = make_system(POSITIONS, cell, pairs, vectors_offset=0.25)
    request = Quantity(
        name="energy", unit="eV", sample_kind="system", gradients=["positions"]
    )
    (result,) = execute_model(PairModel(), [system], [request], check_consistency=True)

    block = result.block()
    np.testing.assert_allclose(block.values, [[(vectors**2).sum()]], rtol=1e-6)
    np.testing.assert_allclose(
        block.gradient("positions").values.reshape(3, 3),
        expected_position_gradients(pairs, vectors),
        rtol=1e-6,
        atol=1e-6,
    )


def test_explicit_gradients():
    class ExplicitGradients(PairModel):
        def capabilities(self):
            capabilities = super().capabilities()
            outputs = capabilities.outputs
            outputs[0].gradients = ["positions"]
            capabilities.outputs = outputs
            return capabilities

        def forward(self, systems, selected_atoms, outputs):
            (request,) = outputs
            assert request.gradients == ["positions"]

            n_atoms = len(systems[0])
            vectors = systems[0].pairs(PAIRS_OPTIONS).values.reshape(-1, 3)
            block = TensorBlock(
                values=(vectors**2).sum().reshape(1, 1),
                samples=Labels.range("system", 1),
                components=[],
                properties=Labels.range("energy", 1),
            )
            block.add_gradient(
                "positions",
                TensorBlock(
                    values=jnp.full((n_atoms, 3, 1), 42.0),
                    samples=Labels(
                        ["sample", "system", "atom"],
                        np.array([[0, 0, i] for i in range(n_atoms)]),
                    ),
                    components=[Labels.range("xyz", 3)],
                    properties=Labels.range("energy", 1),
                ),
            )
            return [TensorMap(Labels.range("_", 1), [block])]

    model = ExplicitGradients()
    assert model.capabilities().outputs[0].gradients == ["positions", "strain"]

    system, vectors = make_system(POSITIONS, np.zeros((3, 3)), PAIRS[:3])
    (result,) = execute_model(model, [system], [ENERGY], check_consistency=True)

    block = result.block()
    np.testing.assert_allclose(block.gradient("positions").values, 42.0)
    np.testing.assert_allclose(
        block.gradient("strain").values.reshape(3, 3),
        2.0 * np.einsum("pa,pb->ab", vectors, vectors),
        rtol=1e-6,
        atol=1e-6,
    )


def test_super_calls():
    class Base(PairModel):
        def requested_pair_lists(self, requested_outputs):
            return [PairListOptions(cutoff=2.0, full_list=True, strict=True)]

    class Derived(Base):
        def capabilities(self):
            capabilities = super().capabilities()
            assert capabilities.outputs[0].gradients == []
            return capabilities

        def requested_pair_lists(self, requested_outputs):
            requested = super().requested_pair_lists(requested_outputs)
            assert requested == [
                PairListOptions(cutoff=2.0, full_list=True, strict=True)
            ]
            return requested + [
                PairListOptions(cutoff=1.0, full_list=True, strict=True)
            ]

    model = Derived()
    assert model.capabilities().outputs[0].gradients == ["positions", "strain"]
    assert model.requested_pair_lists(model.capabilities().outputs) == [
        PairListOptions(cutoff=2.0, full_list=True, strict=True),
        PairListOptions(cutoff=1.0, full_list=True, strict=True),
        PairListOptions(cutoff=5.0, full_list=False, strict=True),
    ]


def test_custom_data_and_no_gradients():
    class CustomData(PairModel):
        def forward(self, systems, selected_atoms, outputs):
            data = systems[0].custom_data("test::input").block()
            assert isinstance(data.values, jax.Array)
            return [
                TensorMap(
                    Labels.range("_", 1),
                    [
                        TensorBlock(
                            values=data.values.sum().reshape(1, 1) + 1.0,
                            samples=Labels.range("system", 1),
                            components=[],
                            properties=Labels.range("energy", 1),
                        )
                    ],
                )
            ]

    system, _ = make_system(POSITIONS, CELL, PAIRS)
    request = Quantity(name="energy", unit="eV", sample_kind="system")
    (result,) = execute_model(CustomData(), [system], [request], check_consistency=True)
    np.testing.assert_allclose(result.block().values, [[1.0]])


def test_float64_requires_x64():
    class Float64(PairModel):
        def capabilities(self):
            capabilities = super().capabilities()
            capabilities.dtype = "float64"
            return capabilities

    system, _ = make_system(POSITIONS, CELL, PAIRS)
    message = "jax_enable_x64"
    with pytest.raises(ValueError, match=message):
        Float64().execute_inner([system], None, [ENERGY])


def jax_system(positions, cell, pairs, length_unit="angstrom"):
    """the same system as ``make_system``, as a JaxSystem with computed pair vectors"""
    pairs = np.array(pairs, dtype=np.int32)
    names = [
        "first_atom",
        "second_atom",
        "cell_shift_a",
        "cell_shift_b",
        "cell_shift_c",
    ]
    custom = TensorBlock(
        values=jnp.zeros((3, 1), dtype=jnp.float32),
        samples=Labels(["system", "atom"], np.array([[0, 0], [0, 1], [0, 2]])),
        components=[],
        properties=Labels.range("_", 1),
    )
    return JaxSystem(
        [1, 6, 8],
        jnp.asarray(positions, dtype=jnp.float32),
        jnp.asarray(cell, dtype=jnp.float32),
        [np.any(c != 0.0) for c in np.array(cell)],
        length_unit=length_unit,
        pairs={PAIRS_OPTIONS: Labels(names, pairs)},
        custom_data={"test::input": TensorMap(Labels.range("_", 1), [custom])},
    )


def test_jax_system_pytree():
    system, vectors = make_system(POSITIONS, CELL, PAIRS)
    jax_system = JaxSystem.from_system(system)

    # positions, cell, pair vectors and custom data are leaves
    leaves, treedef = jax.tree.flatten(jax_system)
    assert len(leaves) == 4
    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert rebuilt.known_pairs() == [PAIRS_OPTIONS]
    assert rebuilt.known_custom_data() == ["test::input"]
    np.testing.assert_equal(rebuilt.types, [1, 6, 8])
    np.testing.assert_allclose(
        rebuilt.pairs(PAIRS_OPTIONS).values.reshape(-1, 3), vectors
    )

    out = jax.jit(lambda s: s.replace(positions=s.positions + 1.0))(jax_system)
    assert isinstance(out, JaxSystem)
    np.testing.assert_allclose(out.positions, np.array(POSITIONS) + 1.0)


def test_evaluate():
    model = PairModel()
    system, _ = make_system(POSITIONS, CELL, PAIRS)
    (expected,) = execute_model(model, [system], [ENERGY], check_consistency=True)

    def evaluate(systems):
        return model.evaluate(systems, [ENERGY])

    for function in [evaluate, jax.jit(evaluate)]:
        (result,) = function([jax_system(POSITIONS, CELL, PAIRS)])
        assert isinstance(result, TensorMap)
        assert isinstance(result.block().values, jax.Array)
        assert mts.equal_metadata(result, expected)
        assert mts.allclose(
            result.to(array_backend="numpy"),
            expected.to(array_backend="numpy"),
            rtol=1e-5,
            atol=1e-5,
        )


def test_evaluate_units():
    # systems in nm, outputs in kcal/mol, both through execute_model and evaluate
    request = Quantity(
        name="energy",
        unit="kcal/mol",
        sample_kind="system",
        gradients=["positions", "strain"],
    )
    positions, cell = 0.1 * np.array(POSITIONS), 0.1 * np.array(CELL)
    system, _ = make_system(positions, cell, PAIRS, length_unit="nm")
    (expected,) = execute_model(PairModel(), [system], [request])

    nm = jax_system(positions, cell, PAIRS, length_unit="nm")
    (result,) = PairModel().evaluate([nm], [request])
    assert mts.allclose(
        result.to(array_backend="numpy"),
        expected.to(array_backend="numpy"),
        rtol=1e-4,
        atol=1e-4,
    )

    (reference,) = PairModel().evaluate([jax_system(POSITIONS, CELL, PAIRS)], [ENERGY])
    factor = unit_conversion_factor("eV", "kcal/mol")
    np.testing.assert_allclose(
        result.block().gradient("positions").values,
        10 * factor * reference.block().gradient("positions").values,
        rtol=1e-4,
    )


def test_evaluate_transformations():
    model = PairModel()
    system = jax_system(POSITIONS, CELL, PAIRS)
    request = Quantity(name="energy", unit="eV", sample_kind="system")

    def energy(positions):
        (result,) = model.evaluate([system.replace(positions=positions)], [request])
        return result.block().values[0, 0]

    (expected,) = model.evaluate([system], [ENERGY])
    np.testing.assert_allclose(
        jax.jit(jax.grad(energy))(system.positions),
        expected.block().gradient("positions").values.reshape(3, 3),
        rtol=1e-5,
        atol=1e-5,
    )

    batch = jnp.stack([system.positions * (1 + 0.1 * k) for k in range(3)])
    np.testing.assert_allclose(
        jax.vmap(energy)(batch), [energy(p) for p in batch], rtol=1e-6
    )


def test_evaluate_errors():
    system = jax_system(POSITIONS, CELL, PAIRS)
    message = "can not compute 'test::missing'"
    with pytest.raises(ValueError, match=message):
        PairModel().evaluate(
            [system], [Quantity(name="test::missing", unit="", sample_kind="system")]
        )


# Pair lists given as JaxPairs, computed inside JAX transformations

# `ctypes_dlpack`, which metatomic uses to share arrays, sets these when it is
# imported; loading vesin sets them to something else for the whole process
PYCAPSULE_NEW_ARGTYPES = ctypes.pythonapi.PyCapsule_New.argtypes

TRACED_OPTIONS = PairListOptions(cutoff=2.5, full_list=True, strict=True)
TRACED_CELL = 6.0 * np.eye(3)
IMAGES = jnp.array(list(itertools.product([-1, 0, 1], repeat=3)), dtype=jnp.int32)


def traced_pairs(positions, cell, padded_to, cutoff=TRACED_OPTIONS.cutoff):
    """Full pair list with the layout of ``vesin_jax.NeighborList``: every atom gets
    ``padded_to`` entries, its neighbours first and ``(i, -1)`` padding after. Computed
    by brute force over the neighbouring cells, for positions inside the cell and a
    cutoff smaller than the cell."""
    n_atoms = positions.shape[0]
    grid = np.meshgrid(np.arange(n_atoms), np.arange(len(IMAGES)), indexing="ij")
    j, s = (jnp.asarray(a.ravel()) for a in grid)

    def neighbors(i, position):
        vectors = positions[j] - position + IMAGES[s].astype(positions.dtype) @ cell
        self_pair = (j == i) & jnp.all(IMAGES[s] == 0, axis=1)
        valid = (jnp.sum(vectors**2, axis=1) < cutoff**2) & ~self_pair
        # `-1` marks the entries beyond the neighbours of this atom
        (index,) = jnp.nonzero(valid, size=padded_to, fill_value=-1)
        keep = index >= 0
        return (
            jnp.where(keep, j[index], -1),
            jnp.where(keep[:, None], IMAGES[s[index]], 0),
        )

    second, shifts = jax.vmap(neighbors)(jnp.arange(n_atoms), positions)
    first = jnp.repeat(jnp.arange(n_atoms), padded_to)
    return JaxPairs(first, second.reshape(-1), shifts.reshape(-1, 3))


def static_pairs(pairs):
    """the pairs in ``JaxPairs``, as the samples of a pair block"""
    mask = np.asarray(pairs.mask)
    values = np.column_stack(
        [
            np.asarray(pairs.first)[mask],
            np.asarray(pairs.second)[mask],
            np.asarray(pairs.shifts)[mask],
        ]
    )
    names = [
        "first_atom",
        "second_atom",
        "cell_shift_a",
        "cell_shift_b",
        "cell_shift_c",
    ]
    return Labels(names, values.astype(np.int32))


def fixed_size_pairs(pairs, size):
    """``pairs`` with the layout of ``vesin_jax.NeighborList(fixed_n_pairs=size)``: the
    pairs first, then ``(-1, -1)`` padding, and only padding if they do not fit; with
    the number of pairs as ``n_needed``"""
    order = jnp.argsort((~pairs.mask).astype(jnp.int32), stable=True)[:size]
    keep = (jnp.arange(size) < pairs.n_pairs) & (pairs.n_pairs <= size)
    return JaxPairs(
        jnp.where(keep, pairs.first[order], -1),
        jnp.where(keep, pairs.second[order], -1),
        jnp.where(keep[:, None], pairs.shifts[order], 0),
        pairs.n_pairs,
    )


def random_positions(seed, n_atoms=10):
    return np.random.default_rng(seed).uniform(0.0, 6.0, (n_atoms, 3))


def traced_system(positions, padded_to=16):
    positions = jnp.asarray(positions, dtype=jnp.float32)
    cell = jnp.asarray(TRACED_CELL, dtype=jnp.float32)
    return JaxSystem(
        [1] * len(positions),
        positions,
        cell,
        [True, True, True],
        pairs={TRACED_OPTIONS: traced_pairs(positions, cell, padded_to)},
    )


class TracedPairModel(JaxModel):
    """E = sum_i w_i * sum_j |r_ij|^2, using the pair indices, with w_i = i + 1 by
    default"""

    def __init__(self, weights=None):
        self.weights = weights

    def capabilities(self):
        return ModelCapabilities(
            atomic_types=[1],
            interaction_range=TRACED_OPTIONS.cutoff,
            length_unit="angstrom",
            supported_devices=["cpu"],
            dtype="float32",
            outputs=[Quantity(name="energy", unit="eV", sample_kind="system")],
        )

    def metadata(self):
        return ModelMetadata(name="traced pairs")

    def requested_pair_lists(self, requested_outputs):
        return [TRACED_OPTIONS]

    def forward(self, systems, selected_atoms, outputs):
        energies = []
        for system in systems:
            pairs, vectors = system.pair_arrays(TRACED_OPTIONS)
            # the padding has zero vectors, so it adds nothing to its segment
            per_atom = jax.ops.segment_sum(
                jnp.sum(vectors**2, axis=1), pairs.first, num_segments=len(system)
            )
            weights = self.weights
            if weights is None:
                weights = jnp.arange(1, len(system) + 1)
            energies.append(jnp.sum(per_atom * weights))

        block = TensorBlock(
            values=jnp.stack(energies).reshape(-1, 1),
            samples=Labels.range("system", len(systems)),
            components=[],
            properties=Labels.range("energy", 1),
        )
        return [TensorMap(Labels.range("_", 1), [block])]


def test_traced_pairs():
    model = TracedPairModel()

    @jax.jit
    def evaluate(positions):
        return model.evaluate([traced_system(positions)], [ENERGY])[0]

    positions = random_positions(0)
    result = evaluate(positions)

    system = traced_system(positions)
    pairs, _ = system.pair_arrays(TRACED_OPTIONS)
    assert 0 < pairs.n_pairs < pairs.capacity
    static = system.replace(pairs={TRACED_OPTIONS: static_pairs(pairs)})
    (expected,) = model.evaluate([static], [ENERGY])

    # same energy, forces and stress as with static pairs, including the pair term
    assert mts.allclose(
        result.to(array_backend="numpy"),
        expected.to(array_backend="numpy"),
        rtol=1e-5,
        atol=1e-4,
    )
    assert np.abs(result.block().gradient("positions").values).max() > 1.0

    message = "use JaxSystem.pair_arrays"
    with pytest.raises(ValueError, match=message):
        system.pairs(TRACED_OPTIONS)


def test_traced_pairs_given_vectors():
    system = traced_system(random_positions(1))
    pairs, vectors = system.pair_arrays(TRACED_OPTIONS)
    shifted = system.replace(pairs={TRACED_OPTIONS: (pairs, vectors + 0.5)})

    # the values of the given vectors, with the derivatives of the pairs
    _, given = shifted.pair_arrays(TRACED_OPTIONS)
    np.testing.assert_allclose(given, jnp.where(pairs.mask[:, None], vectors + 0.5, 0))

    weights = jnp.asarray(np.random.default_rng(2).normal(size=vectors.shape))

    def linear(system, positions):
        _, vectors = system.replace(positions=positions).pair_arrays(TRACED_OPTIONS)
        return jnp.sum(weights * vectors)

    np.testing.assert_allclose(
        jax.grad(linear, argnums=1)(shifted, system.positions),
        jax.grad(linear, argnums=1)(system, system.positions),
        rtol=1e-6,
    )


def test_traced_pairs_no_recompilation():
    model = TracedPairModel()
    traces = []

    @jax.jit
    def forces(positions):
        traces.append(positions)
        system = traced_system(positions)
        (result,) = model.evaluate([system], [ENERGY])
        n_pairs = system.pair_arrays(TRACED_OPTIONS)[0].n_pairs
        return n_pairs, -result.block().gradient("positions").values

    n_pairs = {int(forces(random_positions(seed))[0]) for seed in range(4)}
    assert len(n_pairs) > 1
    assert len(traces) == 1


def test_traced_pairs_padding():
    """the padding of vesin sits between the pairs, not only at the end of the list"""
    system = traced_system(random_positions(0))
    pairs, vectors = system.pair_arrays(TRACED_OPTIONS)
    mask = np.asarray(pairs.mask)

    assert not mask.all()
    # a list where the padding only came last would pass the test above
    assert mask[np.argmin(mask) :].any()
    np.testing.assert_equal(np.asarray(pairs.second)[~mask], -1)
    np.testing.assert_equal(np.asarray(pairs.shifts)[~mask], 0)
    np.testing.assert_equal(np.asarray(vectors)[~mask], 0.0)


def test_traced_pairs_from_vesin():
    """the output of vesin_jax goes into JaxPairs as it comes"""
    vesin_jax = pytest.importorskip("vesin_jax")

    positions = jnp.asarray(random_positions(0), dtype=jnp.float32)
    cell = jnp.asarray(TRACED_CELL, dtype=jnp.float32)
    # loading vesin changes the argtypes of `ctypes.pythonapi.PyCapsule_New` for
    # the whole process, which breaks the DLPack conversions `ctypes_dlpack` does
    # afterwards; keep that to this test
    try:
        calculator = vesin_jax.NeighborList(
            cutoff=TRACED_OPTIONS.cutoff,
            full_list=TRACED_OPTIONS.full_list,
            padded_to=64,
        )
        vesin_pairs = JaxPairs(*calculator.compute(positions, cell, True, "ijS"))
        (reference,) = calculator.compute(positions, cell, True, "D")
    finally:
        ctypes.pythonapi.PyCapsule_New.argtypes = PYCAPSULE_NEW_ARGTYPES

    system = JaxSystem(
        [1] * len(positions),
        positions,
        cell,
        [True, True, True],
        pairs={TRACED_OPTIONS: vesin_pairs},
    )
    pairs, vectors = system.pair_arrays(TRACED_OPTIONS)

    # vesin pads the pairs of every atom, so the padding is spread through the
    # list instead of sitting at its end
    mask = np.asarray(pairs.mask)
    assert mask.any() and not mask.all()
    assert mask[np.argmin(mask) :].any()
    np.testing.assert_equal(np.asarray(pairs.second)[~mask], -1)

    # the pair vectors are the ones vesin computes itself, padding included
    np.testing.assert_allclose(
        np.asarray(vectors), np.asarray(reference), rtol=1e-5, atol=1e-5
    )

    # and the same pairs as the samples of a pair block, without the padding
    assert int(pairs.n_pairs) == len(static_pairs(pairs))


def evaluate_traced(system):
    (result,) = TracedPairModel().evaluate([system], [ENERGY])
    return result.block().values, result.block().gradient("positions").values


def test_traced_pairs_fixed_size():
    """pairs first and padding after, with a negative first atom as well"""
    system = traced_system(random_positions(0))
    padded, _ = system.pair_arrays(TRACED_OPTIONS)
    pairs = fixed_size_pairs(padded, 64)

    n_pairs = int(padded.n_pairs)
    assert int(pairs.n_pairs) == int(pairs.n_needed) == n_pairs
    assert not pairs.overflow
    np.testing.assert_equal(np.asarray(pairs.first)[n_pairs:], -1)

    fixed = system.replace(pairs={TRACED_OPTIONS: pairs})
    for value, expected in zip(
        jax.jit(evaluate_traced)(fixed), evaluate_traced(system), strict=True
    ):
        np.testing.assert_allclose(value, expected, rtol=1e-5, atol=1e-4)


def test_traced_pairs_overflow():
    """pairs which did not fit in the list give NaN, and not results computed without
    any pair"""
    systems = [traced_system(random_positions(seed)) for seed in range(2)]
    n_pairs = [int(system.pair_arrays(TRACED_OPTIONS)[0].n_pairs) for system in systems]
    assert n_pairs[0] != n_pairs[1]
    # the pairs of one system fit in the list, not those of the other
    size = sum(n_pairs) // 2
    fixed = [
        system.replace(
            pairs={
                TRACED_OPTIONS: fixed_size_pairs(
                    system.pair_arrays(TRACED_OPTIONS)[0], size
                )
            }
        )
        for system in systems
    ]
    batch = jax.tree.map(lambda *arrays: jnp.stack(arrays), *fixed)

    overflow = jax.vmap(lambda s: s.pair_arrays(TRACED_OPTIONS)[0].overflow)(batch)
    np.testing.assert_equal(np.asarray(overflow), np.array(n_pairs) > size)

    energies, gradients = jax.jit(jax.vmap(evaluate_traced))(batch)
    for system, energy, gradient, too_many in zip(
        systems, energies, gradients, overflow, strict=True
    ):
        if too_many:
            assert np.isnan(energy).all() and np.isnan(gradient).all()
        else:
            expected_energy, expected_gradient = evaluate_traced(system)
            np.testing.assert_allclose(energy, expected_energy, rtol=1e-5)
            np.testing.assert_allclose(
                gradient, expected_gradient, rtol=1e-5, atol=1e-4
            )

    # without `n_needed`, nothing says the list is missing the pairs: it is only
    # padding, and the energy is zero
    system = fixed[int(np.argmax(overflow))]
    pairs, _ = system.pair_arrays(TRACED_OPTIONS)
    unchecked = JaxPairs(pairs.first, pairs.second, pairs.shifts)
    assert not unchecked.overflow
    energy, _ = evaluate_traced(system.replace(pairs={TRACED_OPTIONS: unchecked}))
    np.testing.assert_equal(np.asarray(energy), [[0.0]])


def test_traced_pairs_from_vesin_fixed_size():
    """vesin_jax with `fixed_n_pairs`, and the `"n"` output for lists too small"""
    vesin_jax = pytest.importorskip("vesin_jax")

    positions = jnp.asarray(random_positions(0), dtype=jnp.float32)
    cell = jnp.asarray(TRACED_CELL, dtype=jnp.float32)
    try:
        lists = [
            JaxPairs(
                *vesin_jax.NeighborList(
                    cutoff=TRACED_OPTIONS.cutoff,
                    full_list=TRACED_OPTIONS.full_list,
                    fixed_n_pairs=size,
                ).compute(positions, cell, True, "ijSn")
            )
            for size in (64, 8)
        ]
    finally:
        ctypes.pythonapi.PyCapsule_New.argtypes = PYCAPSULE_NEW_ARGTYPES

    system = traced_system(positions)
    large, small = (system.replace(pairs={TRACED_OPTIONS: pairs}) for pairs in lists)
    assert not lists[0].overflow and lists[1].overflow
    assert int(lists[1].n_needed) == int(lists[0].n_pairs) > 8

    for value, expected in zip(
        jax.jit(evaluate_traced)(large), evaluate_traced(system), strict=True
    ):
        np.testing.assert_allclose(value, expected, rtol=1e-5, atol=1e-4)
    assert all(np.isnan(value).all() for value in evaluate_traced(small))


def test_traced_pairs_vmap():
    model = TracedPairModel()
    systems = [traced_system(random_positions(seed)) for seed in range(3)]
    # positions, cell and the three arrays of JaxPairs
    assert len(jax.tree.leaves(systems[0])) == 5

    # systems with different pairs, stacked and evaluated together
    batch = jax.tree.map(lambda *arrays: jnp.stack(arrays), *systems)

    def evaluate(system):
        (result,) = model.evaluate([system], [ENERGY])
        return result.block().values, result.block().gradient("positions").values

    energies, gradients = jax.jit(jax.vmap(evaluate))(batch)
    for system, energy, gradient in zip(systems, energies, gradients, strict=True):
        expected_energy, expected_gradient = evaluate(system)
        np.testing.assert_allclose(energy, expected_energy, rtol=1e-5)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-4)


def test_traced_pairs_second_derivatives():
    system = traced_system(random_positions(3))
    pairs, _ = system.pair_arrays(TRACED_OPTIONS)
    static = system.replace(pairs={TRACED_OPTIONS: static_pairs(pairs)})

    # a loss on the forces, differentiated with respect to the model parameters (as
    # when fitting to forces or electric fields) goes through the second derivatives
    # of the pair vectors
    rng = np.random.default_rng(4)
    target = jnp.asarray(rng.normal(size=(len(system), 3)), dtype=jnp.float32)

    def loss(weights, system):
        (result,) = TracedPairModel(weights).evaluate([system], [ENERGY])
        forces = -result.block().gradient("positions").values.reshape(-1, 3)
        return jnp.sum((forces - target) ** 2)

    weights = jnp.linspace(0.5, 1.5, len(system), dtype=jnp.float32)
    traced = jax.jit(jax.grad(loss))(weights, system)
    np.testing.assert_allclose(traced, jax.grad(loss)(weights, static), rtol=1e-4)
    assert np.abs(traced).max() > 1.0

    # the hessian with respect to the positions
    request = Quantity(name="energy", unit="eV", sample_kind="system")

    def energy(positions, system):
        system = system.replace(positions=positions)
        return TracedPairModel().evaluate([system], [request])[0].block().values[0, 0]

    hessian = jax.jit(jax.hessian(energy))(system.positions, system)
    expected = jax.hessian(energy)(system.positions, static)
    np.testing.assert_allclose(hessian, expected, rtol=1e-4, atol=1e-4)
    assert np.abs(hessian).max() > 1.0
