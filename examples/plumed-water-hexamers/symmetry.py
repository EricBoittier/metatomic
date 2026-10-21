"""Exact intact-water permutation matching, with proper/improper fits separated.

The matching set is (S2)^6 semidirect S6, optionally times inversion. It is a
geometric equivalence test, not a claim that every operation is dynamically
accessible or belongs to the effective tunneling group.
"""

import itertools

import numpy as np


WATER_PERMUTATIONS = np.array(list(itertools.permutations(range(6))))
SWAPS = np.array(list(itertools.product((False, True), repeat=6)))
OPERATIONS = np.repeat(
    (3 * WATER_PERMUTATIONS[:, :, None] + np.arange(3)).reshape(720, 18), 64, axis=0
)
for water in range(6):
    selected = np.tile(SWAPS[:, water], 720)
    OPERATIONS[selected, 3 * water + 1 : 3 * water + 3] = OPERATIONS[
        selected, 3 * water + 1 : 3 * water + 3
    ][:, ::-1]


def cycles(permutation, labels=None):
    labels = labels or [str(i + 1) for i in range(len(permutation))]
    seen, result = set(), []
    for start in range(len(permutation)):
        if start in seen:
            continue
        cycle, current = [], start
        while current not in seen:
            seen.add(current)
            cycle.append(labels[current])
            current = int(permutation[current])
        if len(cycle) > 1:
            result.append("(" + " ".join(cycle) + ")")
    return "".join(result) or "identity"


def permutation_order(permutation):
    seen, lengths = set(), []
    for start in range(len(permutation)):
        if start in seen:
            continue
        current, length = start, 0
        while current not in seen:
            seen.add(current)
            length += 1
            current = int(permutation[current])
        lengths.append(length)
    return int(np.lcm.reduce(lengths))


def match(atoms, reference):
    """Map moving atom i onto reference atom atom_map[i], after a rigid fit."""
    if (
        atoms.get_chemical_symbols() != ["O", "H", "H"] * 6
        or reference.get_chemical_symbols() != ["O", "H", "H"] * 6
    ):
        raise ValueError("Expected OHH-ordered hexamers")
    p = atoms.positions - atoms.positions.mean(0)
    q = reference.positions - reference.positions.mean(0)
    candidates = q[OPERATIONS]
    u, singular, vh = np.linalg.svd(np.einsum("ia,pib->pab", p, candidates))
    det = np.linalg.det(u @ vh)
    result = {}
    for name, parity in [("proper", 1), ("improper", -1)]:
        scores = singular[:, :2].sum(1) + parity * det * singular[:, 2]
        best = int(np.argmax(scores))
        diagonal = np.array([1.0, 1.0, parity * det[best]])
        rot = (u[best] * diagonal) @ vh[best]
        atom_map = OPERATIONS[best]
        rmsd = float(np.sqrt(((p @ rot - q[atom_map]) ** 2).sum() / 18))
        waters = atom_map[::3] // 3
        result[name] = dict(
            rmsd_A=rmsd,
            atom_map_0based=atom_map.tolist(),
            water_map_0based=waters.tolist(),
            water_cycles=cycles(waters, list("ABCDEF")),
            atom_cycles=cycles(atom_map),
            permutation_order=permutation_order(atom_map),
            determinant=parity,
            rotation=rot.tolist(),
        )
    return result
