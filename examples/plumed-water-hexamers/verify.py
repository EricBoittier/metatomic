"""Compare actual PLUMED output to independent scalar Python calculations."""
import math
from pathlib import Path


def references():
    lines = iter(Path("hexamers.xyz").read_text().splitlines())
    for line in lines:
        count = int(line)
        name = next(lines).split(";")[0]
        atoms = [next(lines).split() for _ in range(count)]
        oxygen = [[float(x) for x in atom[1:4]] for atom in atoms if atom[0] == "O"]
        assert len(oxygen) == 6 and count == 18
        center = [sum(r[k] for r in oxygen) / 6 for k in range(3)]
        rg = math.sqrt(sum(math.dist(r, center) ** 2 for r in oxygen) / 6)
        cn = sum(2 / (6 * (1 + (math.dist(a, b) / 3.2) ** 6))
                 for i, a in enumerate(oxygen) for b in oxygen[i + 1:])
        yield name, rg, cn


if __name__ == "__main__":
    rows = [list(map(float, line.split())) for line in Path("COLVAR").read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    expected = list(references())
    assert len(rows) == len(expected)
    print("isomer     oxygen Rg / A    mean oxygen coordination")
    for row, (name, rg, cn) in zip(rows, expected):
        assert len(row) == 3, row
        assert abs(row[1] - rg) < 1e-8, (name, row[1], rg)
        assert abs(row[2] - cn) < 1e-8, (name, row[2], cn)
        print(f"{name:8s}   {row[1]:.8f}      {row[2]:.8f}")
    print("PASS: all PLUMED values agree with independent Python within 1e-8.")
