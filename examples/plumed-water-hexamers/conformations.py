"""Named reference structures; XYZ comments are the single source of labels."""

from pathlib import Path

from ase.io import read


def load_conformations(path=None):
    path = Path(path) if path else Path(__file__).with_name("hexamers.xyz")
    frames = read(path, ":", format="xyz")
    lines = path.read_text().splitlines()
    names = []
    offset = 0
    while offset < len(lines):
        count = int(lines[offset])
        names.append(lines[offset + 1].split(";")[0].strip())
        offset += count + 2
    if len(names) != len(set(names)) or len(names) != len(frames):
        raise ValueError("Conformation labels must be unique and match the frames")
    for name, frame in zip(names, frames, strict=True):
        if frame.get_chemical_symbols() != ["O", "H", "H"] * 6:
            raise ValueError(f"{name}: expected six OHH water molecules")
        frame.info["conformation"] = name
    return dict(zip(names, frames, strict=True))
