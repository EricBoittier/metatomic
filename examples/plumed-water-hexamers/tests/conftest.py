"""Use the pinned checkout's bindings and the same native libraries as PLUMED."""

import ctypes
import os
import sys
import types
from pathlib import Path


EXAMPLE = Path(__file__).resolve().parents[1]
CORE = EXAMPLE.parents[1]
WORK = CORE.parent
DEPS = Path(
    os.environ.get(
        "HEXAMER_DEPS",
        sys.prefix,
    )
)
KERNEL = Path(
    os.environ.get("PLUMED_TEST_KERNEL", WORK / "plumed2/src/lib/libplumedKernel.dylib")
)
if not KERNEL.is_file():
    raise RuntimeError(f"Build the core-enabled PLUMED kernel first: {KERNEL}")
# Route the bindings through their supported external-library configuration.
# A pip metatensor wheel otherwise loads a second library with a separate registry.
for module_name, symbol, path in [
    (
        "metatensor._external",
        "EXTERNAL_METATENSOR_PATH",
        DEPS / "lib/libmetatensor.dylib",
    ),
    (
        "metatomic._external",
        "EXTERNAL_METATOMIC_PATH",
        WORK / "install/lib/libmetatomic.dylib",
    ),
]:
    configuration = types.ModuleType(module_name)
    setattr(configuration, symbol, str(path))
    sys.modules[module_name] = configuration
# Keep the same native libraries alive for the entire pytest session.
_NATIVE = [
    ctypes.CDLL(str(DEPS / "lib/libmetatensor.dylib"), mode=ctypes.RTLD_GLOBAL),
    ctypes.CDLL(str(WORK / "install/lib/libmetatomic.dylib"), mode=ctypes.RTLD_GLOBAL),
]
sys.path.insert(0, str(CORE / "python/metatomic_core/src"))
sys.path.insert(0, str(Path(__file__).parent))
os.environ["PLUMED_KERNEL"] = str(KERNEL)
