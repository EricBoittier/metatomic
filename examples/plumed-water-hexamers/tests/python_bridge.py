"""Test-only adapter to the pinned C ABI, which predates Python BaseModel.

This is NOT the production Python plugin loader. PLUMED loads models through
mta_register_plugin and the core owns returned TensorMaps. All callback pointers
are retained for the process lifetime; borrowed System views are invalidated.
"""

import ctypes
import json

from backend_models import BackendModel

from metatomic import ModelCapabilities, ModelMetadata, Quantity, System
from metatomic._c_api import mta_model_t, mta_plugin_t, mta_status_t
from metatomic._c_lib import _get_library
from metatomic._status import save_exception


MODELS = {}
_KEEPALIVE = []
_REGISTERED = False


def register():
    global _REGISTERED
    if _REGISTERED:
        return
    lib = _get_library()
    types = dict(mta_model_t._fields_)

    def protected(function):
        def call(*args):
            try:
                function(*args)
                return 0
            except BaseException as error:
                save_exception(error)
                return mta_status_t.MTA_MODEL_ERROR

        return call

    def json_callback(data):
        def callback(_, output):
            output[0] = lib.mta_string_create(json.dumps(data).encode())

        return callback

    caps = ModelCapabilities(
        atomic_types=[1, 8],
        interaction_range=100.0,
        length_unit="angstrom",
        supported_devices=["cpu"],
        dtype="float64",
        outputs=[
            Quantity(
                name="feature",
                unit="",
                sample_kind="system",
                gradients=["positions", "strain"],
            )
        ],
    ).to_dict()

    def unload(data):
        ctypes.cast(data, ctypes.py_object).value.unloads += 1

    common = {
        "unload": unload,
        "metadata": json_callback(
            ModelMetadata(name="Python backend integration test").to_dict()
        ),
        "capabilities": json_callback(caps),
        "requested_pair_lists": json_callback([]),
        "requested_inputs": json_callback([]),
    }
    callbacks = {name: types[name](protected(fn)) for name, fn in common.items()}
    _KEEPALIVE.extend(callbacks.values())

    def load(name, options, output):
        backend = name.decode()
        if backend not in ("numpy", "torch", "jax"):
            return mta_status_t.MTA_UNSUPPORTED_MODEL_ERROR
        try:
            model = BackendModel(backend)
            MODELS[backend] = model

            def execute(_, ptrs, count, selected, request_json, outputs, output_count):
                views = []
                try:
                    if selected:
                        raise ValueError(
                            "selected_atoms not supported in the test model"
                        )
                    if count != 1 or output_count != 1:
                        raise ValueError("expected one system and one output")
                    request = json.loads(request_json)[0]
                    if request["name"] != "feature":
                        raise ValueError("expected feature request")
                    view = System.unsafe_view_from_ptr(ptrs[0])
                    views.append(view)
                    result = model.evaluate(view, request.get("gradients", []))
                    outputs[0] = result.release()
                finally:
                    for view in views:
                        view._ptr = None

            callback = types["execute_inner"](protected(execute))
            _KEEPALIVE.extend([model, callback])
            native = mta_model_t()
            native.data = id(model)
            for key, value in callbacks.items():
                setattr(native, key, value)
            native.execute_inner = callback
            output[0] = native
            return 0
        except BaseException as error:
            save_exception(error)
            return mta_status_t.MTA_MODEL_ERROR

    loader = dict(mta_plugin_t._fields_)["load_model"](load)
    plugin = mta_plugin_t(
        abi_version=1, name=b"pytest-python-backends", load_model=loader
    )
    _KEEPALIVE.extend([loader, plugin])
    lib.mta_register_plugin(plugin)
    _REGISTERED = True
