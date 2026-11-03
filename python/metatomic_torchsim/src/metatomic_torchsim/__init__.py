from ._core import CoreModel


try:
    from ._model import MetatomicModel
except ImportError as e:
    # metatensor-torch and metatomic-torch are only needed for TorchScript models
    _LEGACY_IMPORT_ERROR = e

    def MetatomicModel(model, **kwargs):  # noqa: N802
        """Without metatomic-torch, only metatomic-core models are supported"""
        from ._core import _is_core_model

        if _is_core_model(model, kwargs):
            return CoreModel(model, **kwargs)
        raise ImportError(
            "metatomic-torch is required for TorchScript models in metatomic-torchsim"
        ) from _LEGACY_IMPORT_ERROR


__all__ = ["CoreModel", "MetatomicModel"]
