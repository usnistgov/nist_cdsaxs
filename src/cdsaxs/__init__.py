"""Public numerical helpers, loaded on use so lifecycle commands stay lightweight."""
from importlib import import_module

__all__ = ["filter_filenames", "LoadData", "LoadDataset", "LoadDataset_MetadataCSV",
           "slice_reduced_dataset"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = ".reduction" if name == "slice_reduced_dataset" else ".loaders.load_data"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
