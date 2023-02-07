from __future__ import annotations

import io
import pickle as pkl
from typing import Literal

import numpy as np
import pandas as pd
import torch
from loguru import logger

try:
    import msgpack
except ImportError:
    logger.warning("msgpack not installed, cannot use msgpack serialisation")
try:
    import dill
except ImportError:
    logger.warning("dill not installed, cannot use dill serialisation")
try:
    import biotite
    import biotite.structure.io.npz
except ImportError:
    logger.warning("biotite not installed, cannot serealise biotite serialisation")


def custom_serialisation(obj: object) -> object:
    """Custom serealisation methods for objects that are not serialisable by default."""
    if isinstance(obj, np.ndarray):
        with io.BytesIO() as buf:
            np.save(buf, obj)
            return buf.getvalue()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="split")
    elif isinstance(obj, torch.Tensor):
        return custom_serialisation(obj.detach().cpu().numpy())
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(
        obj, biotite.structure.AtomArray
    ):  # TODO: Switch to string comparison?
        with io.BytesIO() as buf:
            f = biotite.structure.io.npz.NpzFile()
            f.set_structure(obj)
            f.write(buf)
            return buf.getvalue()
    else:
        raise RuntimeError("Cannot serialise object of type %s" % type(obj))


def serialise(
    x: object, serialisation_format: Literal["pkl", "dill", "msgpack"]
) -> bytes:
    """
    Serialises dataset `x` in format given by `serialisation_format` (pkl, dill, msgpack).
    """
    if serialisation_format == "pkl":
        # Pickle
        # Memory efficient but brittle across languages/python versions.
        return pkl.dumps(x)
    elif serialisation_format == "dill":
        return dill.dumps(x)
    elif serialisation_format == "msgpack":
        # msgpack
        # A bit more memory efficient than json, a bit less supported.
        serialised = msgpack.packb(x, default=custom_serialisation)
    elif serialisation_format == None:
        serialised = x if isinstance(x, bytes) else str(x).encode()
    else:
        raise RuntimeError("Invalid serialization format %s" % serialisation_format)
    return serialised


def deserialise(
    x: bytes, serialisation_format: Literal["pkl", "dill", "msgpack"] | None
) -> object:
    """
    Deserialises dataset `x` assuming format given by `serialisation_format` (pkl, dill, msgpack).
    """
    if serialisation_format == "pkl":
        return pkl.loads(x)
    elif serialisation_format == "dill":
        return dill.loads(x)
    elif serialisation_format == "msgpack":
        serialised = msgpack.unpackb(x, strict_map_key=False)
    elif serialisation_format == None:
        serialised = x.decode() if isinstance(x, bytes) else x
    else:
        raise RuntimeError("Invalid serialisation format %s" % serialisation_format)
    return serialised
