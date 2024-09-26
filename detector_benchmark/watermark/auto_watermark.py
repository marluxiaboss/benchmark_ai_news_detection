# =========================================================================
# AutoWatermark.py
# Description: This is a generic watermark class that will be instantiated
#              as one of the watermark classes of the library when created
#              with the [`AutoWatermark.load`] class method.
# =========================================================================

import importlib

from .kgw import KGW
from .sir import SIR

# from .xsir import XSIR
from .kgw_p import KGW_P
from .exp import EXP
from .dip import DIP

# from .kgw_e import KGW_E
from .sweet import SWEET
from .sweet_p import SWEET_P


WATERMARK_MAPPING_NAMES = {
    "KGW": "kgw.KGW",
    "SIR": "sir.SIR",
    "XSIR": "xsir.XSIR",
    "KGW_P": "kgw_p.KGW_P",
    "EXP": "exp.EXP",
    "DIP": "dip.DIP",
    "KGW_E": "kgw_e.KGW_E",
    "SWEET": "sweet.SWEET",
    "SWEET_P": "sweet_p.SWEET_P",
}


def watermark_name_from_alg_name(name):
    """Get the watermark class name from the algorithm name."""
    for algorithm_name, watermark_name in WATERMARK_MAPPING_NAMES.items():
        if name == algorithm_name:
            return watermark_name
    return None


class AutoWatermark:
    """
    This is a generic watermark class that will be instantiated as one of the watermark classes of the library when
    created with the [`AutoWatermark.load`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoWatermark is designed to be instantiated "
            "using the `AutoWatermark.load(algorithm_name, algorithm_config, transformers_config)` method."
        )

    def load(
        algorithm_name, algorithm_config=None, gen_model=None, model_config=None, *args, **kwargs
    ):
        """Load the watermark algorithm instance based on the algorithm name."""
        watermark_name = watermark_name_from_alg_name(algorithm_name)
        module_name, class_name = watermark_name.rsplit(".", 1)

        loader = importlib.import_module("detector_benchmark.watermark")
        module = getattr(loader, module_name)
        watermark_class = getattr(module, class_name)
        watermark_instance = watermark_class(algorithm_config, gen_model, model_config)
        return watermark_instance
