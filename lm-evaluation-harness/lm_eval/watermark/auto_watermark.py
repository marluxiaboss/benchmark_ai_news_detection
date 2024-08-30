# =========================================================================
# AutoWatermark.py
# Description: This is a generic watermark class that will be instantiated 
#              as one of the watermark classes of the library when created 
#              with the [`AutoWatermark.load`] class method.
# =========================================================================

import importlib

WATERMARK_MAPPING_NAMES={
    'KGW': 'watermark.kgw.KGW',
    'SIR': 'watermark.sir.SIR',
    'XSIR': 'watermark.xsir.XSIR',
    "KGW_P": "watermark.kgw_p.KGW_P",
    "EXP": "watermark.exp.EXP",
    "DIP": "watermark.dip.DIP",
    "KGW_E": "watermark.kgw_e.KGW_E",
    "SWEET_P": "watermark.sweet_p.SWEET_P",
    "SWEET": "watermark.sweet.SWEET",
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

    def load(algorithm_name, algorithm_config=None, gen_model=None , model_config=None, *args, **kwargs):
        """Load the watermark algorithm instance based on the algorithm name."""
        watermark_name = watermark_name_from_alg_name(algorithm_name)
        module_name, class_name = watermark_name.rsplit('.', 1)

        # TODO: figure out if we should just keep watermark as an external library
        module_name = "lm_eval." + module_name
        
        module = importlib.import_module(module_name)
        watermark_class = getattr(module, class_name)
        watermark_instance = watermark_class(algorithm_config, gen_model, model_config)
        return watermark_instance

