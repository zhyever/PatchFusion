from .zoedepth_custom import ZoeDepthCustom
from .patchfusion import PatchFusion

all_versions = {
    "custom": ZoeDepthCustom,
    "patchfusion": PatchFusion
}

get_version = lambda v : all_versions[v]