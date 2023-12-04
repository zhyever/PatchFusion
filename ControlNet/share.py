import ControlNet.config
from ControlNet.cldm.hack import disable_verbosity, enable_sliced_attention


disable_verbosity()

if ControlNet.config.save_memory:
    enable_sliced_attention()
