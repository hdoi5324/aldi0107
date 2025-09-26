from ..align import ALIGN_MIXIN_REGISTRY

from aldi.fcos.fcos import FCOS
from aldi.fcos.fcos_torchvision import FCOSTorchvision


@ALIGN_MIXIN_REGISTRY.register()
class FCOSAlignMixin(FCOS): pass

@ALIGN_MIXIN_REGISTRY.register()
class FCOSTorchvisionAlignMixin(FCOSTorchvision): pass