from aldi.distill import DISTILL_MIXIN_REGISTRY

from aldi.fcos.fcos import FCOS
from aldi.fcos.fcos_torchvision import FCOSTorchvision

@DISTILL_MIXIN_REGISTRY.register()
class FCOSDistillMixin(FCOS): pass

@DISTILL_MIXIN_REGISTRY.register()
class FCOSTorchvisionDistillMixin(FCOSTorchvision): pass