from detectron2.checkpoint import DetectionCheckpointer
from fvcore.common.checkpoint import _IncompatibleKeys


def load_model_weights(path, model):
    checkpointer = DetectionCheckpointer(model)  
    ret = checkpointer.load(path)

    if path.endswith(".pth") and "ema" in ret.keys():
        # self.logger.info("Loading EMA weights as model starting point.")
        ema_dict = {
            k.replace('model.', ''): v for k, v in ret['ema'].items()
        }
        # incompatible = self.model.load_state_dict(ema_dict, strict=False)
        ret['model'] = ema_dict
        incompatible = checkpointer._load_model(ret)
        if incompatible is not None:
            checkpointer._log_incompatible_keys(_IncompatibleKeys(
                missing_keys=incompatible.missing_keys,
                unexpected_keys=incompatible.unexpected_keys,
                incorrect_shapes=[]
            ))
    return ret