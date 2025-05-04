import torch
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


def dropout_mask_along_channel(weights, p):
    if p > 0:
        mask = (torch.rand(weights.shape[0], weights.shape[1], 1, 1) > p).float().to(weights.device)
        mask = mask.expand_as(weights)
        mask = mask / (1 - p)
    else:
        mask = torch.ones((weights.shape[0], weights.shape[1], 1, 1)).float().to(weights.device)
        mask = mask.expand_as(weights)
    return mask


def dropout_masks(module, p=.1, weights_filter='res4.2.conv3.weight'):
    state_dict = module.state_dict()
    last_layer_parameters = [(k, v) for k, v in state_dict.items() if weights_filter in k]
    mask_dict = {}
    for (k, w) in last_layer_parameters:
        mask_dict[k] = dropout_mask_along_channel(w, p)
    return mask_dict


def perturb_by_dropout(module, p=.1, mask_dict={}, weights_filter='res4.2.conv3.weight'):
    state_dict = module.state_dict()
    last_layer_parameters = [(k, v) for k, v in state_dict.items() if weights_filter in k]
    for (k, w) in last_layer_parameters:
        state_dict[k] = w * mask_dict[k] if k in mask_dict else w * dropout_mask_along_channel(w, p)
    incompatiable_keys = module.load_state_dict(state_dict, strict=False)
    print(incompatiable_keys)
    return module