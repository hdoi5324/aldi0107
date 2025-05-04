from detectron2.config import get_cfg
from aldi.config import add_aldi_config
from aldi.config_aldi_only import add_aldi_only_config
from detectron2.engine import default_setup

def setup(args):
    """
    Copied from detectron2/tools/train_net.py
    """
    cfg = get_cfg()

    ## Change here
    add_aldi_config(cfg)
    add_aldi_only_config(cfg)  # adds a couple of keys as configs have diverged.
    ## End change

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg