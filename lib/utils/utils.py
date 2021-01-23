import logging
import time
import os

import torch
from lib.utils.lr_scheduler import WarmupMultiStepLR
from lib.optimizer.adam_series import RAdam, PlainRAdam, AdamW


def create_logger(cfg):
    dataset = cfg.DATASET
    net_type = cfg.BACKBONE
    # module_type = cfg.MODULE.TYPE
    log_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_name = "{}_{}_{}.log".format(dataset, net_type, time_str)
    log_file = os.path.join(log_dir, log_name)
    # set up logger
    print("=> creating log {}".format(log_file))
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    logger.info("---------------------Cfg is set as follow--------------------")
    logger.info(cfg)
    logger.info("-------------------------------------------------------------")
    return logger, log_file


def get_optimizer(cfg, model):
    base_lr = cfg.BASE_LR
    params = []

    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append({"params": p})
    # params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = {
        "ADAM": lambda: torch.optim.Adam(params,
                                         lr=base_lr,
                                         betas=(0.9, 0.999),
                                         weight_decay=cfg.WEIGHT_DECAY,
                                         eps=1e-08),

        "RMSprop": lambda: torch.optim.RMSprop(params,
                                               lr=base_lr,
                                               momentum=cfg.MOMENTUM,
                                               eps=1e-08,
                                               weight_decay=cfg.WEIGHT_DECAY),  # 2e-4

        "SGD": lambda: torch.optim.SGD(params,
                                       lr=base_lr,
                                       momentum=cfg.MOMENTUM,
                                       weight_decay=cfg.WEIGHT_DECAY,
                                       nesterov=True, ),

        "Radam": lambda: RAdam(params,
                               lr=base_lr,
                               weight_decay=cfg.WEIGHT_DECAY),

        "PlainRAdam": lambda: PlainRAdam(params,
                                         lr=base_lr,
                                         weight_decay=cfg.WEIGHT_DECAY),

        "AdamW": lambda: AdamW(params,
                               lr=base_lr,
                               weight_decay=cfg.WEIGHT_DECAY),

    }[cfg.OPTIMIZER_TYPE]()
    return optimizer


def get_scheduler(cfg, optimizer):
    if cfg.SCHEDULER_TYPE == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            cfg.LR_STEP,
            gamma=cfg.LR_FACTOR,
        )
    elif cfg.SCHEDULER_TYPE == "cosine":
        if cfg.COSINE_DECAY_END > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.COSINE_DECAY_END, eta_min=1e-4
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.MAX_EPOCH, eta_min=1e-5
            )
    elif cfg.SCHEDULER_TYPE == "warmup":
        scheduler = WarmupMultiStepLR(
            optimizer,
            cfg.LR_STEP,
            gamma=cfg.LR_FACTOR,
            warmup_epochs=cfg.WARM_EPOCH,
        )
    else:
        raise NotImplementedError("Unsupported LR Scheduler: {}".format(cfg.SCHEDULER_TYPE))

    return scheduler


def get_category_list(annotations, num_classes):
    num_list = [0] * num_classes
    cat_list = []
    print("Weight List has been produced")
    for anno in annotations:
        category_id = anno["category_id"]
        num_list[category_id] += 1
        cat_list.append(category_id)
    return num_list, cat_list
