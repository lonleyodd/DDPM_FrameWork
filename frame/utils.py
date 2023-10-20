import os
import datetime
import logging
import coloredlogs


def set_save_path(dir):
    save_path = {}

    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_root = os.path.join(dir, file_name)
    os.makedirs(save_root, exist_ok=True)
    save_path["save_root"] = save_root

    tensorboard_path = os.path.join(save_root, "tensorboard")
    os.makedirs(tensorboard_path, exist_ok=True)
    save_path["tensorboard_path"] = tensorboard_path

    checkpoint_path = os.path.join(save_root, "checkpoint")
    os.makedirs(checkpoint_path, exist_ok=True)
    save_path["checkpoint_path"] = checkpoint_path
    return save_path


def set_logger(log_path, distributed_rank=-1):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
        distributed_rank:
    """

    log_path = os.path.join(log_path, "train.log")

    if int(distributed_rank) > 0:
        logger_not_root = logging.getLogger(name=__name__)
        logger_not_root.propagate = False
        return logger_not_root

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    coloredlogs.install(level='INFO',
                        logger=logger,
                        fmt='%(asctime)s %(name)s %(message)s')
    file_handler = logging.FileHandler(log_path)
    log_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info('Output and logs will be saved to {}'.format(log_path))
    return logger

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)
