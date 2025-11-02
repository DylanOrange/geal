import os
import sys
import time
import logging

def setup_logger(cfg_train, use_stdout=True):
    """
    Set up a unified training logger with both console and file outputs.
    The log file path is automatically created based on save_dir/name.

    Args:
        cfg_train (dict): training configuration block from YAML (contains 'name', 'save_dir', etc.)
        use_stdout (bool): whether to print logs to console in addition to saving to file

    Returns:
        logger (logging.Logger): ready-to-use logger instance
        run_tag (str): unique identifier for the current run (e.g., "PointRefer_at_11.04_23.15.42")
    """
    # === Generate timestamp and unique run identifier ===
    now = time.localtime()
    now_str = f"{now.tm_mon:02d}.{now.tm_mday:02d}_{now.tm_hour:02d}.{now.tm_min:02d}.{now.tm_sec:02d}"
    run_tag = f"{cfg_train['name']}_at_{now_str}"

    # === Create log directory and file ===
    log_dir = os.path.join(cfg_train["save_dir"], cfg_train["name"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, cfg_train.get("log_name", "train.log"))

    # === Initialize logger instance ===
    logger = logging.getLogger(run_tag)
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to prevent duplicate output
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # === File handler (writes to disk) ===
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.DEBUG)

    # === Console handler (optional, prints to stdout) ===
    if use_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.DEBUG)
        logger.addHandler(sh)

    # === Define log message format ===
    fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # === Print startup information ===
    logger.info("=" * 80)
    logger.info(f"New training session started: {run_tag}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', now)}")
    logger.info("Configuration Summary:")
    for k, v in cfg_train.items():
        logger.info(f"   {k}: {v}")
    logger.info("=" * 80 + "\n")

    return logger, run_tag