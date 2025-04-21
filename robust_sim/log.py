import logging, sys
def get_logger(level="INFO"):
    log = logging.getLogger("rms_v18")
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s"))
        log.addHandler(h)
    log.setLevel(level)
    return log
