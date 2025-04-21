import logging, sys
def get_logger(level="INFO"):
    lg = logging.getLogger("robust_meta_sim")
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s"))
        lg.addHandler(h)
    lg.setLevel(level)
    return lg
