import logging, time
import sys

# Logging
class RelativeSecondsFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = record.created - self.start_time
        record.relative_seconds = elapsed_seconds
        return super().format(record)
    
def setup_logger(cfg, log_id):
    log_filename = f"{cfg['log_file']}_{log_id}.log"
    logger = logging.getLogger(f"relative_seconds_{log_id}")
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers
    if not logger.handlers:
        formatter = RelativeSecondsFormatter('[%(relative_seconds).2f] %(message)s')
        # create a file handler
        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # for log_id = 0, also log to console
        if log_id == 0 :
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
    
    return logger