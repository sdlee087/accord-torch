import logging, time, sys
import torch.multiprocessing as mp

# Logging
class RelativeSecondsFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = record.created - self.start_time
        record.relative_seconds = elapsed_seconds
        return super().format(record)
    
def setup_logger(cfg, lamb, log_id):
    log_filename = f"{cfg['log_file']}_{round(lamb * 100)}_{log_id}.log"
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
        # if log_id == 0 :
        #     ch = logging.StreamHandler(sys.stdout)
        #     ch.setLevel(logging.DEBUG)
        #     ch.setFormatter(formatter)
        #     logger.addHandler(ch)
    
    return logger

def setup_main_logger(log_filename = "root.log"):
    logger = logging.getLogger("root")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = RelativeSecondsFormatter('[%(relative_seconds).2f] %(message)s')
        # create a file handler
        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger


# Logging function for the root logger (Runs in a separate process)
def log_listener(queue, log_filename = "root.log"):
    """ Receives logs from child processes and writes to console + root.log """
    logger = logging.getLogger("root")
    logger.setLevel(logging.INFO)
    formatter = RelativeSecondsFormatter('[%(relative_seconds).2f] %(message)s')

    # File handler (writes to root.log)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler (prints to standard output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Listen for logs from the queue
    while True:
        try:
            record = queue.get()
            if record is None:
                break  # Stop when None is received
            logger.handle(record)  # Process log record
        except Exception as e:
            print(f"Logging error: {e}", file=sys.stderr)

# Setup a logger for child processes (Each process sends logs to the queue)
def setup_child_logger(queue):
    logger = logging.getLogger("root")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        queue_handler = logging.handlers.QueueHandler(queue)  # Send logs to queue
        logger.addHandler(queue_handler)

    return logger