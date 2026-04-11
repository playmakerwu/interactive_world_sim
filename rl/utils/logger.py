from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: str):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log(self, step: int, metrics: dict):
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)

    def close(self):
        self.writer.close()
