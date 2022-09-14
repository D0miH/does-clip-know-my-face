from joblib import Parallel
from tqdm.auto import tqdm

class TQDMParallel(Parallel):
    def __init__(self, progress_bar=True, total=None, *args, **kwargs):
        self.progress_bar = progress_bar
        self.total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self.progress_bar, total=self.total) as self.pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self.total is None:
            self.pbar.total = self.n_dispatched_tasks
        self.pbar.n = self.n_completed_tasks
        self.pbar.refresh()