from embedding_reader import EmbeddingReader
import pandas as pd
import sys
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from joblib import Parallel, delayed
import math
sys.path.insert(0, "/workspace")

from datasets import FaceScrub

BATCH_SIZE = 10 ** 7

names = []

facescrub = FaceScrub(group='all', train=False)
names.extend(list(map(lambda x: " ".join(x.split("_")), facescrub.classes)))

european_actors_dataset = ImageFolder(root='./data/laion_european_celebs/actors/images')
european_actresses_dataset = ImageFolder(root='./data/laion_european_celebs/actresses/images')
names.extend(list(map(lambda x: " ".join(x.split("_")), european_actors_dataset.classes + european_actresses_dataset.classes)))

embedding_reader = EmbeddingReader(
    embeddings_folder="./data/laion400m-met-release/laion400m-embeddings/images",
    metadata_folder="./data/laion400m-met-release/laion400m-embeddings/metadata",
    file_format="parquet_npy",
    meta_columns=["caption", "url", "NSFW"]
)


class TQDMParallel(Parallel):
    def __init__(self, progress_bar=True, total=None, *args, **kwargs):
        self.progress_bar = progress_bar
        self.total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self.progress_bar, total=self.total, leave=False) as self.pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self.total is None:
            self.pbar.total = self.n_dispatched_tasks
        self.pbar.n = self.n_completed_tasks
        self.pbar.refresh()

def get_filtered_names(metadata, names):
    samples_with_names_in_caption = []
    for name in names:
        filtered_samples = metadata[metadata['caption'].str.contains(name)]
        if len(filtered_samples) > 0:
            df_copy = filtered_samples.copy(deep=True)
            df_copy['name'] = name
            samples_with_names_in_caption.append(df_copy)
    
    return pd.concat(samples_with_names_in_caption) if len(samples_with_names_in_caption) > 0 else []

n_procs = 128
parallel = TQDMParallel(n_jobs=n_procs, total=n_procs)
samples_with_names_in_caption = []
for emb, meta in embedding_reader(
    batch_size=BATCH_SIZE,
    max_ram_usage_in_bytes=64*(2**32), # 256 GB
    show_progress=True
):
    n = math.ceil(len(meta) / n_procs)
    argument_list = list([(meta[i:i+n], names) for i in range(0, len(meta), n)])
    
    results = parallel(
            delayed(get_filtered_names)(*arguments) for arguments in argument_list
    )

    # filter out all empty lists from processes which didn't find a name in the dataset
    for res in results:
        if len(res) > 0:
            samples_with_names_in_caption.append(res)
    
if len(samples_with_names_in_caption) > 0:
    df = pd.concat(samples_with_names_in_caption).reset_index(drop=True)
    df.to_csv('laion400m_experiments/names_found_in_laion400m_caption_search.csv')
else:
    print('None of the given names were found in the dataset')
