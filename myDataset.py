from torch.utils.data import Dataset
from transformers.pipelines.text_generation import Chat

class MyDataset(Dataset):
    def __init__(self, dataset, format_fn=None):
        if format_fn:
            self.dataset = [format_fn(x) for x in dataset]
        else:
            self.dataset = dataset

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return Chat(self.dataset[idx])