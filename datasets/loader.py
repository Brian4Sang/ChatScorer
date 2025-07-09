# datasets/loader.py

from torch.utils.data import DataLoader
from datasets.dataset import StyleDataset
from datasets.collate import collate_style_batch

def get_dataloader(cfg, split: str):
    """
    cfg: OmegaConf 配置对象
    split: "train" / "test"
    Returns:
        PyTorch DataLoader
    """
    dataset = StyleDataset(cfg, split=split)
    is_train = split == "train"
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size if is_train else cfg.eval.batch_size,
        shuffle=is_train,
        num_workers=cfg.train.num_workers if is_train else 2,
        pin_memory=True,
        collate_fn=collate_style_batch
    )
    return loader