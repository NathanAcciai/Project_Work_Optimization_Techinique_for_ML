
import utils
from utils import *



TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def download_tiny_imagenet(data_dir: str) -> str:
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = data_dir / "tiny-imagenet-200"

    if dataset_dir.exists():
        print(f"Dataset già presente in {dataset_dir}")
        return str(dataset_dir)

    zip_path = data_dir / "tiny-imagenet-200.zip"
    if not zip_path.exists():
        print(f"Download da {TINY_IMAGENET_URL} ...")
        urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)
        print("Download completato.")

    print("Estrazione in corso...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    print(f"Estratto in {dataset_dir}")

    return str(dataset_dir)



class TinyImageNetTrain(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = Path(root) / "train"
        self.transform = transform

        classes = sorted(os.listdir(self.root))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        self.samples = []
        for cls in classes:
            img_dir = self.root / cls / "images"
            for fname in os.listdir(img_dir):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append((str(img_dir / fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class TinyImageNetVal(Dataset):
    def __init__(self, root: str, class_to_idx: dict, transform=None):
        self.root = Path(root) / "val"
        self.img_dir = self.root / "images"
        self.transform = transform
        self.class_to_idx = class_to_idx

        
        ann_path = self.root / "val_annotations.txt"
        self.samples = []
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                fname, wnid = parts[0], parts[1]
                self.samples.append((str(self.img_dir / fname), self.class_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    
class TinyImageNetTest(Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.samples = [base_dataset.samples[i] for i in indices]
        self.transform = transform

        self.class_to_idx = base_dataset.class_to_idx
        self.idx_to_class = base_dataset.idx_to_class

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


def build_transforms(img_size: int, train: bool, pretrained_norm: bool):
    mean = [0.485, 0.456, 0.406] if pretrained_norm else [0.4802, 0.4481, 0.3975]
    std = [0.229, 0.224, 0.225] if pretrained_norm else [0.2770, 0.2691, 0.2821]

    ops = []
    if train:
        ops += [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=img_size // 8),
        ]
    else:
        ops += [transforms.Resize((img_size, img_size))]

    ops += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(ops)


def get_dataloaders(
    data_dir: str = "./data",
    img_size: int = 64,
    batch_size: int = 128,
    pretrained_norm: bool = True,
    num_workers: int = 4,
):
    dataset_root = download_tiny_imagenet(data_dir)

    train_tf = build_transforms(img_size, train=True, pretrained_norm=pretrained_norm)
    val_tf = build_transforms(img_size, train=False, pretrained_norm=pretrained_norm)

    full_train = TinyImageNetTrain(dataset_root, transform=train_tf)
    
    test_size = int(0.2 * len(full_train))
    train_size = len(full_train) - test_size


    generator = torch.Generator().manual_seed(42)

    train_indices, test_indices = torch.utils.data.random_split(range(len(full_train)),[train_size, test_size],generator=generator)
    train_set = torch.utils.data.Subset(
        full_train,
        train_indices
    )
    full_train_no_aug = TinyImageNetTrain(
        dataset_root,
        transform=val_tf
    )
    test_set = TinyImageNetTest(
        full_train_no_aug,
        test_indices
    )
    val_set = TinyImageNetVal(
        dataset_root,
        class_to_idx=full_train.class_to_idx,
        transform=val_tf
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    ) 
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )  

    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)} | Classi: {len(full_train.class_to_idx)}")
    return train_loader, val_loader, test_loader


