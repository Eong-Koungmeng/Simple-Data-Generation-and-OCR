import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import json


class OCRDataset(Dataset):
    def __init__(self, labels_file, images_dir, char_to_idx, img_height=32, img_width=None, transform=None, augment=False):
        self.images_dir = Path(images_dir)
        self.char_to_idx = char_to_idx
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform
        self.augment = augment

        self.samples = []
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    filename, text = parts
                    self.samples.append((filename, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, text = self.samples[idx]

        img_path = self.images_dir / filename
        image = Image.open(img_path).convert('L')  # Force grayscale

        w, h = image.size
        aspect_ratio = w / h
        new_h = self.img_height
        new_w = int(new_h * aspect_ratio)

        if self.img_width:
            new_w = min(new_w, self.img_width)

        image = image.resize((new_w, new_h), Image.LANCZOS)

        image = np.array(image, dtype=np.float32)

        if self.augment:
            import random
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                image = np.clip(image * brightness_factor, 0, 255)
            if random.random() > 0.5:
                contrast_factor = random.uniform(0.8, 1.2)
                mean = image.mean()
                image = np.clip((image - mean) *
                                contrast_factor + mean, 0, 255)

        image = image / 255.0
        image = torch.FloatTensor(image).unsqueeze(0)  # [1, H, W]

        encoded_text = [self.char_to_idx.get(
            char, self.char_to_idx.get('[UNK]', 1)) for char in text]

        return image, encoded_text, text


def collate_fn(batch):
    images, encoded_texts, texts = zip(*batch)

    max_width = max([img.size(2) for img in images])
    padded_images = []
    for img in images:
        _, h, w = img.size()
        padded = torch.zeros(1, h, max_width)
        padded[:, :, :w] = img
        padded_images.append(padded)

    images = torch.stack(padded_images, 0)

    text_lengths = torch.LongTensor([len(text) for text in encoded_texts])

    encoded_texts_flat = []
    for text in encoded_texts:
        encoded_texts_flat.extend(text)
    encoded_texts = torch.LongTensor(encoded_texts_flat)

    return images, encoded_texts, text_lengths, texts


def create_dataloaders(data_dir, labels_file, char_to_idx, batch_size=32, img_height=32, num_workers=4, is_train=False):
    dataset = OCRDataset(
        labels_file=labels_file,
        images_dir=data_dir,
        char_to_idx=char_to_idx,
        img_height=img_height,
        augment=is_train  # Only augment training data
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,  # Only shuffle training data
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return dataloader


def split_dataset(labels_file, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    import random
    random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    with open(labels_file, 'r', encoding='utf-8') as f:
        samples = [line.strip() for line in f if line.strip()]

    random.shuffle(samples)

    total = len(samples)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    with open(output_dir / 'train_labels.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_samples) + '\n')

    with open(output_dir / 'val_labels.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_samples) + '\n')

    with open(output_dir / 'test_labels.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_samples) + '\n')

    print(f"Dataset split complete:")
    print(f"  Train: {len(train_samples)} ({train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_samples)} ({val_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_samples)} ({test_ratio*100:.0f}%)")

    return train_samples, val_samples, test_samples


def build_vocab(labels_file):
    chars = set()
    with open(labels_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                text = parts[1]
                chars.update(text)

    chars = sorted(list(chars))

    char_to_idx = {'[BLANK]': 0, '[UNK]': 1}
    for i, char in enumerate(chars, start=2):
        char_to_idx[char] = i

    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    print(f"Vocabulary size: {len(char_to_idx)}")
    print(f"Characters: {len(chars)}")

    return char_to_idx, idx_to_char
