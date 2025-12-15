import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import json
import time
from tqdm import tqdm

from models.crnn import CRNN, CRNN_Small
from dataset import create_dataloaders, build_vocab, split_dataset
from utils import decode_predictions, calculate_cer, calculate_wer, save_checkpoint, load_checkpoint


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Build vocabulary
        print("\nBuilding vocabulary...")
        self.char_to_idx, self.idx_to_char = build_vocab(config['labels_file'])

        # Save vocab
        with open(self.output_dir / 'vocab.json', 'w', encoding='utf-8') as f:
            json.dump({'char_to_idx': self.char_to_idx,
                      'idx_to_char': self.idx_to_char}, f, ensure_ascii=False, indent=2)

        # Split dataset
        print("\nSplitting dataset...")
        split_dataset(
            labels_file=config['labels_file'],
            output_dir=self.output_dir,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42
        )

        # Create dataloaders
        print("\nCreating dataloaders...")
        self.train_loader = create_dataloaders(
            data_dir=config['images_dir'],
            labels_file=self.output_dir / 'train_labels.txt',
            char_to_idx=self.char_to_idx,
            batch_size=config['batch_size'],
            img_height=config['img_height'],
            num_workers=config['num_workers']
        )

        self.val_loader = create_dataloaders(
            data_dir=config['images_dir'],
            labels_file=self.output_dir / 'val_labels.txt',
            char_to_idx=self.char_to_idx,
            batch_size=config['batch_size'],
            img_height=config['img_height'],
            num_workers=config['num_workers']
        )

        # Create model
        print("\nInitializing model...")
        num_classes = len(self.char_to_idx)

        if config['model_type'] == 'small':
            self.model = CRNN_Small(
                img_height=config['img_height'],
                num_channels=1,
                num_classes=num_classes,
                hidden_size=config['hidden_size']
            )
        else:
            self.model = CRNN(
                img_height=config['img_height'],
                num_channels=1,
                num_classes=num_classes,
                hidden_size=config['hidden_size']
            )

        self.model = self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        # Tensorboard
        self.writer = SummaryWriter(self.output_dir / 'logs')

        self.best_val_loss = float('inf')
        self.start_epoch = 0

        print(
            f"\nModel parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (images, targets, target_lengths, texts) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward
            outputs = self.model(images)  # [batch, seq_len, num_classes]
            outputs = outputs.permute(1, 0, 2)  # [seq_len, batch, num_classes]
            outputs = outputs.log_softmax(2)

            # Calculate sequence lengths
            input_lengths = torch.full(
                (images.size(0),), outputs.size(0), dtype=torch.long)

            # Loss
            loss = self.criterion(
                outputs, targets, input_lengths, target_lengths)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets, target_lengths, texts in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                outputs_for_loss = outputs.permute(1, 0, 2).log_softmax(2)
                input_lengths = torch.full(
                    (images.size(0),), outputs_for_loss.size(0), dtype=torch.long)

                loss = self.criterion(
                    outputs_for_loss, targets, input_lengths, target_lengths)
                total_loss += loss.item()

                # Decode predictions
                preds = decode_predictions(outputs, self.idx_to_char)
                all_preds.extend(preds)
                all_targets.extend(texts)

        avg_loss = total_loss / len(self.val_loader)
        cer = calculate_cer(all_preds, all_targets)
        wer = calculate_wer(all_preds, all_targets)

        # Calculate exact match accuracy
        exact_matches = sum(1 for pred, target in zip(
            all_preds, all_targets) if pred == target)
        exact_match_acc = (exact_matches / len(all_targets)
                           * 100) if len(all_targets) > 0 else 0

        return avg_loss, cer, wer, exact_match_acc, all_preds[:5], all_targets[:5]

    def train(self):
        print(f"\n{'='*70}")
        print("Starting training...")
        print(f"{'='*70}\n")

        for epoch in range(self.start_epoch, self.config['epochs']):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_cer, val_wer, val_exact_acc, sample_preds, sample_targets = self.validate(
                epoch)

            # Scheduler step
            self.scheduler.step(val_loss)

            # Log
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss:    {train_loss:.4f}")
            print(f"  Val Loss:      {val_loss:.4f}")
            print(f"  Val CER:       {val_cer:.2f}%")
            print(f"  Val WER:       {val_wer:.2f}%")
            print(f"  Val Exact Acc: {val_exact_acc:.2f}%")
            print(f"  Time:          {epoch_time:.1f}s")
            print(f"\nSample predictions:")
            for pred, target in zip(sample_preds, sample_targets):
                print(f"  Pred: {pred}")
                print(f"  True: {target}")
                print()

            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('CER/val', val_cer, epoch)
            self.writer.add_scalar('WER/val', val_wer, epoch)
            self.writer.add_scalar('ExactMatch/val', val_exact_acc, epoch)
            self.writer.add_scalar(
                'LR', self.optimizer.param_groups[0]['lr'], epoch)

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_cer': val_cer,
                    'val_wer': val_wer,
                    'val_exact_acc': val_exact_acc,
                    'config': self.config
                },
                is_best,
                self.checkpoint_dir
            )

        self.writer.close()
        print(f"\n{'='*70}")
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")
        print(f"\n{'='*70}")
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")


def main():
    config = {
        'labels_file': './output/labels.txt',
        'images_dir': './output/images',
        'output_dir': './runs/crnn_experiment2',
        'model_type': 'small',  # 'small' or 'standard'
        'img_height': 32,
        'hidden_size': 256,
        'batch_size': 64,
        'num_workers': 4,
        'learning_rate': 0.001,
        'epochs': 50
    }

    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
