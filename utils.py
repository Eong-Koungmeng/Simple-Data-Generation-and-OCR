import torch
from pathlib import Path
import Levenshtein


def decode_predictions(outputs, idx_to_char):
    # outputs: [batch, seq_len, num_classes]
    _, preds = outputs.max(2)  # [batch, seq_len]

    decoded = []
    batch_size = outputs.size(0)

    for i in range(batch_size):
        pred_seq = preds[i]  # Get sequence for this batch item

        # CTC decode: remove blanks and duplicates
        chars = []
        prev_idx = None
        for idx in pred_seq.cpu().numpy():
            if idx == 0:  # blank token
                prev_idx = None
                continue
            if idx != prev_idx:  # Remove consecutive duplicates
                char = idx_to_char.get(int(idx), '')
                if char and char not in ['[BLANK]', '[UNK]']:
                    chars.append(char)
                prev_idx = idx

        decoded.append(''.join(chars))

    return decoded


def calculate_accuracy(predictions, targets):
    correct = sum(1 for pred, target in zip(
        predictions, targets) if pred == target)
    accuracy = (correct / len(targets) * 100) if len(targets) > 0 else 0
    return accuracy


def calculate_char_accuracy(predictions, targets):
    total_correct = 0
    total_chars = 0

    for pred, target in zip(predictions, targets):
        # Use Levenshtein distance for accurate character matching
        max_len = max(len(pred), len(target))
        if max_len == 0:
            continue
        distance = Levenshtein.distance(pred, target)
        correct = max_len - distance
        total_correct += correct
        total_chars += max_len

    char_acc = (total_correct / total_chars * 100) if total_chars > 0 else 0
    return char_acc


def calculate_cer(predictions, targets):
    total_distance = 0
    total_length = 0

    for pred, target in zip(predictions, targets):
        distance = Levenshtein.distance(pred, target)
        total_distance += distance
        total_length += len(target)

    cer = (total_distance / total_length * 100) if total_length > 0 else 0
    return cer


def calculate_wer(predictions, targets):
    total_distance = 0
    total_words = 0

    for pred, target in zip(predictions, targets):
        pred_words = pred.split()
        target_words = target.split()

        distance = Levenshtein.distance(
            ' '.join(pred_words), ' '.join(target_words))
        total_distance += distance
        total_words += len(target_words)

    wer = (total_distance / total_words * 100) if total_words > 0 else 0
    return wer


def save_checkpoint(state, is_best, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save latest
    checkpoint_path = checkpoint_dir / 'checkpoint_latest.pth'
    torch.save(state, checkpoint_path)

    if is_best:
        best_path = checkpoint_dir / 'checkpoint_best.pth'
        torch.save(state, best_path)

    if state['epoch'] % 10 == 0:
        epoch_path = checkpoint_dir / f"checkpoint_epoch_{state['epoch']}.pth"
        torch.save(state, epoch_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch'], checkpoint.get('val_loss', float('inf'))
