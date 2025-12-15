import torch
from PIL import Image
import numpy as np
import json
from pathlib import Path

from models.crnn import CRNN, CRNN_Small
from utils import decode_predictions


class OCRPredictor:
    def __init__(self, checkpoint_path, vocab_path, device=None):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
            self.char_to_idx = vocab['char_to_idx']
            self.idx_to_char = {
                int(k): v for k, v in vocab['idx_to_char'].items()}

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']

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

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.img_height = config['img_height']

        print(f"Model loaded from: {checkpoint_path}")
        print(f"Vocabulary size: {len(self.char_to_idx)}")
        print(f"Device: {self.device}")

    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        image = Image.open(image_path).convert('L')

        w, h = image.size
        aspect_ratio = w / h
        new_h = self.img_height
        new_w = int(new_h * aspect_ratio)
        image = image.resize((new_w, new_h), Image.LANCZOS)

        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.FloatTensor(image).unsqueeze(
            0).unsqueeze(0)  # [1, 1, H, W]

        return image

    def predict(self, image_path):
        """Predict text from image"""
        image = self.preprocess_image(image_path).to(self.device)

        with torch.no_grad():
            output = self.model(image)  # [1, seq_len, num_classes]

        predictions = decode_predictions(output, self.idx_to_char)
        return predictions[0]

    def predict_batch(self, image_paths):
        """Predict text from multiple images"""
        results = []
        for image_path in image_paths:
            pred = self.predict(image_path)
            results.append(pred)
        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='OCR Inference')
    parser.add_argument('--checkpoint', type=str,
                        required=True, help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, required=True,
                        help='Path to vocabulary file')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--images_dir', type=str,
                        help='Path to images directory')
    parser.add_argument('--output', type=str,
                        help='Output file for predictions')

    args = parser.parse_args()

    predictor = OCRPredictor(args.checkpoint, args.vocab)

    if args.image:
        prediction = predictor.predict(args.image)
        print(f"\nImage: {args.image}")
        print(f"Prediction: {prediction}\n")

    elif args.images_dir:
        image_paths = list(Path(args.images_dir).glob('*.png')) + \
            list(Path(args.images_dir).glob('*.jpg'))
        print(f"\nProcessing {len(image_paths)} images...\n")

        predictions = predictor.predict_batch(image_paths)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                for img_path, pred in zip(image_paths, predictions):
                    f.write(f"{img_path.name}\t{pred}\n")
            print(f"Results saved to: {args.output}")
        else:
            for img_path, pred in zip(image_paths[:10], predictions[:10]):
                print(f"{img_path.name}: {pred}")


if __name__ == '__main__':
    main()
