import os
import random
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from io import BytesIO
import glob


class SyntheticDataGenerator:
    def __init__(self, source_data_dir, fonts_dir, output_dir, images_per_text=8, use_stratified=True):
        self.source_data_dir = Path(source_data_dir)
        self.fonts_dir = Path(fonts_dir)
        self.output_dir = Path(output_dir)
        self.images_per_text = images_per_text
        self.use_stratified = use_stratified

        self.augmentation_plan = [
            {'level': 'clean', 'types': []},
            {'level': 'light', 'types': ['blur']},
            {'level': 'light', 'types': ['noise']},
            {'level': 'medium', 'types': ['compression']},
            {'level': 'medium', 'types': ['blur', 'noise']},
            {'level': 'heavy', 'types': ['blur', 'compression']},
            {'level': 'heavy', 'types': ['noise', 'compression']},
            {'level': 'heavy', 'types': ['social_media']},
        ]

        self.images_dir = self.output_dir / "images"
        self.annotations_dir = self.output_dir / "annotations"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

        self.fonts = self._load_fonts()
        self.font_sizes = [24, 28, 32, 36, 40, 48, 56, 64]

        self.metadata = []

    def _load_fonts(self):
        font_files = list(self.fonts_dir.glob("*.ttf")) + \
            list(self.fonts_dir.glob("*.otf"))
        if not font_files:
            raise ValueError(f"No fonts found in {self.fonts_dir}")
        return font_files

    def _load_text_files(self):
        text_files = list(self.source_data_dir.glob("*.txt"))
        if not text_files:
            raise ValueError(f"No text files found in {self.source_data_dir}")
        return text_files

    def _read_text_lines(self, text_file):
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines

    def _create_text_image(self, text, font_path, font_size):
        try:
            font = ImageFont.truetype(str(font_path), font_size)
        except Exception as e:
            print(f"Error loading font {font_path}: {e}")
            font = ImageFont.load_default()

        temp_img = Image.new('RGB', (1, 1), 'white')
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        padding = random.randint(2, 5)
        img_width = text_width + 2 * padding
        img_height = text_height + 2 * padding

        image = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(image)

        draw.text((padding - bbox[0], padding - bbox[1]),
                  text, font=font, fill='black')

        return image, (text_width, text_height)

    def _add_gaussian_noise(self, image, mean=0, std=10):
        img_array = np.array(image, dtype=np.float32)
        noise = np.random.normal(mean, std, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    def _add_salt_pepper_noise(self, image, amount=0.01):
        img_array = np.array(image)

        num_salt = int(amount * img_array.size * 0.5)
        coords = [np.random.randint(0, i - 1, num_salt)
                  for i in img_array.shape[:2]]
        img_array[coords[0], coords[1], :] = 255

        num_pepper = int(amount * img_array.size * 0.5)
        coords = [np.random.randint(0, i - 1, num_pepper)
                  for i in img_array.shape[:2]]
        img_array[coords[0], coords[1], :] = 0

        return Image.fromarray(img_array)

    def _add_blur(self, image, blur_type='gaussian'):
        if blur_type == 'gaussian':
            radius = random.uniform(0.5, 2.5)
            return image.filter(ImageFilter.GaussianBlur(radius))
        elif blur_type == 'box':
            radius = random.randint(1, 3)
            return image.filter(ImageFilter.BoxBlur(radius))
        else:
            return image.filter(ImageFilter.BLUR)

    def _jpeg_compression(self, image, quality):
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

    def _social_media_compression(self, image):
        img = image.copy()
        width, height = img.size

        scale_down = random.uniform(0.6, 0.9)
        new_width = int(width * scale_down)
        new_height = int(height * scale_down)
        img = img.resize((new_width, new_height), Image.BILINEAR)

        img = self._jpeg_compression(img, quality=random.randint(45, 65))

        img = img.resize((width, height), Image.BILINEAR)

        img = self._jpeg_compression(img, quality=random.randint(40, 60))

        if random.random() > 0.5:
            img = self._jpeg_compression(img, quality=random.randint(35, 55))

        return img

    def _apply_augmentations(self, image, augmentation_level='medium', augmentation_types=None):
        img = image.copy()

        if augmentation_level == 'clean':
            return img, ['clean']

        augmentations = []

        if augmentation_level == 'light':
            if augmentation_types:
                if 'blur' in augmentation_types:
                    img = self._add_blur(img, 'gaussian')
                    augmentations.append('blur_light')
                if 'noise' in augmentation_types:
                    img = self._add_gaussian_noise(
                        img, std=random.uniform(5, 12))
                    augmentations.append('gaussian_noise_light')
                if 'compression' in augmentation_types:
                    quality = random.randint(60, 75)
                    img = self._jpeg_compression(img, quality)
                    augmentations.append(f'jpeg_q{quality}')
            else:
                if random.random() > 0.4:
                    img = self._add_blur(img, 'gaussian')
                    augmentations.append('blur_light')

                if random.random() > 0.5:
                    img = self._add_gaussian_noise(
                        img, std=random.uniform(5, 12))
                    augmentations.append('gaussian_noise_light')

                quality = random.randint(60, 75)
                img = self._jpeg_compression(img, quality)
                augmentations.append(f'jpeg_q{quality}')

        elif augmentation_level == 'medium':
            if augmentation_types:
                if 'blur' in augmentation_types:
                    img = self._add_blur(
                        img, random.choice(['gaussian', 'box']))
                    augmentations.append('blur_medium')
                if 'noise' in augmentation_types:
                    img = self._add_gaussian_noise(
                        img, std=random.uniform(8, 18))
                    augmentations.append('gaussian_noise_medium')
                if 'compression' in augmentation_types:
                    quality = random.randint(45, 65)
                    img = self._jpeg_compression(img, quality)
                    augmentations.append(f'jpeg_q{quality}')
                if len(augmentation_types) > 1 and random.random() > 0.5:
                    img = self._add_salt_pepper_noise(
                        img, amount=random.uniform(0.008, 0.02))
                    augmentations.append('salt_pepper')
            else:
                if random.random() > 0.3:
                    img = self._add_blur(
                        img, random.choice(['gaussian', 'box']))
                    augmentations.append('blur_medium')

                if random.random() > 0.3:
                    img = self._add_gaussian_noise(
                        img, std=random.uniform(8, 18))
                    augmentations.append('gaussian_noise_medium')

                quality = random.randint(45, 65)
                img = self._jpeg_compression(img, quality)
                augmentations.append(f'jpeg_q{quality}')

                if random.random() > 0.6:
                    img = self._add_salt_pepper_noise(
                        img, amount=random.uniform(0.008, 0.02))
                    augmentations.append('salt_pepper')

        elif augmentation_level == 'heavy':
            if augmentation_types:
                if 'social_media' in augmentation_types:
                    img = self._social_media_compression(img)
                    augmentations.append('social_media_compression')
                else:
                    if 'blur' in augmentation_types:
                        img = self._add_blur(
                            img, random.choice(['gaussian', 'box']))
                        augmentations.append('blur_heavy')
                    if 'noise' in augmentation_types:
                        img = self._add_gaussian_noise(
                            img, std=random.uniform(12, 28))
                        augmentations.append('gaussian_noise_heavy')
                    if 'compression' in augmentation_types:
                        quality = random.randint(25, 50)
                        img = self._jpeg_compression(img, quality)
                        augmentations.append(f'jpeg_q{quality}')
                    if len(augmentation_types) > 1 and random.random() > 0.4:
                        img = self._add_salt_pepper_noise(
                            img, amount=random.uniform(0.015, 0.04))
                        augmentations.append('salt_pepper')
            else:
                if random.random() > 0.2:
                    img = self._add_blur(
                        img, random.choice(['gaussian', 'box']))
                    augmentations.append('blur_heavy')

                img = self._add_gaussian_noise(img, std=random.uniform(12, 28))
                augmentations.append('gaussian_noise_heavy')

                if random.random() > 0.3:
                    img = self._social_media_compression(img)
                    augmentations.append('social_media_compression')
                else:
                    quality = random.randint(25, 50)
                    img = self._jpeg_compression(img, quality)
                    augmentations.append(f'jpeg_q{quality}')

                if random.random() > 0.4:
                    img = self._add_salt_pepper_noise(
                        img, amount=random.uniform(0.015, 0.04))
                    augmentations.append('salt_pepper')

        return img, augmentations

    def generate_image(self, text, output_filename, source_file, stratified_plan=None):
        font_path = random.choice(self.fonts)
        font_size = random.choice(self.font_sizes)

        image, text_size = self._create_text_image(text, font_path, font_size)

        if stratified_plan:
            aug_level = stratified_plan['level']
            aug_types = stratified_plan['types']
            augmented_image, augmentations = self._apply_augmentations(
                image, aug_level, aug_types)
        else:
            aug_level = random.choices(
                ['light', 'medium', 'heavy'],
                weights=[0.2, 0.4, 0.4]
            )[0]
            augmented_image, augmentations = self._apply_augmentations(
                image, aug_level)

        image_path = self.images_dir / output_filename
        augmented_image.save(image_path, format='PNG')

        annotation = {
            'filename': output_filename,
            'text': text,
            'source_file': source_file,
            'font': font_path.name,
            'font_size': font_size,
            'image_size': {
                'width': augmented_image.width,
                'height': augmented_image.height
            },
            'text_size': {
                'width': text_size[0],
                'height': text_size[1]
            },
            'augmentation_level': aug_level,
            'augmentations_applied': augmentations,
            'stratified_plan': stratified_plan if stratified_plan else None
        }

        annotation_path = self.annotations_dir / \
            f"{Path(output_filename).stem}.json"
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)

        return annotation

    def generate_dataset(self):
        text_files = self._load_text_files()

        print(f"Found {len(text_files)} text files")
        print(f"Found {len(self.fonts)} fonts")
        print(f"Generating {self.images_per_text} images per text line")

        image_counter = 0

        for text_file in text_files:
            print(f"\nProcessing: {text_file.name}")
            lines = self._read_text_lines(text_file)
            print(f"  - {len(lines)} lines")

            for line_idx, text in enumerate(lines):
                if not text:
                    continue

                for img_idx in range(self.images_per_text):
                    image_counter += 1
                    output_filename = f"img_{image_counter:06d}.png"

                    stratified_plan = None
                    if self.use_stratified and img_idx < len(self.augmentation_plan):
                        stratified_plan = self.augmentation_plan[img_idx]

                    annotation = self.generate_image(
                        text=text,
                        output_filename=output_filename,
                        source_file=text_file.name,
                        stratified_plan=stratified_plan
                    )

                    self.metadata.append(annotation)

                    if image_counter % 100 == 0:
                        print(f"  Generated {image_counter} images...")

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        # Save simple text annotation file (commonly used for OCR training)
        labels_path = self.output_dir / "labels.txt"
        with open(labels_path, 'w', encoding='utf-8') as f:
            for annotation in self.metadata:
                f.write(f"{annotation['filename']}\t{annotation['text']}\n")

        print(f"\n✓ Dataset generation complete!")
        print(f"  Total images: {image_counter}")
        print(f"  Images directory: {self.images_dir}")
        print(f"  Annotations directory: {self.annotations_dir}")
        print(f"  Metadata file: {metadata_path}")
        print(f"  Labels file: {labels_path}")


def main():
    source_data_dir = "./source_data"
    fonts_dir = "./fonts"
    output_dir = "./output"
    images_per_text = 8
    use_stratified = True

    generator = SyntheticDataGenerator(
        source_data_dir=source_data_dir,
        fonts_dir=fonts_dir,
        output_dir=output_dir,
        images_per_text=images_per_text,
        use_stratified=use_stratified
    )

    generator.generate_dataset()


if __name__ == "__main__":
    main()
