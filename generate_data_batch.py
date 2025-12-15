import os
import json
from pathlib import Path
from generate_data import SyntheticDataGenerator


class BatchDataGenerator(SyntheticDataGenerator):
    def __init__(self, *args, progress_file="progress.json", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_file = self.output_dir / progress_file
        self.progress = self._load_progress()

    def _load_progress(self):
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'processed_files': {}, 'total_images': 0}

    def _save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def generate_dataset_batch(self, batch_size=1000, save_interval=100):
        text_files = self._load_text_files()

        print(f"Found {len(text_files)} text files")
        print(f"Found {len(self.fonts)} fonts")
        print(f"Generating {self.images_per_text} images per text line")
        print(f"Batch size: {batch_size}, Save interval: {save_interval}")

        if self.progress['total_images'] > 0:
            print(
                f"\nResuming from previous run: {self.progress['total_images']} images already generated")

        image_counter = self.progress['total_images']

        for text_file in text_files:
            file_key = text_file.name

            print(f"\nProcessing: {file_key}")
            lines = self._read_text_lines(text_file)
            total_lines = len(lines)
            print(f"  - {total_lines} lines")

            if file_key in self.progress['processed_files']:
                processed_lines = self.progress['processed_files'][file_key]
                if processed_lines >= total_lines:
                    print(f"Already completed, skipping")
                    continue
                else:
                    print(f"Resuming from line {processed_lines + 1}")
                    lines = lines[processed_lines:]
                    start_line = processed_lines
            else:
                start_line = 0
                self.progress['processed_files'][file_key] = 0

            for line_idx, text in enumerate(lines, start=start_line):
                if not text:
                    continue

                for img_idx in range(self.images_per_text):
                    image_counter += 1
                    output_filename = f"img_{image_counter:06d}.png"

                    annotation = self.generate_image(
                        text=text,
                        output_filename=output_filename,
                        source_file=file_key
                    )

                    self.metadata.append(annotation)

                    if image_counter % save_interval == 0:
                        print(f"  Generated {image_counter} images...")
                        self.progress['total_images'] = image_counter
                        self.progress['processed_files'][file_key] = line_idx + 1
                        self._save_progress()
                        self._save_metadata_incremental()

            self.progress['processed_files'][file_key] = total_lines
            self.progress['total_images'] = image_counter
            self._save_progress()
            print(f"  ✓ Completed {file_key}")

        self._save_metadata_final()

        print(f"\nDataset generation complete!")
        print(f"  Total images: {image_counter}")
        print(f"  Images directory: {self.images_dir}")
        print(f"  Annotations directory: {self.annotations_dir}")

    def _save_metadata_incremental(self):
        labels_path = self.output_dir / "labels.txt"
        with open(labels_path, 'w', encoding='utf-8') as f:
            for annotation in self.metadata:
                f.write(f"{annotation['filename']}\t{annotation['text']}\n")

    def _save_metadata_final(self):
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        labels_path = self.output_dir / "labels.txt"
        with open(labels_path, 'w', encoding='utf-8') as f:
            for annotation in self.metadata:
                f.write(f"{annotation['filename']}\t{annotation['text']}\n")

        print(f"  Metadata file: {metadata_path}")
        print(f"  Labels file: {labels_path}")


def main():

    source_data_dir = "./source_data"
    fonts_dir = "./fonts"
    output_dir = "./output"
    images_per_text = 8
    use_stratified = True
    batch_size = 1000
    save_interval = 100

    generator = BatchDataGenerator(
        source_data_dir=source_data_dir,
        fonts_dir=fonts_dir,
        output_dir=output_dir,
        images_per_text=images_per_text,
        use_stratified=use_stratified
    )

    generator.generate_dataset_batch(
        batch_size=batch_size,
        save_interval=save_interval
    )

    print("Complete")


if __name__ == "__main__":
    main()
