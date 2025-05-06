import os
import shutil
import yaml
import uuid
import random
from pathlib import Path
from collections import defaultdict
from typing import List

def load_classes(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def save_yaml(out_path, class_list):
    yaml_data = {
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_list),
        'names': class_list
    }
    with open(out_path, 'w') as f:
        yaml.dump(yaml_data, f)

def remap_label_file(file_path, class_map, keep_classes):
    new_lines = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                if cls_id in class_map:
                    new_cls = class_map[cls_id]
                    if keep_classes is None or new_cls in keep_classes:
                        new_line = f"{new_cls} " + " ".join(parts[1:]) + "\n"
                        new_lines.append(new_line)
    return new_lines

def copy_and_remap_images_labels(src_img_dir, src_lbl_dir, out_img_dir, out_lbl_dir, class_map, name_prefix, keep_classes, logf):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    image_files = list(Path(src_img_dir).glob("*.*"))
    for img_file in image_files:
        lbl_file = Path(src_lbl_dir) / (img_file.stem + ".txt")
        if not lbl_file.exists():
            continue

        uid = uuid.uuid4().hex[:8]
        new_name = f"{name_prefix}_{uid}{img_file.suffix}"
        new_lbl_name = f"{name_prefix}_{uid}.txt"

        new_img_path = out_img_dir / new_name
        new_lbl_path = out_lbl_dir / new_lbl_name

        shutil.copy(img_file, new_img_path)

        remapped_lines = remap_label_file(lbl_file, class_map, keep_classes)
        if remapped_lines:
            with open(new_lbl_path, 'w') as f:
                f.writelines(remapped_lines)
            logf.write(f"✔ Copied {img_file.name} as {new_name} with {len(remapped_lines)} objects.\n")

def auto_split_dataset(image_dir, label_dir, ratio=(0.8, 0.1, 0.1)):
    all_files = list(Path(image_dir).glob("*.*"))
    random.shuffle(all_files)
    n_total = len(all_files)
    n_train = int(n_total * ratio[0])
    n_val = int(n_total * ratio[1])

    train = all_files[:n_train]
    val = all_files[n_train:n_train+n_val]
    test = all_files[n_train+n_val:]

    return {'train': train, 'val': val, 'test': test}

def reorganize_split(files, src_label_dir, out_img_dir, out_lbl_dir):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for f in files:
        lbl = src_label_dir / (f.stem + ".txt")
        if lbl.exists():
            shutil.move(str(f), out_img_dir / f.name)
            shutil.move(str(lbl), out_lbl_dir / (f.stem + ".txt"))

def merge_yolo_datasets(
    dataset_a, dataset_b, output_dir,
    class_filter: List[str] = None,
    generate_split_if_missing=True
):
    output_dir = Path(output_dir)
    log_path = output_dir / "merge_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, 'w') as logf:
        logf.write("=== YOLO Dataset Merge Log ===\n")

        # Load classes
        classes_a = load_classes(Path(dataset_a) / "data.yaml")
        classes_b = load_classes(Path(dataset_b) / "data.yaml")

        merged_classes = list(classes_a)
        class_map_a = {i: i for i in range(len(classes_a))}

        class_map_b = {}
        for idx_b, cls in enumerate(classes_b):
            if cls in merged_classes:
                class_map_b[idx_b] = merged_classes.index(cls)
            else:
                merged_classes.append(cls)
                class_map_b[idx_b] = len(merged_classes) - 1

        logf.write(f"\n🔢 Final Merged Class List ({len(merged_classes)} classes):\n")
        for i, name in enumerate(merged_classes):
            logf.write(f"  {i}: {name}\n")

        keep_classes = None
        if class_filter:
            keep_classes = [merged_classes.index(c) for c in class_filter if c in merged_classes]
            logf.write(f"\n🔎 Filtering for classes: {class_filter}\n")

        # Handle all splits
        for split in ['train', 'val', 'test']:
            for dataset, prefix, class_map in [(dataset_a, 'A', class_map_a), (dataset_b, 'B', class_map_b)]:
                img_dir = Path(dataset) / f"images/{split}"
                lbl_dir = Path(dataset) / f"labels/{split}"
                if img_dir.exists() and lbl_dir.exists():
                    copy_and_remap_images_labels(
                        img_dir, lbl_dir,
                        output_dir / f"images/{split}",
                        output_dir / f"labels/{split}",
                        class_map, prefix, keep_classes, logf
                    )

        # Optional: generate val/test if missing
        if generate_split_if_missing:
            for split in ['val', 'test']:
                img_split_dir = output_dir / f"images/{split}"
                if not img_split_dir.exists() or len(list(img_split_dir.glob("*.*"))) < 10:
                    logf.write(f"\n⚠️ {split.upper()} split missing or too small. Generating new splits...\n")
                    full_img_dir = output_dir / "images/train"
                    full_lbl_dir = output_dir / "labels/train"
                    split_data = auto_split_dataset(full_img_dir, full_lbl_dir)

                    for s in ['val', 'test']:
                        reorganize_split(
                            split_data[s],
                            full_lbl_dir,
                            output_dir / f"images/{s}",
                            output_dir / f"labels/{s}"
                        )

        # Write final YAML
        save_yaml(output_dir / "data.yaml", merged_classes)
        logf.write("\n✅ Merge complete.\n")

# Beispielnutzung
if __name__ == "__main__":
    merge_yolo_datasets(
        dataset_a="Dataset_A",
        dataset_b="Dataset_B",
        output_dir="Merged_Dataset",
        class_filter=None,  # z.B. ["car", "dog"]
        generate_split_if_missing=True
    )
