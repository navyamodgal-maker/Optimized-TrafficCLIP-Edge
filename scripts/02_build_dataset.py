"""
Build traffic_data.npz for ProposedTrafficCLIP.

Reads raw pcap/7z files already downloaded by 01_download_pcap_data.py
into data/raw/<ClassName>/, extracts flows, converts each flow into a
28x28 grayscale image (resized to 224x224 for the ResNet-50 branch),
computes the 8 statistical features, and saves everything into a
single npz file with keys: x, y, labels, m.

Usage:
    python scripts/02_build_dataset.py --max-flows-per-class 300
"""

import argparse
import glob
import logging
import os

import numpy as np
from PIL import Image

from src.preprocessing.pcap_reader import read_pcap
from src.preprocessing.flow_extractor import extract_flows
from src.preprocessing.pcap_to_image import packets_to_image
from src.preprocessing.statistics import calculate_statistics


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Class list matches config/config.yaml (USTC-TFC2016, 10-class subset)
CLASS_NAMES = [
    "Skype", "MySQL", "BitTorrent", "Facetime", "Weibo",
    "Zeus", "Tinba", "Cridex", "Geodo", "Miuref"
]

IMAGE_SIZE_FOR_MODEL = 224  # ResNet-50 needs more than 28x28


def find_pcap_files(class_dir):
    """Find all pcap-like files under a class's raw data directory."""
    patterns = ["*.pcap", "*.pcapng"]
    files = []
    for pattern in patterns:
        files.extend(
            glob.glob(os.path.join(class_dir, "**", pattern), recursive=True)
        )
    return files


def process_class(class_name, raw_data_dir, max_flows_per_class):
    """Process all pcaps for one class, return (images, stats) lists."""

    class_dir = os.path.join(raw_data_dir, class_name)

    if not os.path.isdir(class_dir):
        logging.warning(f"No raw data directory found for class: {class_name} (expected {class_dir})")
        return [], []

    pcap_files = find_pcap_files(class_dir)

    if not pcap_files:
        logging.warning(f"No pcap files found for class: {class_name}")
        return [], []

    images = []
    stats_list = []

    for pcap_path in pcap_files:

        if len(images) >= max_flows_per_class:
            break

        logging.info(f"[{class_name}] Reading {os.path.basename(pcap_path)}")

        try:
            packets = read_pcap(pcap_path)
        except Exception as error:
            logging.warning(f"Failed to read {pcap_path}: {error}")
            continue

        flows = extract_flows(packets)

        for flow_packets in flows.values():

            if len(images) >= max_flows_per_class:
                break

            if len(flow_packets) < 2:
                continue

            try:
                image = packets_to_image(flow_packets)
                image = image.resize(
                    (IMAGE_SIZE_FOR_MODEL, IMAGE_SIZE_FOR_MODEL),
                    Image.BILINEAR
                )
                image_array = np.asarray(image, dtype=np.float32) / 255.0

                stats = calculate_statistics(flow_packets)

            except Exception as error:
                logging.warning(f"Skipping a flow in {class_name}: {error}")
                continue

            images.append(image_array)
            stats_list.append(stats)

    logging.info(f"[{class_name}] Collected {len(images)} flow samples")

    return images, stats_list


def main():
    parser = argparse.ArgumentParser(
        description="Build traffic_data.npz from raw USTC-TFC2016 pcaps"
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output", default="data/traffic_data.npz")
    parser.add_argument(
        "--max-flows-per-class",
        type=int,
        default=300,
        help="Cap samples per class to keep preprocessing + training fast"
    )
    args = parser.parse_args()

    all_images = []
    all_labels = []
    all_stats = []

    for class_index, class_name in enumerate(CLASS_NAMES):

        images, stats_list = process_class(
            class_name,
            args.raw_dir,
            args.max_flows_per_class
        )

        all_images.extend(images)
        all_stats.extend(stats_list)
        all_labels.extend([class_index] * len(images))

    if len(all_images) == 0:
        raise RuntimeError(
            "No samples were collected at all. Check that data/raw/<ClassName>/ "
            "contains pcap files (run scripts/01_download_pcap_data.py first)."
        )

    x = np.stack(all_images, axis=0)          # [N, 224, 224]
    x = np.expand_dims(x, axis=1)             # [N, 1, 224, 224]
    y = np.array(all_labels, dtype=np.int64)  # [N]
    m = np.stack(all_stats, axis=0)           # [N, 8]
    labels = np.array(CLASS_NAMES, dtype=object)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    np.savez_compressed(
        args.output,
        x=x,
        y=y,
        labels=labels,
        m=m
    )

    logging.info("=" * 60)
    logging.info(f"Saved dataset to: {args.output}")
    logging.info(f"Total samples : {len(all_images)}")
    logging.info(f"Image shape   : {x.shape}")
    logging.info(f"Stats shape   : {m.shape}")

    unique, counts = np.unique(y, return_counts=True)
    for class_index, count in zip(unique, counts):
        logging.info(f"  {CLASS_NAMES[class_index]}: {count} samples")

    logging.info("=" * 60)


if __name__ == "__main__":
    main()
