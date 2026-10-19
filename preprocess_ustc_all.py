import os
import glob
import argparse

from src.preprocessing.pcap_reader import read_pcap
from src.preprocessing.flow_extractor import extract_flows
from src.preprocessing.pcap_to_image import packets_to_image


# Your actual dataset location
INPUT_DIR = "data/raw"

# Where generated PNG images will be stored
OUTPUT_DIR = "data/processed/USTC-TFC2016"


# Full USTC-TFC2016 dataset: 20 classes
CLASS_PATHS = {

    # =========================
    # BENIGN
    # =========================

    "BitTorrent": [
        "Benign/BitTorrent.pcap"
    ],

    "Facetime": [
        "Benign/Facetime.pcap"
    ],

    "FTP": [
        "Benign/FTP.pcap"
    ],

    "Gmail": [
        "Benign/Gmail.pcap"
    ],

    "MySQL": [
        "Benign/MySQL.pcap"
    ],

    "Outlook": [
        "Benign/Outlook.pcap"
    ],

    "Skype": [
        "Benign/Skype.pcap"
    ],

   "SMB": [
    "Benign/SMB/SMB/SMB-1.pcap",
    "Benign/SMB/SMB/SMB-2.pcap"
],

   "Weibo": [
    "Benign/Weibo/Weibo/Weibo-1.pcap",
    "Benign/Weibo/Weibo/Weibo-2.pcap",
    "Benign/Weibo/Weibo/Weibo-3.pcap",
    "Benign/Weibo/Weibo/Weibo-4.pcap"
],

    "WorldOfWarcraft": [
        "Benign/WorldOfWarcraft.pcap"
    ],


    # =========================
    # MALWARE
    # =========================

    "Cridex": [
        "Malware/Cridex/Cridex.pcap"
    ],

    "Geodo": [
        "Malware/Geodo/Geodo.pcap"
    ],

    "Htbot": [
        "Malware/Htbot/Htbot.pcap"
    ],

    "Miuref": [
        "Malware/Miuref.pcap"
    ],

    "Neris": [
        "Malware/Neris/Neris.pcap"
    ],

    "Nsis-ay": [
        "Malware/Nsis-ay/Nsis-ay.pcap"
    ],

    "Shifu": [
        "Malware/Shifu/Shifu.pcap"
    ],

    "Tinba": [
        "Malware/Tinba.pcap"
    ],

    "Virut": [
        "Malware/Virut/Virut.pcap"
    ],

    "Zeus": [
        "Malware/Zeus.pcap"
    ]
}


def process_pcap(pcap_path, output_folder, start_index):

    print(f"\nProcessing: {pcap_path}")

    packets = read_pcap(pcap_path)

    print(f"Packets loaded: {len(packets)}")

    flows = extract_flows(packets)

    print(f"Flows found: {len(flows)}")

    index = start_index

    for flow_packets in flows.values():

        image = packets_to_image(flow_packets)

        output_path = os.path.join(
            output_folder,
            f"{index}.png"
        )

        image.save(output_path)

        index += 1

    return index


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--classes",
        nargs="+",
        default=None
    )

    args = parser.parse_args()

    classes_to_process = (
        args.classes
        if args.classes
        else CLASS_PATHS.keys()
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    for class_name in classes_to_process:

        if class_name not in CLASS_PATHS:

            print(
                f"Skipping unknown class: {class_name}"
            )

            continue

        print("\n" + "=" * 50)
        print(f"PROCESSING CLASS: {class_name}")
        print("=" * 50)

        output_folder = os.path.join(
            OUTPUT_DIR,
            class_name
        )

        os.makedirs(
            output_folder,
            exist_ok=True
        )

        existing_images = glob.glob(
            os.path.join(
                output_folder,
                "*.png"
            )
        )

        image_index = len(existing_images)

        for relative_path in CLASS_PATHS[class_name]:

            pcap_path = os.path.join(
                INPUT_DIR,
                relative_path
            )

            if not os.path.exists(pcap_path):

                print(
                    f"File not found: {pcap_path}"
                )

                continue

            image_index = process_pcap(
                pcap_path,
                output_folder,
                image_index
            )

        print(
            f"\n{class_name} completed!"
        )

        print(
            f"Total images: {image_index}"
        )


if __name__ == "__main__":
    main()