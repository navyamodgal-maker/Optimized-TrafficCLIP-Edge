import os
import pandas as pd
import numpy as np
from scapy.all import rdpcap, raw

from src.preprocessing.flow_extractor import extract_flows


# --------------------------------------------------
# Settings
# --------------------------------------------------

PCAP_DIR = "data/raw"
OUTPUT_CSV = "all_statistics.csv"

QUICK_FILES = [
    "splits/train_quick.csv",
    "splits/val_quick.csv",
    "splits/test_quick.csv"
]


# Same PCAP paths used when creating the images
CLASS_PATHS = {
    "BitTorrent": ["Benign/BitTorrent.pcap"],
    "Facetime": ["Benign/Facetime.pcap"],
    "FTP": ["Benign/FTP.pcap"],
    "Gmail": ["Benign/Gmail.pcap"],
    "MySQL": ["Benign/MySQL.pcap"],
    "Outlook": ["Benign/Outlook.pcap"],
    "Skype": ["Benign/Skype.pcap"],

    "SMB": [
        "Benign/SMB/SMB/SMB-1.pcap",
        "Benign/SMB/SMB-2.pcap"
    ],

    "Weibo": [
        "Benign/Weibo/Weibo/Weibo-1.pcap",
        "Benign/Weibo/Weibo-2.pcap",
        "Benign/Weibo/Weibo-3.pcap",
        "Benign/Weibo/Weibo-4.pcap"
    ],

    "WorldOfWarcraft": ["Benign/WorldOfWarcraft.pcap"],

    "Cridex": ["Malware/Cridex/Cridex.pcap"],
    "Geodo": ["Malware/Geodo/Geodo.pcap"],
    "Htbot": ["Malware/Htbot/Htbot.pcap"],
    "Miuref": ["Malware/Miuref.pcap"],
    "Neris": ["Malware/Neris/Neris.pcap"],
    "Nsis-ay": ["Malware/Nsis-ay/Nsis-ay.pcap"],
    "Shifu": ["Malware/Shifu/Shifu.pcap"],
    "Tinba": ["Malware/Tinba.pcap"],
    "Virut": ["Malware/Virut/Virut.pcap"],
    "Zeus": ["Malware/Zeus.pcap"]
}


def calculate_stats(flow_packets):

    # Mean inter-arrival time
    if len(flow_packets) < 2:
        mean_iat = 0.0
        jitter = 0.0
    else:
        timestamps = np.array(
            [float(packet.time) for packet in flow_packets],
            dtype=np.float64
        )

        iats = np.diff(timestamps) * 1000.0

        mean_iat = float(np.mean(iats))
        jitter = float(np.std(iats))

    # Entropy of raw packet bytes
    try:
        raw_bytes = b"".join(
            [bytes(raw(packet)) for packet in flow_packets]
        )
    except Exception:
        raw_bytes = b""

    if len(raw_bytes) == 0:
        entropy = 0.0
    else:
        data = np.frombuffer(raw_bytes, dtype=np.uint8)

        counts = np.bincount(
            data,
            minlength=256
        )

        probabilities = counts / np.sum(counts)

        probabilities = probabilities[
            probabilities > 0
        ]

        entropy = float(
            -np.sum(
                probabilities *
                np.log2(probabilities)
            )
        )

    return mean_iat, jitter, entropy


def main():

    # --------------------------------------------------
    # Find exactly which image indices were selected
    # --------------------------------------------------

    selected = pd.concat(
        [pd.read_csv(f) for f in QUICK_FILES],
        ignore_index=True
    )

    selected["flow_index"] = selected["image_path"].apply(
        lambda x: int(
            os.path.basename(x).replace(".png", "")
        )
    )

    needed = {}

    for class_name, group in selected.groupby("label"):
        needed[class_name] = set(
            group["flow_index"].tolist()
        )

    print("Selected statistics required:")

    for class_name in sorted(needed):
        print(
            f"{class_name}: "
            f"{len(needed[class_name])} flows"
        )

    results = []

    # --------------------------------------------------
    # Process each class
    # --------------------------------------------------

    for class_name in sorted(needed):

        print("\n" + "=" * 60)
        print(f"PROCESSING: {class_name}")
        print("=" * 60)

        target_indices = needed[class_name]

        current_index = 0

        for relative_path in CLASS_PATHS[class_name]:

            pcap_path = os.path.join(
                PCAP_DIR,
                relative_path
            )

            print(f"Reading: {pcap_path}")

            if not os.path.exists(pcap_path):
                print("FILE NOT FOUND")
                continue

            packets = rdpcap(pcap_path)

            print(
                f"Packets loaded: {len(packets)}"
            )

            flows = extract_flows(packets)

            print(
                f"Flows found: {len(flows)}"
            )

            for flow_packets in flows.values():

                if current_index in target_indices:

                    mean_iat, jitter, entropy = (
                        calculate_stats(flow_packets)
                    )

                    results.append({
                        "label": class_name,
                        "flow_index": current_index,
                        "mean_iat": mean_iat,
                        "jitter": jitter,
                        "entropy": entropy
                    })

                current_index += 1

                # Stop once all required flows for this class
                # have been collected.
                if len(results) >= sum(
                    len(v) for v in needed.values()
                ):
                    pass

        print(
            f"Statistics collected so far: "
            f"{len(results)}"
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    stats = pd.DataFrame(results)

    stats = stats.sort_values(
        ["label", "flow_index"]
    )

    stats.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\n" + "=" * 60)
    print("STATISTICS COMPLETE")
    print("=" * 60)

    print(f"Total rows: {len(stats)}")
    print(f"Saved to: {OUTPUT_CSV}")

    print("\nRows per class:")
    print(
        stats["label"]
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()