import os
import pandas as pd
import numpy as np
import scapy.all as scapy
import scipy.stats

from src.preprocessing.flow_extractor import get_flow_key

RAW_DIR = "data/raw"

PCAP_MAP = {
    "BitTorrent": ["Benign/BitTorrent.pcap"],
    "Facetime": ["Benign/Facetime.pcap"],
    "FTP": ["Benign/FTP.pcap"],
    "Gmail": ["Benign/Gmail.pcap"],
    "MySQL": ["Benign/MySQL.pcap"],
    "Outlook": ["Benign/Outlook.pcap"],
    "Skype": ["Benign/Skype.pcap"],
    "WorldOfWarcraft": ["Benign/WorldOfWarcraft.pcap"],

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

    "Miuref": ["Malware/Miuref.pcap"],
    "Tinba": ["Malware/Tinba.pcap"],
    "Zeus": ["Malware/Zeus.pcap"],
    "Cridex": ["Malware/Cridex/Cridex.pcap"],
    "Geodo": ["Malware/Geodo/Geodo.pcap"],
    "Htbot": ["Malware/Htbot/Htbot.pcap"],
    "Neris": ["Malware/Neris/Neris.pcap"],
    "Nsis-ay": ["Malware/Nsis-ay/Nsis-ay.pcap"],
    "Shifu": ["Malware/Shifu/Shifu.pcap"],
    "Virut": ["Malware/Virut/Virut.pcap"],
}

NEEDED_PER_CLASS = 700


def calculate_stats(flow_packets):
    timestamps = [float(p.time) for p in flow_packets]

    if len(timestamps) > 1:
        iats = np.diff(timestamps)
        mean_iat = np.mean(iats) * 1000
        jitter = np.std(iats) * 1000
    else:
        mean_iat = 0.0
        jitter = 0.0

    raw_bytes = b"".join([scapy.raw(p) for p in flow_packets])

    if len(raw_bytes) > 0:
        counts = np.bincount(
            np.frombuffer(raw_bytes, dtype=np.uint8),
            minlength=256
        )
        probs = counts / len(raw_bytes)
        entropy = scipy.stats.entropy(probs, base=2)
    else:
        entropy = 0.0

    return float(mean_iat), float(jitter), float(entropy)


def process_class(class_name, pcap_paths):

    results = []
    flow_count = 0

    for relative_path in pcap_paths:

        if flow_count >= NEEDED_PER_CLASS:
            break

        path = os.path.join(RAW_DIR, relative_path)

        print("Reading:", path)

        packets = scapy.rdpcap(path)

        flows = {}

        for packet in packets:

            if not packet.haslayer(scapy.IP):
                continue

            try:
                key = get_flow_key(packet)
            except Exception:
                continue

            if key is None:
                continue

            if key not in flows:
                flows[key] = []

            flows[key].append(packet)

        print("Flows found:", len(flows))

        for flow_packets in flows.values():

            if flow_count >= NEEDED_PER_CLASS:
                break

            mean_iat, jitter, entropy = calculate_stats(flow_packets)

            results.append({
                "label": class_name,
                "flow_index": flow_count,
                "mean_iat": mean_iat,
                "jitter": jitter,
                "entropy": entropy
            })

            flow_count += 1

    print("Statistics collected:", flow_count)

    return results


def main():

    print("=" * 60)
    print("CREATING QUICK TRAFFIC STATISTICS")
    print("=" * 60)

    all_results = []

    for class_name, pcap_paths in PCAP_MAP.items():

        print("\nClass:", class_name)

        results = process_class(class_name, pcap_paths)

        all_results.extend(results)

    df = pd.DataFrame(all_results)

    df.to_csv("all_statistics.csv", index=False)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print("Total rows:", len(df))
    print("Saved: all_statistics.csv")


if __name__ == "__main__":
    main()