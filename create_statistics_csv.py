import os
import pandas as pd
import numpy as np
from scapy.utils import rdpcap
from scapy.layers.inet import IP

from src.preprocessing.flow_extractor import get_flow_key
from src.preprocessing.statistics import calculate_statistics


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
        "Benign/SMB/SMB/SMB-2.pcap"
    ],

    "Weibo": [
        "Benign/Weibo/Weibo/Weibo-1.pcap",
        "Benign/Weibo/Weibo/Weibo-2.pcap",
        "Benign/Weibo/Weibo/Weibo-3.pcap",
        "Benign/Weibo/Weibo/Weibo-4.pcap"
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


def collect_flows(pcap_paths):
    flows = {}

    for relative_path in pcap_paths:
        path = os.path.join(RAW_DIR, relative_path)

        print("Reading:", path)

        packets = rdpcap(path)

        for packet in packets:
            try:
                if not packet.haslayer(IP):
                    continue

                key = get_flow_key(packet)

                if key is None:
                    continue

                if key not in flows:
                    flows[key] = []

                flows[key].append(packet)

            except Exception:
                continue

    return list(flows.values())


def main():
    print("=" * 60)
    print("CREATING TRAFFIC STATISTICS")
    print("=" * 60)

    all_stats = {}

    for class_name, paths in PCAP_MAP.items():

        print("\nClass:", class_name)

        flows = collect_flows(paths)

        print("Flows found:", len(flows))

        class_stats = []

        for flow in flows:
            stats = calculate_statistics(flow)

            class_stats.append(stats)

        all_stats[class_name] = class_stats

        print("Statistics calculated:", len(class_stats))

    print("\nStatistics calculation complete.")


if __name__ == "__main__":
    main()