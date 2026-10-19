import os
import pandas as pd

from src.preprocessing.pcap_reader import read_pcap
from src.preprocessing.flow_extractor import extract_flows
from src.preprocessing.statistics import calculate_statistics


INPUT_DIR = "data/raw"
OUTPUT_CSV = "all_statistics_8.csv"


CLASS_PATHS = {

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
        "Benign/SMB/SMB-2.pcap"
    ],

    "Weibo": [
        "Benign/Weibo/Weibo/Weibo-1.pcap",
        "Benign/Weibo/Weibo-2.pcap",
        "Benign/Weibo/Weibo-3.pcap",
        "Benign/Weibo/Weibo-4.pcap"
    ],

    "WorldOfWarcraft": [
        "Benign/WorldOfWarcraft.pcap"
    ],

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


rows = []


for class_name, pcap_list in CLASS_PATHS.items():

    print("\n" + "=" * 60)
    print(f"PROCESSING CLASS: {class_name}")
    print("=" * 60)

    global_index = 0

    for relative_path in pcap_list:

        pcap_path = os.path.join(
            INPUT_DIR,
            relative_path
        )

        if not os.path.exists(pcap_path):
            print(f"File not found: {pcap_path}")
            continue

        print(f"\nProcessing: {pcap_path}")

        packets = read_pcap(pcap_path)

        print(f"Packets loaded: {len(packets)}")

        flows = extract_flows(packets)

        print(f"Flows found: {len(flows)}")

        for flow_packets in flows.values():

            stats = calculate_statistics(flow_packets)

            rows.append({
                "label": class_name,
                "flow_index": global_index,
                "mean_iat": float(stats[0]),
                "iat_variance": float(stats[1]),
                "jitter": float(stats[2]),
                "entropy": float(stats[3]),
                "mean_packet_length": float(stats[4]),
                "packet_length_variance": float(stats[5]),
                "flow_duration": float(stats[6]),
                "burstiness": float(stats[7])
            })

            global_index += 1

    print(f"Total statistics for {class_name}: {global_index}")


df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\n" + "=" * 60)
print("8-STATISTICS GENERATION COMPLETE")
print("=" * 60)

print(f"Total rows: {len(df)}")
print(f"Total classes: {df['label'].nunique()}")
print(f"Output file: {OUTPUT_CSV}")

print("\nColumns:")
print(df.columns.tolist())