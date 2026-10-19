import pandas as pd
import os
import numpy as np
from scapy.all import rdpcap
from src.preprocessing.flow_extractor import extract_flows

files = [
    (r"C:\Users\nehal saxena\OneDrive\Desktop\New folder\Fair Comparision\fair_comparision\data\raw\Benign\Weibo\Weibo\Weibo-1.pcap", 0),
    (r"C:\Users\nehal saxena\OneDrive\Desktop\New folder\Fair Comparision\fair_comparision\data\raw\Benign\Weibo\Weibo\Weibo-2.pcap", 49861),
    (r"C:\Users\nehal saxena\OneDrive\Desktop\New folder\Fair Comparision\fair_comparision\data\raw\Benign\Weibo\Weibo\Weibo-3.pcap", 59928),
    (r"C:\Users\nehal saxena\OneDrive\Desktop\New folder\Fair Comparision\fair_comparision\data\raw\Benign\Weibo\Weibo\Weibo-4.pcap", 69962),
]
q = pd.concat([
    pd.read_csv("splits/train_quick.csv"),
    pd.read_csv("splits/val_quick.csv"),
    pd.read_csv("splits/test_quick.csv")
])

s = pd.read_csv("all_statistics.csv")

target = set(
    q[q["label"] == "Weibo"]["image_path"]
    .apply(lambda x: int(os.path.basename(x).replace(".png", "")))
)

existing = set(s[s["label"] == "Weibo"]["flow_index"])
missing = sorted(target - existing)

print("Missing Weibo statistics:", len(missing))

results = []

for path, start_index in files:
    print("Processing:", path)

    packets = rdpcap(path)
    flows = list(extract_flows(packets).values())

    for local_index, flow_packets in enumerate(flows):
        global_index = start_index + local_index

        if global_index not in missing:
            continue

        if len(flow_packets) > 1:
            timestamps = np.array(
                [float(pkt.time) for pkt in flow_packets],
                dtype=np.float64
            )
            iats = np.diff(timestamps) * 1000.0
            mean_iat = float(np.mean(iats))
            jitter = float(np.std(iats))
        else:
            mean_iat = 0.0
            jitter = 0.0

        raw_bytes = b"".join(bytes(pkt) for pkt in flow_packets)

        if raw_bytes:
            data = np.frombuffer(raw_bytes, dtype=np.uint8)
            counts = np.bincount(data, minlength=256)
            probabilities = counts[counts > 0] / len(data)
            entropy = float(
                -np.sum(probabilities * np.log2(probabilities))
            )
        else:
            entropy = 0.0

        results.append([
            global_index,
            "Weibo",
            mean_iat,
            jitter,
            entropy
        ])

print("Calculated:", len(results))

add = pd.DataFrame(
    results,
    columns=["flow_index", "label", "mean_iat", "jitter", "entropy"]
)

s = pd.concat([s, add], ignore_index=True)
s.to_csv("all_statistics.csv", index=False)

print("Added:", len(add))
print("Total statistics rows:", len(s))