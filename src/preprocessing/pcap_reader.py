from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether


def read_pcap(path):
    packets = []

    reader = RawPcapReader(path)

    try:
        for raw_data, packet_metadata in reader:
            try:
                packet = Ether(raw_data)

                # Preserve the original capture timestamp.
                packet.time = (
                    packet_metadata.sec
                    + packet_metadata.usec / 1000000.0
                )

                packets.append(packet)

            except Exception:
                continue

    finally:
        reader.close()

    return packets