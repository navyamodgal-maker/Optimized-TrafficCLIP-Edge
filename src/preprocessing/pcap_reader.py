from scapy.all import rdpcap


def read_pcap(path):
    """
    Read a PCAP file and return all packets.
    """
    packets = rdpcap(path)
    return packets
