from collections import defaultdict


def get_flow_key(packet):
    """
    Create a flow key using the network 5-tuple:

    Source IP
    Destination IP
    Source Port
    Destination Port
    Protocol
    """

    if not packet.haslayer("IP"):
        return None

    src_ip = packet["IP"].src
    dst_ip = packet["IP"].dst
    protocol = packet["IP"].proto

    src_port = None
    dst_port = None

    if packet.haslayer("TCP"):
        src_port = packet["TCP"].sport
        dst_port = packet["TCP"].dport

    elif packet.haslayer("UDP"):
        src_port = packet["UDP"].sport
        dst_port = packet["UDP"].dport

    return (
        src_ip,
        dst_ip,
        src_port,
        dst_port,
        protocol
    )


def extract_flows(packets):
    """
    Group packets belonging to the same flow.
    """

    flows = defaultdict(list)

    for packet in packets:

        flow_key = get_flow_key(packet)

        if flow_key is not None:
            flows[flow_key].append(packet)

    return dict(flows)
