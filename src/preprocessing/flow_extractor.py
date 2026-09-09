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

    # Normalize direction: without this, a request (A->B) and its response
    # (B->A) get different keys, so almost every "flow" ends up being a
    # single packet and gets dropped downstream (min 2 packets required).
    endpoint_a = (src_ip, src_port if src_port is not None else -1)
    endpoint_b = (dst_ip, dst_port if dst_port is not None else -1)

    if endpoint_a > endpoint_b:
        src_ip, dst_ip = dst_ip, src_ip
        src_port, dst_port = dst_port, src_port

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
