import numpy as np


def calculate_mean_iat(packets):
    """
    Calculate the mean inter-arrival time (IAT)
    of packets in a flow.

    Returns:
        float: Mean IAT in milliseconds.
    """

    if len(packets) < 2:
        return 0.0

    timestamps = np.array(
        [float(packet.time) for packet in packets],
        dtype=np.float64
    )

    iats = np.diff(timestamps) * 1000.0

    return float(np.mean(iats))


def calculate_iat_variance(packets):
    """
    Calculate the variance of inter-arrival times.

    Returns:
        float: IAT variance.
    """

    if len(packets) < 2:
        return 0.0

    timestamps = np.array(
        [float(packet.time) for packet in packets],
        dtype=np.float64
    )

    iats = np.diff(timestamps) * 1000.0

    return float(np.var(iats))


def calculate_jitter(packets):
    """
    Calculate jitter as the standard deviation
    of inter-arrival times.

    Returns:
        float: Jitter in milliseconds.
    """

    if len(packets) < 2:
        return 0.0

    timestamps = np.array(
        [float(packet.time) for packet in packets],
        dtype=np.float64
    )

    iats = np.diff(timestamps) * 1000.0

    return float(np.std(iats))


def calculate_entropy(packets):
    """
    Calculate Shannon entropy of the payload bytes
    in a flow.

    Returns:
        float: Shannon entropy.
    """

    payload_bytes = []

    for packet in packets:

        if hasattr(packet, "payload"):

            try:
                payload_bytes.extend(
                    bytes(packet.payload)
                )

            except Exception:
                pass

    if len(payload_bytes) == 0:
        return 0.0

    data = np.array(
        payload_bytes,
        dtype=np.uint8
    )

    counts = np.bincount(
        data,
        minlength=256
    )

    probabilities = (
        counts / np.sum(counts)
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        probabilities *
        np.log2(probabilities)
    )

    return float(entropy)


def calculate_mean_packet_length(packets):
    """
    Calculate the average packet length.

    Returns:
        float: Mean packet length in bytes.
    """

    if len(packets) == 0:
        return 0.0

    lengths = np.array(
        [len(packet) for packet in packets],
        dtype=np.float64
    )

    return float(np.mean(lengths))


def calculate_packet_length_variance(packets):
    """
    Calculate the variance of packet lengths.

    Returns:
        float: Packet-length variance.
    """

    if len(packets) == 0:
        return 0.0

    lengths = np.array(
        [len(packet) for packet in packets],
        dtype=np.float64
    )

    return float(np.var(lengths))


def calculate_flow_duration(packets):
    """
    Calculate the duration of a traffic flow.

    Returns:
        float: Flow duration in milliseconds.
    """

    if len(packets) < 2:
        return 0.0

    timestamps = np.array(
        [float(packet.time) for packet in packets],
        dtype=np.float64
    )

    duration = (
        np.max(timestamps) -
        np.min(timestamps)
    ) * 1000.0

    return float(duration)


def calculate_burstiness(packets):
    """
    Calculate a burstiness measure from
    inter-arrival times.

    The value is approximately in the range
    [-1, 1].

    Returns:
        float: Burstiness score.
    """

    if len(packets) < 2:
        return 0.0

    timestamps = np.array(
        [float(packet.time) for packet in packets],
        dtype=np.float64
    )

    iats = np.diff(timestamps)

    mean_iat = np.mean(iats)
    std_iat = np.std(iats)

    denominator = (
        std_iat + mean_iat
    )

    if denominator == 0:
        return 0.0

    burstiness = (
        (std_iat - mean_iat)
        / denominator
    )

    return float(burstiness)


def calculate_statistics(packets):
    """
    Calculate all eight statistical traffic features.

    Features:

        1. Mean IAT
        2. IAT variance
        3. Jitter
        4. Entropy
        5. Mean packet length
        6. Packet-length variance
        7. Flow duration
        8. Burstiness

    Returns:
        numpy.ndarray:
        [
            mean_iat,
            iat_variance,
            jitter,
            entropy,
            mean_packet_length,
            packet_length_variance,
            flow_duration,
            burstiness
        ]
    """

    mean_iat = calculate_mean_iat(
        packets
    )

    iat_variance = calculate_iat_variance(
        packets
    )

    jitter = calculate_jitter(
        packets
    )

    entropy = calculate_entropy(
        packets
    )

    mean_packet_length = (
        calculate_mean_packet_length(
            packets
        )
    )

    packet_length_variance = (
        calculate_packet_length_variance(
            packets
        )
    )

    flow_duration = (
        calculate_flow_duration(
            packets
        )
    )

    burstiness = calculate_burstiness(
        packets
    )

    statistics = np.array(
        [
            mean_iat,
            iat_variance,
            jitter,
            entropy,
            mean_packet_length,
            packet_length_variance,
            flow_duration,
            burstiness
        ],
        dtype=np.float32
    )

    return statistics