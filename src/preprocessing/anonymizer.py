def anonymize_bytes(data, header_bytes=54):
    """
    Remove the first part of the byte sequence
    to reduce exposure of network-identifying
    header information.

    The bytes are replaced with zero.
    """

    data = bytearray(data)

    limit = min(
        header_bytes,
        len(data)
    )

    for i in range(limit):
        data[i] = 0

    return bytes(data)
