import numpy as np
from PIL import Image

from .anonymizer import anonymize_bytes


IMAGE_SIZE = 28
MAX_BYTES = IMAGE_SIZE * IMAGE_SIZE
HEADER_BYTES = 54


def packets_to_bytes(packets, max_bytes=MAX_BYTES):
    """
    Convert flow packets to bytes.

    The first 54 bytes of each packet are removed to reduce
    IP/port/header leakage, following the project guideline.
    """

    data = bytearray()

    for packet in packets:
        raw_packet = bytes(packet)

        # Remove the first 54 bytes
        payload = raw_packet[HEADER_BYTES:]

        remaining = max_bytes - len(data)

        if remaining <= 0:
            break

        data.extend(payload[:remaining])

    return bytes(data)


def bytes_to_image(data):
    """
    Convert exactly 784 bytes into a 28x28 grayscale image.
    """

    data = anonymize_bytes(data)

    data = data[:MAX_BYTES]

    if len(data) < MAX_BYTES:
        data += bytes(MAX_BYTES - len(data))

    array = np.frombuffer(data, dtype=np.uint8)
    array = array.reshape(IMAGE_SIZE, IMAGE_SIZE)

    return Image.fromarray(array, mode="L")


def packets_to_image(packets):
    """
    Convert one network flow into a 28x28 traffic image.
    """

    data = packets_to_bytes(packets)
    return bytes_to_image(data)