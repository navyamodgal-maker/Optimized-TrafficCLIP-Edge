import numpy as np

from PIL import Image

from .anonymizer import anonymize_bytes


IMAGE_SIZE = 28
MAX_BYTES = IMAGE_SIZE * IMAGE_SIZE


def packets_to_bytes(
    packets,
    max_bytes=MAX_BYTES
):
    """
    Convert packets into a single byte sequence.

    The sequence is limited to 784 bytes,
    which is enough for a 28x28 grayscale image.
    """

    data = bytearray()

    for packet in packets:

        raw_packet = bytes(packet)

        remaining = max_bytes - len(data)

        if remaining <= 0:
            break

        data.extend(
            raw_packet[:remaining]
        )

    return bytes(data)


def bytes_to_image(data):
    """
    Convert a byte sequence into a 28x28
    grayscale traffic image.
    """

    data = anonymize_bytes(data)

    # Keep exactly 784 bytes
    data = data[:MAX_BYTES]

    # Pad if fewer than 784 bytes
    if len(data) < MAX_BYTES:

        data += bytes(
            MAX_BYTES - len(data)
        )

    array = np.frombuffer(
        data,
        dtype=np.uint8
    )

    array = array.reshape(
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    image = Image.fromarray(
        array,
        mode="L"
    )

    return image


def packets_to_image(packets):
    """
    Convert packets belonging to one flow
    into a 28x28 traffic image.
    """

    data = packets_to_bytes(packets)

    image = bytes_to_image(data)

    return image
