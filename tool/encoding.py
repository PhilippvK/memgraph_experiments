import numpy as np


def calc_encoding_footprint(enc_bits_sum, enc_size):
    if enc_size == 32:
        opcode_bits = 7
        remaining_bits = enc_size - opcode_bits
        enc_bits_left = remaining_bits - enc_bits_sum
        if enc_bits_left >= 0:
            enc_weight = 1 / (2**enc_bits_left)
        else:
            enc_weight = np.nan

        enc_footprint = enc_bits_sum / remaining_bits
    else:
        NotImplementedError(f"Encoding Size: {enc_size}")
    return enc_bits_left, enc_weight, enc_footprint

