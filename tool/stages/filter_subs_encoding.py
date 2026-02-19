import logging

from tqdm import tqdm

from ..enums import ExportFilter

logger = logging.getLogger("filter_subs_encoding")


def filter_subs_encoding(settings, subs, subs_df):
    filtered_enc = set()
    logger.info("Filtering subgraphs (Encoding)...")
    filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.gen_flt.value) > 0].copy()
    subs_iter = [(i, sub) for i, sub in enumerate(subs) if i in filtered_subs_df.index]
    # for i, sub in enumerate(tqdm(subs, disable=not settings.progress)):
    for i, sub in tqdm(subs_iter, disable=not settings.progress):
        sub_data = subs_df.iloc[i]
        valids = []
        for enc_size in settings.filters.allowed_enc_sizes:
            enc_bits_left = sub_data[f"EncodingBitsLeft ({enc_size} bits)"]
            enc_weight = sub_data[f"EncodingWeight ({enc_size} bits)"]
            enc_footprint = sub_data[f"EncodingFootprint ({enc_size} bits)"]
            valid = True
            if enc_bits_left < settings.filters.min_enc_bits_left:
                valid = False
            elif enc_weight > settings.filters.max_enc_weight:
                valid = False
            elif enc_footprint > settings.filters.max_enc_footprint:
                valid = False
            valids.append(valid)
        # print("valids", valids)
        valid = any(valids)
        # print("valid", valid)
        if not valid:
            filtered_enc.add(i)
    # print("filtered_enc", filtered_enc)
    subs_df.loc[list(filtered_enc), "Status"] = ExportFilter.FILTERED_ENC.value
