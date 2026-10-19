import pandas as pd


def create_csv(split_name):

    image_csv = f"splits/{split_name}_quick.csv"

    stats_csv = "all_statistics.csv"

    output_csv = f"splits/{split_name}_quick_opt.csv"

    images = pd.read_csv(image_csv)
    stats = pd.read_csv(stats_csv)

    # Add flow_index based on the image number.
    images["flow_index"] = images["image_path"].apply(
        lambda x: int(x.replace("\\", "/").split("/")[-1].replace(".png", ""))
    )

    # Match statistics using class + flow index.
    merged = images.merge(
        stats,
        on=["label", "flow_index"],
        how="left"
    )

    # Check for missing statistics.
    missing = merged[
        merged[["mean_iat", "jitter", "entropy"]].isna().any(axis=1)
    ]

    if len(missing) > 0:
        print(f"ERROR: {len(missing)} rows are missing statistics.")
        print(missing.head())
        return

    merged = merged[
        ["image_path", "label", "mean_iat", "jitter", "entropy"]
    ]

    merged.to_csv(output_csv, index=False)

    print(f"{output_csv} created: {len(merged)} rows")


create_csv("train")
create_csv("val")
create_csv("test")