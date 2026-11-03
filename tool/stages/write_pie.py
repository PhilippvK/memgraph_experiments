import logging
import os
import plotly.io as pio

# import h5py
# import numpy as np


from ..enums import ExportFormat
from ..pie import generate_pie_chart, generate_pie2_chart

logger = logging.getLogger("write_pie")


def write_pie(settings, subs_df, index_artifacts):
    if mathjax := os.environ.get("PLOTLY_MATHJAX"):
        pio.defaults.mathjax = mathjax
    logger.info("Generating PieChart...")
    filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.pie_flt.value) > 0].copy()
    pie_df, pie_fig = generate_pie_chart(filtered_subs_df)
    pie2_df, pie2_fig = generate_pie2_chart(filtered_subs_df)
    logger.info("Exporting PieChart...")
    figures, destinations = [], []
    for flag, extension in ((ExportFormat.PDF, "pdf"), (ExportFormat.PNG, "png")):
        if settings.write.pie_fmt & flag:
            figures.extend((pie_fig, pie2_fig))
            destinations.extend(
                (settings.out_dir / f"pie.{extension}", settings.out_dir / f"pie2.{extension}")
            )
    if figures:
        pio.write_images(fig=figures, file=destinations)
    if settings.write.pie_fmt & ExportFormat.HTML:
        pie_fig.write_html(settings.out_dir / "pie.html")
        pie2_fig.write_html(settings.out_dir / "pie2.html")
    if settings.write.pie_fmt & ExportFormat.CSV:
        pie_df.to_csv(settings.out_dir / "pie.csv", index=False)
        pie2_df.to_csv(settings.out_dir / "pie2.csv", index=False)
    if settings.write.pie_fmt & ExportFormat.PKL:
        pie_df.to_pickle(settings.out_dir / "pie.pkl")
        pie2_df.to_pickle(settings.out_dir / "pie2.pkl")
        index_artifacts[None]["pie"] = settings.out_dir / "pie.pkl"
        index_artifacts[None]["pie2"] = settings.out_dir / "pie2.pkl"
