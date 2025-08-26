import logging

# import h5py
# import numpy as np


from ..enums import ExportFormat
from ..pie import generate_pie_chart, generate_pie2_chart

logger = logging.getLogger("write_pie")


def write_pie(settings, subs_df, index_artifacts):
    logger.info("Generating PieChart...")
    filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.pie_flt) > 0].copy()
    pie_df, pie_fig = generate_pie_chart(filtered_subs_df)
    pie2_df, pie2_fig = generate_pie2_chart(filtered_subs_df)
    logger.info("Exporting PieChart...")
    if settings.write.pie_fmt & ExportFormat.PDF:
        pie_fig.write_image(settings.out_dir / "pie.pdf", engine="kaleido")
        pie2_fig.write_image(settings.out_dir / "pie2.pdf", engine="kaleido")
    if settings.write.pie_fmt & ExportFormat.PNG:
        pie_fig.write_image(settings.out_dir / "pie.png")
        pie2_fig.write_image(settings.out_dir / "pie2.png")
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
