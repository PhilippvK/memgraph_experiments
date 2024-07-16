import plotly.express as px

from .enums import ExportFilter


def generate_pie_chart(subs_df):
    def helper(x):
        if x & ExportFilter.SELECTED:
            return "Selected"
        if x & ExportFilter.ISO:
            return "Iso"
        if x & ExportFilter.FILTERED_IO:
            return "Filtered (I/O)"
        if x & ExportFilter.FILTERED_COMPLEX:
            return "Filtered (Complex)"
        if x & ExportFilter.FILTERED_SIMPLE:
            return "Filtered (Simple)"
        if x & ExportFilter.FILTERED_PRED:
            return "Filtered (Pred)"
        if x & ExportFilter.INVALID:
            return "Invalid"
        if x & ExportFilter.ERROR:
            return "ERROR"
        return "Unknown"

    subs_df["Label"] = subs_df["Status"].apply(helper)
    pie_df = subs_df.value_counts("Label").rename_axis("Label").reset_index(name="Count")
    # print("pie_df")
    # print(pie_df)
    pie_fig = px.pie(pie_df, values="Count", names="Label", title="Candidates")
    pie_fig.update_traces(hoverinfo="label+percent", textinfo="value")
    # fig.show()
    return pie_df, pie_fig
