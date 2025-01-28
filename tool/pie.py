import plotly.express as px

from .enums import ExportFilter, Variation


def pie_name_helper(x):
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
    if x & ExportFilter.FILTERED_MEM:
        return "Filtered (Mem)"
    if x & ExportFilter.FILTERED_BRANCH:
        return "Filtered (Branch)"
    if x & ExportFilter.FILTERED_ENC:
        return "Filtered (Enc)"
    if x & ExportFilter.FILTERED_OPERANDS:
        return "Filtered (Operands)"
    if x & ExportFilter.FILTERED_WEIGHTS:
        return "Filtered (Weights)"
    if x & ExportFilter.INVALID:
        return "Invalid"
    if x & ExportFilter.ERROR:
        return "ERROR"
    return "Unknown"


def pie_name_helper2(x):
    if x & Variation.REUSE_IO:
        return "Variation (ReuseIO)"
    if x & Variation.REG2IMM:
        return "Variation (Reg2Imm)"
    if x & Variation.CONST2IMM:
        return "Variation (Const2Imm)"
    if x & Variation.CONST2REG:
        return "Variation (Const2Reg)"
    return "Unknown"


def generate_pie_chart(subs_df):

    subs_df["Label"] = subs_df["Status"].apply(pie_name_helper)
    pie_df = subs_df.value_counts("Label").rename_axis("Label").reset_index(name="Count")
    # print("pie_df")
    # print(pie_df)
    pie_fig = px.pie(pie_df, values="Count", names="Label", title="Candidates Status")
    pie_fig.update_traces(hoverinfo="label+percent", textinfo="value")
    # fig.show()
    return pie_df, pie_fig


def generate_pie2_chart(subs_df):
    subs_df["Label"] = subs_df["Variations"].apply(pie_name_helper2)
    pie_df = subs_df.value_counts("Label").rename_axis("Label").reset_index(name="Count")
    # print("pie_df")
    # print(pie_df)
    pie_fig = px.pie(pie_df, values="Count", names="Label", title="Candidates Source")
    pie_fig.update_traces(hoverinfo="label+percent", textinfo="value")
    # fig.show()
    return pie_df, pie_fig
