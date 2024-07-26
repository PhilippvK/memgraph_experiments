# Warning: This is too restrictive (does not allow any overlaps in candidate nodes as well als all iso nodes)

import ast
import pandas as pd

df = pd.read_csv("out-abc-20240723T110855-temp/subs.csv")

df["Isos Weight"] = (df["#Isos"] + 1) * df["Weight"]
df["Isos Weight NO"] = (df["#IsosNO"] + 1) * df["Weight"]

df.sort_values("Isos Weight", ascending=False, inplace=True)

df["IsosNO"] = df["IsosNO"].apply(lambda x: ast.literal_eval(x) if x != "set()" else set())
df["Nodes"] = df["Nodes"].apply(lambda x: ast.literal_eval(x))

df_ = df[df["#IsosNO"] > 0][["Nodes", "Instrs", "IsosNO", "#Isos", "#IsosNO", "Isos Weight", "Isos Weight NO"]]
df_["IsosNO"] = df_["IsosNO"].apply(lambda x: ast.literal_eval(x) if x != "set()" else set())
df_["Nodes"] = df_["Nodes"].apply(lambda x: ast.literal_eval(x))

final_selection, final_selection_isos, final_weight = set(), set(), 0
for index, row in df_.iterrows():
    if len(final_selection) == 0:
        final_selection.add(index)
        for iso in row["IsosNO"]:
            final_selection_isos.add(iso)
        final_weight += row["Isos Weight NO"]
    else:
        ol = any(check_overlap(row["Nodes"], df_.loc[tmp, "Nodes"]) for tmp in final_selection)
        if not ol:
            ol = any(check_overlap(row["Nodes"], df.loc[tmp, "Nodes"]) for tmp in final_selection_isos)
        if not ol:
            for iso in row["IsosNO"]:
                ol = any(check_overlap(df.loc[iso, "Nodes"], df.loc[tmp, "Nodes"]) for tmp in final_selection)
                if ol:
                    break
                ol = any(check_overlap(df.loc[iso, "Nodes"], df.loc[tmp, "Nodes"]) for tmp in final_selection_isos)
                if ol:
                    break
        if not ol:
            final_selection.add(index)
            final_weight += row["Isos Weight NO"]
