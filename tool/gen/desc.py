def generate_desc(i, sub_data, name="result"):
    dt = sub_data["DateTime"]
    # inputs = sub_data["InputNodes"]
    num_inputs = int(sub_data["#InputNodes"])
    # outputs = sub_data["OutputNodes"]
    num_outputs = int(sub_data["#OutputNodes"])
    desc = f"DateTiome: {dt}, Sub: {i}, Name: {name}, Inputs: {num_inputs}, Outputs: {num_outputs}"
    return desc
