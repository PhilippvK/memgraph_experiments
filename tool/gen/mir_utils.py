def gen_mir_func(func_name, code, desc=None):
    ret = ""
    if desc:
        ret += f"# {desc}\n\n"
    ret += f"""
---
name: {func_name}
body: |
  bb.0:
"""
    ret += "\n".join(["    " + line for line in code.splitlines()])
    return ret
