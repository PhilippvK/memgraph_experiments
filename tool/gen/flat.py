from .flat_utils import FlatCodeEmitter  # TODO: move


def generate_flat_code(stmts, desc=None):
    codes = []
    if desc:
        header = f"// {desc}"
        codes.append(header)
    emitter = FlatCodeEmitter()
    emitter.visit(stmts)
    output = emitter.output
    codes += output.split("\n")
    code = "\n".join(codes) + "\n"
    return code
