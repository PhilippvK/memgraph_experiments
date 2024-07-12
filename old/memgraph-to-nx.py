from gqlalchemy.transformations.translators.nx_translator import NxTranslator
import networkx as nx
from networkx.drawing.nx_agraph import write_dot

translator = NxTranslator()
graph = translator.get_instance()

print(graph.number_of_edges())
print(graph.number_of_nodes())

G = graph

# nx.write_latex(G, "just_my_figure.tex", as_document=True)
write_dot(G, "out.dot")
