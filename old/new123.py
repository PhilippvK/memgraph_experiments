from neo4j import GraphDatabase
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import matplotlib.pyplot as plt

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("", ""))

query_func = """
MATCH p0=(n00)-[r01:DFG]->(n01)
WHERE n00.func_name = 'tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_per_axis_add_clip_cast_subtract_1_compute_'
RETURN p0;
"""
query = """
MATCH p0=(n00)-[r01:DFG]->(n01)-[r02:DFG]->(n02)-[r03:DFG]->(n03)-[r04:DFG]->(n04)-[r05:DFG]->(n05)-[r06:DFG]->(n06)-[r07:DFG]->(n07) MATCH p1=(n00)-[r11:DFG]->(n11)-[r12:DFG]->(n12)-[r13:DFG]->(n13)-[r14:DFG]->(n14)-[r15:DFG]->(n15)-[r16:DFG]->(n16)-[r17:DFG]->(n17) MATCH p2=(n00)-[r21:DFG]->(n21)-[r22:DFG]->(n22)-[r23:DFG]->(n23)-[r24:DFG]->(n24)-[r25:DFG]->(n25)-[r26:DFG]->(n26)-[r27:DFG]->(n27) MATCH p3=(n00)-[r31:DFG]->(n31)-[r32:DFG]->(n32)-[r33:DFG]->(n33)-[r34:DFG]->(n34)-[r35:DFG]->(n35)-[r36:DFG]->(n36)-[r37:DFG]->(n37) WHERE (n00.name != 'PHI' AND n01.name != 'PHI' AND n02.name != 'PHI' AND n03.name != 'PHI' AND n04.name != 'PHI' AND n05.name != 'PHI' AND n06.name != 'PHI' AND n07.name != 'PHI' AND n11.name != 'PHI' AND n12.name != 'PHI' AND n13.name != 'PHI' AND n14.name != 'PHI' AND n15.name != 'PHI' AND n16.name != 'PHI' AND n17.name != 'PHI' AND n21.name != 'PHI' AND n22.name != 'PHI' AND n23.name != 'PHI' AND n24.name != 'PHI' AND n25.name != 'PHI' AND n26.name != 'PHI' AND n27.name != 'PHI' AND n31.name != 'PHI' AND n32.name != 'PHI' AND n33.name != 'PHI' AND n34.name != 'PHI' AND n35.name != 'PHI' AND n36.name != 'PHI' AND n37.name != 'PHI' AND n00.name != 'COPY' AND n01.name != 'COPY' AND n02.name != 'COPY' AND n03.name != 'COPY' AND n04.name != 'COPY' AND n05.name != 'COPY' AND n06.name != 'COPY' AND n07.name != 'COPY' AND n11.name != 'COPY' AND n12.name != 'COPY' AND n13.name != 'COPY' AND n14.name != 'COPY' AND n15.name != 'COPY' AND n16.name != 'COPY' AND n17.name != 'COPY' AND n21.name != 'COPY' AND n22.name != 'COPY' AND n23.name != 'COPY' AND n24.name != 'COPY' AND n25.name != 'COPY' AND n26.name != 'COPY' AND n27.name != 'COPY' AND n31.name != 'COPY' AND n32.name != 'COPY' AND n33.name != 'COPY' AND n34.name != 'COPY' AND n35.name != 'COPY' AND n36.name != 'COPY' AND n37.name != 'COPY' AND n00.name != 'Const' AND n01.name != 'Const' AND n02.name != 'Const' AND n03.name != 'Const' AND n04.name != 'Const' AND n05.name != 'Const' AND n06.name != 'Const' AND n07.name != 'Const' AND n11.name != 'Const' AND n12.name != 'Const' AND n13.name != 'Const' AND n14.name != 'Const' AND n15.name != 'Const' AND n16.name != 'Const' AND n17.name != 'Const' AND n21.name != 'Const' AND n22.name != 'Const' AND n23.name != 'Const' AND n24.name != 'Const' AND n25.name != 'Const' AND n26.name != 'Const' AND n27.name != 'Const' AND n31.name != 'Const' AND n32.name != 'Const' AND n33.name != 'Const' AND n34.name != 'Const' AND n35.name != 'Const' AND n36.name != 'Const' AND n37.name != 'Const' AND n00.name != '$x0' AND n01.name != '$x0' AND n02.name != '$x0' AND n03.name != '$x0' AND n04.name != '$x0' AND n05.name != '$x0' AND n06.name != '$x0' AND n07.name != '$x0' AND n11.name != '$x0' AND n12.name != '$x0' AND n13.name != '$x0' AND n14.name != '$x0' AND n15.name != '$x0' AND n16.name != '$x0' AND n17.name != '$x0' AND n21.name != '$x0' AND n22.name != '$x0' AND n23.name != '$x0' AND n24.name != '$x0' AND n25.name != '$x0' AND n26.name != '$x0' AND n27.name != '$x0' AND n31.name != '$x0' AND n32.name != '$x0' AND n33.name != '$x0' AND n34.name != '$x0' AND n35.name != '$x0' AND n36.name != '$x0' AND n37.name != '$x0' AND n00.name != '$x1' AND n01.name != '$x1' AND n02.name != '$x1' AND n03.name != '$x1' AND n04.name != '$x1' AND n05.name != '$x1' AND n06.name != '$x1' AND n07.name != '$x1' AND n11.name != '$x1' AND n12.name != '$x1' AND n13.name != '$x1' AND n14.name != '$x1' AND n15.name != '$x1' AND n16.name != '$x1' AND n17.name != '$x1' AND n21.name != '$x1' AND n22.name != '$x1' AND n23.name != '$x1' AND n24.name != '$x1' AND n25.name != '$x1' AND n26.name != '$x1' AND n27.name != '$x1' AND n31.name != '$x1' AND n32.name != '$x1' AND n33.name != '$x1' AND n34.name != '$x1' AND n35.name != '$x1' AND n36.name != '$x1' AND n37.name != '$x1' AND n00.name != '$x2' AND n01.name != '$x2' AND n02.name != '$x2' AND n03.name != '$x2' AND n04.name != '$x2' AND n05.name != '$x2' AND n06.name != '$x2' AND n07.name != '$x2' AND n11.name != '$x2' AND n12.name != '$x2' AND n13.name != '$x2' AND n14.name != '$x2' AND n15.name != '$x2' AND n16.name != '$x2' AND n17.name != '$x2' AND n21.name != '$x2' AND n22.name != '$x2' AND n23.name != '$x2' AND n24.name != '$x2' AND n25.name != '$x2' AND n26.name != '$x2' AND n27.name != '$x2' AND n31.name != '$x2' AND n32.name != '$x2' AND n33.name != '$x2' AND n34.name != '$x2' AND n35.name != '$x2' AND n36.name != '$x2' AND n37.name != '$x2' AND n00.name != '$x3' AND n01.name != '$x3' AND n02.name != '$x3' AND n03.name != '$x3' AND n04.name != '$x3' AND n05.name != '$x3' AND n06.name != '$x3' AND n07.name != '$x3' AND n11.name != '$x3' AND n12.name != '$x3' AND n13.name != '$x3' AND n14.name != '$x3' AND n15.name != '$x3' AND n16.name != '$x3' AND n17.name != '$x3' AND n21.name != '$x3' AND n22.name != '$x3' AND n23.name != '$x3' AND n24.name != '$x3' AND n25.name != '$x3' AND n26.name != '$x3' AND n27.name != '$x3' AND n31.name != '$x3' AND n32.name != '$x3' AND n33.name != '$x3' AND n34.name != '$x3' AND n35.name != '$x3' AND n36.name != '$x3' AND n37.name != '$x3' AND n00.name != '$x4' AND n01.name != '$x4' AND n02.name != '$x4' AND n03.name != '$x4' AND n04.name != '$x4' AND n05.name != '$x4' AND n06.name != '$x4' AND n07.name != '$x4' AND n11.name != '$x4' AND n12.name != '$x4' AND n13.name != '$x4' AND n14.name != '$x4' AND n15.name != '$x4' AND n16.name != '$x4' AND n17.name != '$x4' AND n21.name != '$x4' AND n22.name != '$x4' AND n23.name != '$x4' AND n24.name != '$x4' AND n25.name != '$x4' AND n26.name != '$x4' AND n27.name != '$x4' AND n31.name != '$x4' AND n32.name != '$x4' AND n33.name != '$x4' AND n34.name != '$x4' AND n35.name != '$x4' AND n36.name != '$x4' AND n37.name != '$x4' AND n00.name != '$x5' AND n01.name != '$x5' AND n02.name != '$x5' AND n03.name != '$x5' AND n04.name != '$x5' AND n05.name != '$x5' AND n06.name != '$x5' AND n07.name != '$x5' AND n11.name != '$x5' AND n12.name != '$x5' AND n13.name != '$x5' AND n14.name != '$x5' AND n15.name != '$x5' AND n16.name != '$x5' AND n17.name != '$x5' AND n21.name != '$x5' AND n22.name != '$x5' AND n23.name != '$x5' AND n24.name != '$x5' AND n25.name != '$x5' AND n26.name != '$x5' AND n27.name != '$x5' AND n31.name != '$x5' AND n32.name != '$x5' AND n33.name != '$x5' AND n34.name != '$x5' AND n35.name != '$x5' AND n36.name != '$x5' AND n37.name != '$x5' AND n00.name != '$x6' AND n01.name != '$x6' AND n02.name != '$x6' AND n03.name != '$x6' AND n04.name != '$x6' AND n05.name != '$x6' AND n06.name != '$x6' AND n07.name != '$x6' AND n11.name != '$x6' AND n12.name != '$x6' AND n13.name != '$x6' AND n14.name != '$x6' AND n15.name != '$x6' AND n16.name != '$x6' AND n17.name != '$x6' AND n21.name != '$x6' AND n22.name != '$x6' AND n23.name != '$x6' AND n24.name != '$x6' AND n25.name != '$x6' AND n26.name != '$x6' AND n27.name != '$x6' AND n31.name != '$x6' AND n32.name != '$x6' AND n33.name != '$x6' AND n34.name != '$x6' AND n35.name != '$x6' AND n36.name != '$x6' AND n37.name != '$x6' AND n00.name != '$x7' AND n01.name != '$x7' AND n02.name != '$x7' AND n03.name != '$x7' AND n04.name != '$x7' AND n05.name != '$x7' AND n06.name != '$x7' AND n07.name != '$x7' AND n11.name != '$x7' AND n12.name != '$x7' AND n13.name != '$x7' AND n14.name != '$x7' AND n15.name != '$x7' AND n16.name != '$x7' AND n17.name != '$x7' AND n21.name != '$x7' AND n22.name != '$x7' AND n23.name != '$x7' AND n24.name != '$x7' AND n25.name != '$x7' AND n26.name != '$x7' AND n27.name != '$x7' AND n31.name != '$x7' AND n32.name != '$x7' AND n33.name != '$x7' AND n34.name != '$x7' AND n35.name != '$x7' AND n36.name != '$x7' AND n37.name != '$x7' AND n00.name != '$x8' AND n01.name != '$x8' AND n02.name != '$x8' AND n03.name != '$x8' AND n04.name != '$x8' AND n05.name != '$x8' AND n06.name != '$x8' AND n07.name != '$x8' AND n11.name != '$x8' AND n12.name != '$x8' AND n13.name != '$x8' AND n14.name != '$x8' AND n15.name != '$x8' AND n16.name != '$x8' AND n17.name != '$x8' AND n21.name != '$x8' AND n22.name != '$x8' AND n23.name != '$x8' AND n24.name != '$x8' AND n25.name != '$x8' AND n26.name != '$x8' AND n27.name != '$x8' AND n31.name != '$x8' AND n32.name != '$x8' AND n33.name != '$x8' AND n34.name != '$x8' AND n35.name != '$x8' AND n36.name != '$x8' AND n37.name != '$x8' AND n00.name != '$x9' AND n01.name != '$x9' AND n02.name != '$x9' AND n03.name != '$x9' AND n04.name != '$x9' AND n05.name != '$x9' AND n06.name != '$x9' AND n07.name != '$x9' AND n11.name != '$x9' AND n12.name != '$x9' AND n13.name != '$x9' AND n14.name != '$x9' AND n15.name != '$x9' AND n16.name != '$x9' AND n17.name != '$x9' AND n21.name != '$x9' AND n22.name != '$x9' AND n23.name != '$x9' AND n24.name != '$x9' AND n25.name != '$x9' AND n26.name != '$x9' AND n27.name != '$x9' AND n31.name != '$x9' AND n32.name != '$x9' AND n33.name != '$x9' AND n34.name != '$x9' AND n35.name != '$x9' AND n36.name != '$x9' AND n37.name != '$x9' AND n00.name != '$x10' AND n01.name != '$x10' AND n02.name != '$x10' AND n03.name != '$x10' AND n04.name != '$x10' AND n05.name != '$x10' AND n06.name != '$x10' AND n07.name != '$x10' AND n11.name != '$x10' AND n12.name != '$x10' AND n13.name != '$x10' AND n14.name != '$x10' AND n15.name != '$x10' AND n16.name != '$x10' AND n17.name != '$x10' AND n21.name != '$x10' AND n22.name != '$x10' AND n23.name != '$x10' AND n24.name != '$x10' AND n25.name != '$x10' AND n26.name != '$x10' AND n27.name != '$x10' AND n31.name != '$x10' AND n32.name != '$x10' AND n33.name != '$x10' AND n34.name != '$x10' AND n35.name != '$x10' AND n36.name != '$x10' AND n37.name != '$x10' AND n00.name != '$x11' AND n01.name != '$x11' AND n02.name != '$x11' AND n03.name != '$x11' AND n04.name != '$x11' AND n05.name != '$x11' AND n06.name != '$x11' AND n07.name != '$x11' AND n11.name != '$x11' AND n12.name != '$x11' AND n13.name != '$x11' AND n14.name != '$x11' AND n15.name != '$x11' AND n16.name != '$x11' AND n17.name != '$x11' AND n21.name != '$x11' AND n22.name != '$x11' AND n23.name != '$x11' AND n24.name != '$x11' AND n25.name != '$x11' AND n26.name != '$x11' AND n27.name != '$x11' AND n31.name != '$x11' AND n32.name != '$x11' AND n33.name != '$x11' AND n34.name != '$x11' AND n35.name != '$x11' AND n36.name != '$x11' AND n37.name != '$x11' AND n00.name != '$x12' AND n01.name != '$x12' AND n02.name != '$x12' AND n03.name != '$x12' AND n04.name != '$x12' AND n05.name != '$x12' AND n06.name != '$x12' AND n07.name != '$x12' AND n11.name != '$x12' AND n12.name != '$x12' AND n13.name != '$x12' AND n14.name != '$x12' AND n15.name != '$x12' AND n16.name != '$x12' AND n17.name != '$x12' AND n21.name != '$x12' AND n22.name != '$x12' AND n23.name != '$x12' AND n24.name != '$x12' AND n25.name != '$x12' AND n26.name != '$x12' AND n27.name != '$x12' AND n31.name != '$x12' AND n32.name != '$x12' AND n33.name != '$x12' AND n34.name != '$x12' AND n35.name != '$x12' AND n36.name != '$x12' AND n37.name != '$x12' AND n00.name != '$x13' AND n01.name != '$x13' AND n02.name != '$x13' AND n03.name != '$x13' AND n04.name != '$x13' AND n05.name != '$x13' AND n06.name != '$x13' AND n07.name != '$x13' AND n11.name != '$x13' AND n12.name != '$x13' AND n13.name != '$x13' AND n14.name != '$x13' AND n15.name != '$x13' AND n16.name != '$x13' AND n17.name != '$x13' AND n21.name != '$x13' AND n22.name != '$x13' AND n23.name != '$x13' AND n24.name != '$x13' AND n25.name != '$x13' AND n26.name != '$x13' AND n27.name != '$x13' AND n31.name != '$x13' AND n32.name != '$x13' AND n33.name != '$x13' AND n34.name != '$x13' AND n35.name != '$x13' AND n36.name != '$x13' AND n37.name != '$x13' AND n00.name != '$x14' AND n01.name != '$x14' AND n02.name != '$x14' AND n03.name != '$x14' AND n04.name != '$x14' AND n05.name != '$x14' AND n06.name != '$x14' AND n07.name != '$x14' AND n11.name != '$x14' AND n12.name != '$x14' AND n13.name != '$x14' AND n14.name != '$x14' AND n15.name != '$x14' AND n16.name != '$x14' AND n17.name != '$x14' AND n21.name != '$x14' AND n22.name != '$x14' AND n23.name != '$x14' AND n24.name != '$x14' AND n25.name != '$x14' AND n26.name != '$x14' AND n27.name != '$x14' AND n31.name != '$x14' AND n32.name != '$x14' AND n33.name != '$x14' AND n34.name != '$x14' AND n35.name != '$x14' AND n36.name != '$x14' AND n37.name != '$x14' AND n00.name != '$x15' AND n01.name != '$x15' AND n02.name != '$x15' AND n03.name != '$x15' AND n04.name != '$x15' AND n05.name != '$x15' AND n06.name != '$x15' AND n07.name != '$x15' AND n11.name != '$x15' AND n12.name != '$x15' AND n13.name != '$x15' AND n14.name != '$x15' AND n15.name != '$x15' AND n16.name != '$x15' AND n17.name != '$x15' AND n21.name != '$x15' AND n22.name != '$x15' AND n23.name != '$x15' AND n24.name != '$x15' AND n25.name != '$x15' AND n26.name != '$x15' AND n27.name != '$x15' AND n31.name != '$x15' AND n32.name != '$x15' AND n33.name != '$x15' AND n34.name != '$x15' AND n35.name != '$x15' AND n36.name != '$x15' AND n37.name != '$x15' AND n00.name != '$x16' AND n01.name != '$x16' AND n02.name != '$x16' AND n03.name != '$x16' AND n04.name != '$x16' AND n05.name != '$x16' AND n06.name != '$x16' AND n07.name != '$x16' AND n11.name != '$x16' AND n12.name != '$x16' AND n13.name != '$x16' AND n14.name != '$x16' AND n15.name != '$x16' AND n16.name != '$x16' AND n17.name != '$x16' AND n21.name != '$x16' AND n22.name != '$x16' AND n23.name != '$x16' AND n24.name != '$x16' AND n25.name != '$x16' AND n26.name != '$x16' AND n27.name != '$x16' AND n31.name != '$x16' AND n32.name != '$x16' AND n33.name != '$x16' AND n34.name != '$x16' AND n35.name != '$x16' AND n36.name != '$x16' AND n37.name != '$x16' AND n00.name != '$x17' AND n01.name != '$x17' AND n02.name != '$x17' AND n03.name != '$x17' AND n04.name != '$x17' AND n05.name != '$x17' AND n06.name != '$x17' AND n07.name != '$x17' AND n11.name != '$x17' AND n12.name != '$x17' AND n13.name != '$x17' AND n14.name != '$x17' AND n15.name != '$x17' AND n16.name != '$x17' AND n17.name != '$x17' AND n21.name != '$x17' AND n22.name != '$x17' AND n23.name != '$x17' AND n24.name != '$x17' AND n25.name != '$x17' AND n26.name != '$x17' AND n27.name != '$x17' AND n31.name != '$x17' AND n32.name != '$x17' AND n33.name != '$x17' AND n34.name != '$x17' AND n35.name != '$x17' AND n36.name != '$x17' AND n37.name != '$x17' AND n00.name != '$x18' AND n01.name != '$x18' AND n02.name != '$x18' AND n03.name != '$x18' AND n04.name != '$x18' AND n05.name != '$x18' AND n06.name != '$x18' AND n07.name != '$x18' AND n11.name != '$x18' AND n12.name != '$x18' AND n13.name != '$x18' AND n14.name != '$x18' AND n15.name != '$x18' AND n16.name != '$x18' AND n17.name != '$x18' AND n21.name != '$x18' AND n22.name != '$x18' AND n23.name != '$x18' AND n24.name != '$x18' AND n25.name != '$x18' AND n26.name != '$x18' AND n27.name != '$x18' AND n31.name != '$x18' AND n32.name != '$x18' AND n33.name != '$x18' AND n34.name != '$x18' AND n35.name != '$x18' AND n36.name != '$x18' AND n37.name != '$x18' AND n00.name != '$x19' AND n01.name != '$x19' AND n02.name != '$x19' AND n03.name != '$x19' AND n04.name != '$x19' AND n05.name != '$x19' AND n06.name != '$x19' AND n07.name != '$x19' AND n11.name != '$x19' AND n12.name != '$x19' AND n13.name != '$x19' AND n14.name != '$x19' AND n15.name != '$x19' AND n16.name != '$x19' AND n17.name != '$x19' AND n21.name != '$x19' AND n22.name != '$x19' AND n23.name != '$x19' AND n24.name != '$x19' AND n25.name != '$x19' AND n26.name != '$x19' AND n27.name != '$x19' AND n31.name != '$x19' AND n32.name != '$x19' AND n33.name != '$x19' AND n34.name != '$x19' AND n35.name != '$x19' AND n36.name != '$x19' AND n37.name != '$x19' AND n00.name != '$x20' AND n01.name != '$x20' AND n02.name != '$x20' AND n03.name != '$x20' AND n04.name != '$x20' AND n05.name != '$x20' AND n06.name != '$x20' AND n07.name != '$x20' AND n11.name != '$x20' AND n12.name != '$x20' AND n13.name != '$x20' AND n14.name != '$x20' AND n15.name != '$x20' AND n16.name != '$x20' AND n17.name != '$x20' AND n21.name != '$x20' AND n22.name != '$x20' AND n23.name != '$x20' AND n24.name != '$x20' AND n25.name != '$x20' AND n26.name != '$x20' AND n27.name != '$x20' AND n31.name != '$x20' AND n32.name != '$x20' AND n33.name != '$x20' AND n34.name != '$x20' AND n35.name != '$x20' AND n36.name != '$x20' AND n37.name != '$x20' AND n00.name != '$x21' AND n01.name != '$x21' AND n02.name != '$x21' AND n03.name != '$x21' AND n04.name != '$x21' AND n05.name != '$x21' AND n06.name != '$x21' AND n07.name != '$x21' AND n11.name != '$x21' AND n12.name != '$x21' AND n13.name != '$x21' AND n14.name != '$x21' AND n15.name != '$x21' AND n16.name != '$x21' AND n17.name != '$x21' AND n21.name != '$x21' AND n22.name != '$x21' AND n23.name != '$x21' AND n24.name != '$x21' AND n25.name != '$x21' AND n26.name != '$x21' AND n27.name != '$x21' AND n31.name != '$x21' AND n32.name != '$x21' AND n33.name != '$x21' AND n34.name != '$x21' AND n35.name != '$x21' AND n36.name != '$x21' AND n37.name != '$x21' AND n00.name != '$x22' AND n01.name != '$x22' AND n02.name != '$x22' AND n03.name != '$x22' AND n04.name != '$x22' AND n05.name != '$x22' AND n06.name != '$x22' AND n07.name != '$x22' AND n11.name != '$x22' AND n12.name != '$x22' AND n13.name != '$x22' AND n14.name != '$x22' AND n15.name != '$x22' AND n16.name != '$x22' AND n17.name != '$x22' AND n21.name != '$x22' AND n22.name != '$x22' AND n23.name != '$x22' AND n24.name != '$x22' AND n25.name != '$x22' AND n26.name != '$x22' AND n27.name != '$x22' AND n31.name != '$x22' AND n32.name != '$x22' AND n33.name != '$x22' AND n34.name != '$x22' AND n35.name != '$x22' AND n36.name != '$x22' AND n37.name != '$x22' AND n00.name != '$x23' AND n01.name != '$x23' AND n02.name != '$x23' AND n03.name != '$x23' AND n04.name != '$x23' AND n05.name != '$x23' AND n06.name != '$x23' AND n07.name != '$x23' AND n11.name != '$x23' AND n12.name != '$x23' AND n13.name != '$x23' AND n14.name != '$x23' AND n15.name != '$x23' AND n16.name != '$x23' AND n17.name != '$x23' AND n21.name != '$x23' AND n22.name != '$x23' AND n23.name != '$x23' AND n24.name != '$x23' AND n25.name != '$x23' AND n26.name != '$x23' AND n27.name != '$x23' AND n31.name != '$x23' AND n32.name != '$x23' AND n33.name != '$x23' AND n34.name != '$x23' AND n35.name != '$x23' AND n36.name != '$x23' AND n37.name != '$x23' AND n00.name != '$x24' AND n01.name != '$x24' AND n02.name != '$x24' AND n03.name != '$x24' AND n04.name != '$x24' AND n05.name != '$x24' AND n06.name != '$x24' AND n07.name != '$x24' AND n11.name != '$x24' AND n12.name != '$x24' AND n13.name != '$x24' AND n14.name != '$x24' AND n15.name != '$x24' AND n16.name != '$x24' AND n17.name != '$x24' AND n21.name != '$x24' AND n22.name != '$x24' AND n23.name != '$x24' AND n24.name != '$x24' AND n25.name != '$x24' AND n26.name != '$x24' AND n27.name != '$x24' AND n31.name != '$x24' AND n32.name != '$x24' AND n33.name != '$x24' AND n34.name != '$x24' AND n35.name != '$x24' AND n36.name != '$x24' AND n37.name != '$x24' AND n00.name != '$x25' AND n01.name != '$x25' AND n02.name != '$x25' AND n03.name != '$x25' AND n04.name != '$x25' AND n05.name != '$x25' AND n06.name != '$x25' AND n07.name != '$x25' AND n11.name != '$x25' AND n12.name != '$x25' AND n13.name != '$x25' AND n14.name != '$x25' AND n15.name != '$x25' AND n16.name != '$x25' AND n17.name != '$x25' AND n21.name != '$x25' AND n22.name != '$x25' AND n23.name != '$x25' AND n24.name != '$x25' AND n25.name != '$x25' AND n26.name != '$x25' AND n27.name != '$x25' AND n31.name != '$x25' AND n32.name != '$x25' AND n33.name != '$x25' AND n34.name != '$x25' AND n35.name != '$x25' AND n36.name != '$x25' AND n37.name != '$x25' AND n00.name != '$x26' AND n01.name != '$x26' AND n02.name != '$x26' AND n03.name != '$x26' AND n04.name != '$x26' AND n05.name != '$x26' AND n06.name != '$x26' AND n07.name != '$x26' AND n11.name != '$x26' AND n12.name != '$x26' AND n13.name != '$x26' AND n14.name != '$x26' AND n15.name != '$x26' AND n16.name != '$x26' AND n17.name != '$x26' AND n21.name != '$x26' AND n22.name != '$x26' AND n23.name != '$x26' AND n24.name != '$x26' AND n25.name != '$x26' AND n26.name != '$x26' AND n27.name != '$x26' AND n31.name != '$x26' AND n32.name != '$x26' AND n33.name != '$x26' AND n34.name != '$x26' AND n35.name != '$x26' AND n36.name != '$x26' AND n37.name != '$x26' AND n00.name != '$x27' AND n01.name != '$x27' AND n02.name != '$x27' AND n03.name != '$x27' AND n04.name != '$x27' AND n05.name != '$x27' AND n06.name != '$x27' AND n07.name != '$x27' AND n11.name != '$x27' AND n12.name != '$x27' AND n13.name != '$x27' AND n14.name != '$x27' AND n15.name != '$x27' AND n16.name != '$x27' AND n17.name != '$x27' AND n21.name != '$x27' AND n22.name != '$x27' AND n23.name != '$x27' AND n24.name != '$x27' AND n25.name != '$x27' AND n26.name != '$x27' AND n27.name != '$x27' AND n31.name != '$x27' AND n32.name != '$x27' AND n33.name != '$x27' AND n34.name != '$x27' AND n35.name != '$x27' AND n36.name != '$x27' AND n37.name != '$x27' AND n00.name != '$x28' AND n01.name != '$x28' AND n02.name != '$x28' AND n03.name != '$x28' AND n04.name != '$x28' AND n05.name != '$x28' AND n06.name != '$x28' AND n07.name != '$x28' AND n11.name != '$x28' AND n12.name != '$x28' AND n13.name != '$x28' AND n14.name != '$x28' AND n15.name != '$x28' AND n16.name != '$x28' AND n17.name != '$x28' AND n21.name != '$x28' AND n22.name != '$x28' AND n23.name != '$x28' AND n24.name != '$x28' AND n25.name != '$x28' AND n26.name != '$x28' AND n27.name != '$x28' AND n31.name != '$x28' AND n32.name != '$x28' AND n33.name != '$x28' AND n34.name != '$x28' AND n35.name != '$x28' AND n36.name != '$x28' AND n37.name != '$x28' AND n00.name != '$x29' AND n01.name != '$x29' AND n02.name != '$x29' AND n03.name != '$x29' AND n04.name != '$x29' AND n05.name != '$x29' AND n06.name != '$x29' AND n07.name != '$x29' AND n11.name != '$x29' AND n12.name != '$x29' AND n13.name != '$x29' AND n14.name != '$x29' AND n15.name != '$x29' AND n16.name != '$x29' AND n17.name != '$x29' AND n21.name != '$x29' AND n22.name != '$x29' AND n23.name != '$x29' AND n24.name != '$x29' AND n25.name != '$x29' AND n26.name != '$x29' AND n27.name != '$x29' AND n31.name != '$x29' AND n32.name != '$x29' AND n33.name != '$x29' AND n34.name != '$x29' AND n35.name != '$x29' AND n36.name != '$x29' AND n37.name != '$x29' AND n00.name != '$x30' AND n01.name != '$x30' AND n02.name != '$x30' AND n03.name != '$x30' AND n04.name != '$x30' AND n05.name != '$x30' AND n06.name != '$x30' AND n07.name != '$x30' AND n11.name != '$x30' AND n12.name != '$x30' AND n13.name != '$x30' AND n14.name != '$x30' AND n15.name != '$x30' AND n16.name != '$x30' AND n17.name != '$x30' AND n21.name != '$x30' AND n22.name != '$x30' AND n23.name != '$x30' AND n24.name != '$x30' AND n25.name != '$x30' AND n26.name != '$x30' AND n27.name != '$x30' AND n31.name != '$x30' AND n32.name != '$x30' AND n33.name != '$x30' AND n34.name != '$x30' AND n35.name != '$x30' AND n36.name != '$x30' AND n37.name != '$x30' AND n00.name != '$x31' AND n01.name != '$x31' AND n02.name != '$x31' AND n03.name != '$x31' AND n04.name != '$x31' AND n05.name != '$x31' AND n06.name != '$x31' AND n07.name != '$x31' AND n11.name != '$x31' AND n12.name != '$x31' AND n13.name != '$x31' AND n14.name != '$x31' AND n15.name != '$x31' AND n16.name != '$x31' AND n17.name != '$x31' AND n21.name != '$x31' AND n22.name != '$x31' AND n23.name != '$x31' AND n24.name != '$x31' AND n25.name != '$x31' AND n26.name != '$x31' AND n27.name != '$x31' AND n31.name != '$x31' AND n32.name != '$x31' AND n33.name != '$x31' AND n34.name != '$x31' AND n35.name != '$x31' AND n36.name != '$x31' AND n37.name != '$x31' AND n00.name != '$x32' AND n01.name != '$x32' AND n02.name != '$x32' AND n03.name != '$x32' AND n04.name != '$x32' AND n05.name != '$x32' AND n06.name != '$x32' AND n07.name != '$x32' AND n11.name != '$x32' AND n12.name != '$x32' AND n13.name != '$x32' AND n14.name != '$x32' AND n15.name != '$x32' AND n16.name != '$x32' AND n17.name != '$x32' AND n21.name != '$x32' AND n22.name != '$x32' AND n23.name != '$x32' AND n24.name != '$x32' AND n25.name != '$x32' AND n26.name != '$x32' AND n27.name != '$x32' AND n31.name != '$x32' AND n32.name != '$x32' AND n33.name != '$x32' AND n34.name != '$x32' AND n35.name != '$x32' AND n36.name != '$x32' AND n37.name != '$x32' AND n02 != n12 AND n03 != n13 AND n04 != n14 AND n05 != n15 AND n06 != n16 AND n07 != n17 AND p0 != p1 AND n12 != n22 AND n13 != n23 AND n14 != n24 AND n15 != n25 AND n16 != n26 AND n17 != n27 AND p1 != p2 AND n22 != n32 AND n23 != n33 AND n24 != n34 AND n25 != n35 AND n26 != n36 AND n27 != n37 AND p2 != p3 AND (n00.func_name = 'tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_per_axis_add_clip_cast_subtract_1_compute_')) RETURN DISTINCT p0, p1, p2, p3;
"""
query2 = """
MATCH p0=(n00)-[r01:DFG]->(n01) WHERE (n00.name != 'PHI' AND n01.name != 'PHI' AND n00.name != 'COPY' AND n01.name != 'COPY' AND n00.name != 'Const' AND n01.name != 'Const' AND n00.name != '$x0' AND n01.name != '$x0' AND n00.name != '$x1' AND n01.name != '$x1' AND n00.name != '$x2' AND n01.name != '$x2' AND n00.name != '$x3' AND n01.name != '$x3' AND n00.name != '$x4' AND n01.name != '$x4' AND n00.name != '$x5' AND n01.name != '$x5' AND n00.name != '$x6' AND n01.name != '$x6' AND n00.name != '$x7' AND n01.name != '$x7' AND n00.name != '$x8' AND n01.name != '$x8' AND n00.name != '$x9' AND n01.name != '$x9' AND n00.name != '$x10' AND n01.name != '$x10' AND n00.name != '$x11' AND n01.name != '$x11' AND n00.name != '$x12' AND n01.name != '$x12' AND n00.name != '$x13' AND n01.name != '$x13' AND n00.name != '$x14' AND n01.name != '$x14' AND n00.name != '$x15' AND n01.name != '$x15' AND n00.name != '$x16' AND n01.name != '$x16' AND n00.name != '$x17' AND n01.name != '$x17' AND n00.name != '$x18' AND n01.name != '$x18' AND n00.name != '$x19' AND n01.name != '$x19' AND n00.name != '$x20' AND n01.name != '$x20' AND n00.name != '$x21' AND n01.name != '$x21' AND n00.name != '$x22' AND n01.name != '$x22' AND n00.name != '$x23' AND n01.name != '$x23' AND n00.name != '$x24' AND n01.name != '$x24' AND n00.name != '$x25' AND n01.name != '$x25' AND n00.name != '$x26' AND n01.name != '$x26' AND n00.name != '$x27' AND n01.name != '$x27' AND n00.name != '$x28' AND n01.name != '$x28' AND n00.name != '$x29' AND n01.name != '$x29' AND n00.name != '$x30' AND n01.name != '$x30' AND n00.name != '$x31' AND n01.name != '$x31' AND n00.name != '$x32' AND n01.name != '$x32' AND (n00.func_name = 'tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_per_axis_add_clip_cast_subtract_1_compute_')) RETURN p0;
"""
query3 = """
MATCH p0=(n00)-[r01:DFG]->(n01)-[r02:DFG]->(n02) WHERE (n00.name != 'PHI' AND n01.name != 'PHI' AND n02.name != 'PHI' AND n00.name != 'COPY' AND n01.name != 'COPY' AND n02.name != 'COPY' AND n00.name != 'Const' AND n01.name != 'Const' AND n02.name != 'Const' AND n00.name != '$x0' AND n01.name != '$x0' AND n02.name != '$x0' AND n00.name != '$x1' AND n01.name != '$x1' AND n02.name != '$x1' AND n00.name != '$x2' AND n01.name != '$x2' AND n02.name != '$x2' AND n00.name != '$x3' AND n01.name != '$x3' AND n02.name != '$x3' AND n00.name != '$x4' AND n01.name != '$x4' AND n02.name != '$x4' AND n00.name != '$x5' AND n01.name != '$x5' AND n02.name != '$x5' AND n00.name != '$x6' AND n01.name != '$x6' AND n02.name != '$x6' AND n00.name != '$x7' AND n01.name != '$x7' AND n02.name != '$x7' AND n00.name != '$x8' AND n01.name != '$x8' AND n02.name != '$x8' AND n00.name != '$x9' AND n01.name != '$x9' AND n02.name != '$x9' AND n00.name != '$x10' AND n01.name != '$x10' AND n02.name != '$x10' AND n00.name != '$x11' AND n01.name != '$x11' AND n02.name != '$x11' AND n00.name != '$x12' AND n01.name != '$x12' AND n02.name != '$x12' AND n00.name != '$x13' AND n01.name != '$x13' AND n02.name != '$x13' AND n00.name != '$x14' AND n01.name != '$x14' AND n02.name != '$x14' AND n00.name != '$x15' AND n01.name != '$x15' AND n02.name != '$x15' AND n00.name != '$x16' AND n01.name != '$x16' AND n02.name != '$x16' AND n00.name != '$x17' AND n01.name != '$x17' AND n02.name != '$x17' AND n00.name != '$x18' AND n01.name != '$x18' AND n02.name != '$x18' AND n00.name != '$x19' AND n01.name != '$x19' AND n02.name != '$x19' AND n00.name != '$x20' AND n01.name != '$x20' AND n02.name != '$x20' AND n00.name != '$x21' AND n01.name != '$x21' AND n02.name != '$x21' AND n00.name != '$x22' AND n01.name != '$x22' AND n02.name != '$x22' AND n00.name != '$x23' AND n01.name != '$x23' AND n02.name != '$x23' AND n00.name != '$x24' AND n01.name != '$x24' AND n02.name != '$x24' AND n00.name != '$x25' AND n01.name != '$x25' AND n02.name != '$x25' AND n00.name != '$x26' AND n01.name != '$x26' AND n02.name != '$x26' AND n00.name != '$x27' AND n01.name != '$x27' AND n02.name != '$x27' AND n00.name != '$x28' AND n01.name != '$x28' AND n02.name != '$x28' AND n00.name != '$x29' AND n01.name != '$x29' AND n02.name != '$x29' AND n00.name != '$x30' AND n01.name != '$x30' AND n02.name != '$x30' AND n00.name != '$x31' AND n01.name != '$x31' AND n02.name != '$x31' AND n00.name != '$x32' AND n01.name != '$x32' AND n02.name != '$x32' AND (n00.func_name = 'tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_per_axis_add_clip_cast_subtract_1_compute_')) RETURN p0;
"""

func_results = driver.session().run(query_func)
results = driver.session().run(query3)
# print("results", results, dir(results))

GF = nx.MultiDiGraph()
nodes = list(func_results.graph()._nodes.values())
# print("nodes", nodes)
for node in nodes:
    print("node", node)
    if len(node._labels) > 0:
        label = list(node._labels)[0]
    else:
        label = "?!"
    name = node._properties.get("name", "?")
    GF.add_node(node.id, xlabel=label, label=name, properties=node._properties)

rels = list(func_results.graph()._relationships.values())
for rel in rels:
    label = rel.type
    # GF.add_edge(rel.start_node.element_id, rel.end_node.element_id, key=rel.element_id, label=label, type=rel.type, properties=rel._properties)
    GF.add_edge(rel.start_node.id, rel.end_node.id, key=rel.id, label=label, type=rel.type, properties=rel._properties)
print("GF", GF)
write_dot(GF, f"func.dot")
input("?")

G = nx.MultiDiGraph()
nodes = list(results.graph()._nodes.values())
# print("nodes", nodes)
for node in nodes:
    print("node", node)
    if len(node._labels) > 0:
        label = list(node._labels)[0]
    else:
        label = "?!"
    name = node._properties.get("name", "?")
    G.add_node(node.id, xlabel=label, label=name, properties=node._properties)

rels = list(results.graph()._relationships.values())
for rel in rels:
    label = rel.type
    # G.add_edge(rel.start_node.element_id, rel.end_node.element_id, key=rel.element_id, label=label, type=rel.type, properties=rel._properties)
    G.add_edge(rel.start_node.id, rel.end_node.id, key=rel.id, label=label, type=rel.type, properties=rel._properties)
print("G", G)
write_dot(G, f"results.dot")

subs = []
for i, result in enumerate(results):
    # print("result", result, dir(result))
    # print("result.data", result.data())
    # print("result.value", result.value(), dir())
    nodes_ = set()
    # path = result.value()
    for path in result:
        # print("path", path)
        # input("p")
        # print("path", path, dir(path))
        nodes__ = path.nodes
        # print("nodes__", nodes__[0].element_id)
        # 'count', 'data', 'get', 'index', 'items', 'keys', 'value', 'values'
        nodes_ |= {n.id for n in nodes__}
    # print("nodes_", nodes_)
    G_ = G.subgraph(nodes_)
    # G_ = nx.subgraph_view(G, filter_node=lambda x: x in nodes_)
    print("G_", G_)
    write_dot(G_, f"result{i}.dot")
    count = subs.count(G_)
    if count > 0:
        input("2")
    subs.append(G_)

print("GF", GF)
print("GF.nodes", GF.nodes)
mapping = dict(zip(GF.nodes.keys(), range(len(GF.nodes))))
GF = nx.relabel_nodes(GF, mapping)
G = nx.relabel_nodes(G, mapping)
for i in range(len(subs)):
    subs[i] = nx.relabel_nodes(subs[i], mapping)
print("GF", GF)
print("G", G)
print("GF.nodes", GF.nodes)
print("G.nodes", G.nodes)
# mapping = dict(zip(G.nodes.keys(), range(len(G.nodes))))
# G = nx.relabel_nodes(G, mapping)
# for i in range(len(subs)):
#     subs[i] = nx.relabel_nodes(subs[i], mapping)
# print("G", G)
# print("G.nodes", G.nodes)
# topo = list(reversed(list(nx.topological_sort(G))))
topo = list(reversed(list(nx.topological_sort(GF))))
print("topo", topo)
# mapping = dict(zip(G.nodes.keys(), topo))
mapping = dict(zip(GF.nodes.keys(), topo))
GF = nx.relabel_nodes(GF, mapping)
G = nx.relabel_nodes(G, mapping)
for i in range(len(subs)):
    subs[i] = nx.relabel_nodes(subs[i], mapping)
print("GF", GF)
print("G", G)
print("GF.nodes", GF.nodes)
print("G.nodes", G.nodes)

print("subs[0]", subs[0])
print("subs[0].nodes", subs[0].nodes)


def calc_inputs(G, sub):
    print("calc_inputs", sub)
    inputs = []
    ret = 0
    sub_nodes = sub.nodes
    print("sub_nodes", sub_nodes)
    for node in sub_nodes:
        print("node", node, G.nodes[node].get("label"))
        ins = G.in_edges(node)
        print("ins", ins)
        for in_ in ins:
            # print("in_", in_, G.nodes[in_[0]].get("label"))
            src = in_[0]
            print("src", src, G.nodes[src].get("label"))
            # print("src in sub_nodes", src in sub_nodes)
            # print("src not in inputs", src not in inputs)
            if not (src in sub_nodes) and (src not in inputs):
                print("IN")
                ret += 1
                inputs.append(src)
    print("ret", ret)
    return ret


def calc_outputs(G, sub):
    print("calc_outputs", sub)
    ret = 0
    sub_nodes = sub.nodes
    print("sub_nodes", sub_nodes)
    for node in sub_nodes:
        print("node", node, G.nodes[node].get("label"))
        if G.nodes[node]["properties"]["op_type"] == "output":
            # print("A")
            print("OUT2")
            ret += 1
        else:
            # print("B")
            outs = G.out_edges(node)
            print("outs", outs)
            for out_ in outs:
                # print("out_", out_, G.nodes[out_[0]].get("label"))
                dst = out_[1]
                print("dst", dst, G.nodes[dst].get("label"))
                if dst not in sub_nodes:
                    print("OUT")
                    ret += 1
    print("ret", ret)
    # input("1")
    return ret


# if True:
for i, sub in enumerate(subs):
    # i = 3
    sub = subs[i]
    print("i, sub", i, sub)
    num_inputs = calc_inputs(GF, sub)
    num_outputs = calc_outputs(GF, sub)
    print("num_inputs", num_inputs)
    print("num_outputs", num_outputs)
    if num_outputs == 0 or num_inputs == 0:
        input(">?")
    elif num_outputs in [1] and num_inputs in [1, 2, 3]:
        input(">")
