import unittest
from types import SimpleNamespace
import networkx as nx
import pandas as pd

from tool.gen.cdsl_utils import CDSLEmitter
from tool.gen.tree_utils import Operation, Ref, Constant, Statements
from tool.memory import LOADS, STORES, unsupported_memory
from tool.enums import ExportFilter
from tool.stages.filter_subs import filter_subs


class MemoryGenerationTest(unittest.TestCase):
    def emit(self, op, xlen=32):
        children=[Ref('base'),Constant(-4)]
        if op in STORES:
            children.insert(0,Ref('value'))
        emitter=CDSLEmitter(xlen)
        emitter.visit(Statements(children=[Operation(op,children=children)]))
        return emitter.output

    def test_load_width_and_extension(self):
        for op,(width,signed) in LOADS.items():
            xlen=64 if op in ('LWU','LD') else 32
            with self.subTest(op=op):
                text=self.emit(op,xlen)
                self.assertIn(f"({'signed' if signed else 'unsigned'}<{width}>)",text)
                suffix=f'+{width//8-1}:_isaac_mem_addr_0' if width>8 else ''
                self.assertIn(f'MEM[_isaac_mem_addr_0{suffix}]',text)
                self.assertEqual(text.count('MEM['),1)
                self.assertIn(f'(unsigned<{xlen}>)(base+(signed<{xlen}>)((-4)))',text)

    def test_store_width(self):
        for op,width in STORES.items():
            with self.subTest(op=op):
                text=self.emit(op,64 if width==64 else 32)
                self.assertIn(f'=(unsigned<{width}>)(value);',text)
                self.assertNotIn('X[',text)

    def test_invalid_xlen(self):
        for op in ('LWU','LD','SD'):
            with self.subTest(op=op),self.assertRaises(ValueError):
                self.emit(op,32)

    def test_special_memory(self):
        for key in ('isVolatile','isAtomic','isOrdered','volatile','atomic'):
            self.assertTrue(unsupported_memory({key:True}))
        for text in ('load volatile','store atomic','load acquire','seq_cst'):
            self.assertTrue(unsupported_memory({'inst':text}))
        self.assertFalse(unsupported_memory({'inst':'LW %0, %1, 0'}))

    def test_generic_memory_needs_width(self):
        with self.assertRaises(ValueError):
            self.emit('G_LOAD')

    def test_candidate_filter(self):
        settings=SimpleNamespace(progress=False,filters=SimpleNamespace(
            min_inputs=1,max_inputs=5,min_outputs=0,max_outputs=1,
            min_nodes=1,max_nodes=10,instr_predicates=511,allowed_reg_classes=3,
            max_loads=1,max_stores=1,max_mems=1,max_branches=0))
        for names,special,branches,valid in (
            (['LW'],{},0,True), (['SW'],{},0,True),
            (['LW'],{'isVolatile':True},0,False),
            (['LW'],{'inst':'load atomic'},0,False),
            (['G_LOAD'],{},0,False), (['FLW'],{},0,False),
            (['LW','SW'],{},0,False), (['LW'],{},1,False),
        ):
            with self.subTest(names=names,special=special,branches=branches):
                graph=nx.DiGraph()
                for i,name in enumerate(names):
                    graph.add_node(i,properties=dict(name=name,mayLoad=name!='SW',mayStore=name=='SW',**special))
                df=pd.DataFrame([{'#InputNodes':2,'#OutputNodes':1,'#Branches':branches,
                    'BranchNodes':[],'Predicates':0,'Registers':2,'#Loads':int('LW' in names),
                    '#Stores':int('SW' in names),'#Mems':len(names),'Status':ExportFilter.SELECTED.value}])
                filter_subs(settings,[graph],df,[])
                self.assertEqual(df.iloc[0]['Status'],
                    ExportFilter.SELECTED.value if valid else ExportFilter.FILTERED_MEM.value)
