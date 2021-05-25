import json
import pydot
import argparse
import datetime
import os

from onnx import NodeProto
from typing import Callable
from collections import defaultdict


OP_STYLE = {"shape": "box", "color": "#252c36", "fontcolor": "#252c36"}
BLOB_STYLE = {"shape": "octagon", "color": "#ff4971", "fontcolor": "#ff4971"}

_NodeProducer = Callable[[NodeProto, int], pydot.Node]


def _escape_label(name):

    return json.dumps(name)


def _form_and_sanitize_docstring(s):
    url = "javascript:alert("
    url += _escape_label(s).replace('"', "'").replace("<", "").replace(">", "")
    url += ")"
    return url


def GetOpNodeProducer(**kwargs):
    def ReallyGetOpNode(op, op_id):
        if op.name:
            node_name = "%s/%s" % (op.op_type, op.name)
        else:
            node_name = "%s" % (op.op_type)
        node = pydot.Node(node_name, **kwargs)

        return node

    return ReallyGetOpNode


def get_model_graph(graph, name=None, flow="LR"):
    node_producer = GetOpNodeProducer(**OP_STYLE)

    pydot_graph = pydot.Dot(name, rankdir=flow)
    pydot_node_counts = defaultdict(int)
    pydot_nodes = {}

    for op_id, op in enumerate(graph.node):
        op_node = node_producer(op, op_id)
        pydot_graph.add_node(op_node)

        for input_name in op.input:
            if input_name not in pydot_nodes:
                input_node = pydot.Node(
                    _escape_label(input_name + str(pydot_node_counts[input_name])),
                    label=_escape_label(input_name),
                    **BLOB_STYLE
                )
                pydot_nodes[input_name] = input_node
            else:
                input_node = pydot_nodes[input_name]
            pydot_graph.add_node(input_node)
            pydot_graph.add_edge(pydot.Edge(input_node, op_node))

        for output_name in op.output:
            if output_name in pydot_nodes:
                pydot_node_counts[output_name] += 1
            output_node = pydot.Node(
                _escape_label(output_name + str(pydot_node_counts[output_name])),
                label=_escape_label(output_name),
                **BLOB_STYLE
            )
            pydot_nodes[output_name] = output_node
            pydot_graph.add_node(output_node)
            pydot_graph.add_edge(pydot.Edge(op_node, output_node))

    return pydot_graph

def form_timestamp(ts):
  st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
  st2 = st.replace(" ","_")
  st3 = st2.replace(":","_")
  return st3

#Pandas training data pointer (currently for data updates only)
def extract_varnames_fromtrainingdata(trainingdata="default"):
  import pandas as pd
  import numpy as np
  if isinstance(trainingdata, pd.DataFrame):
    variabletypes=list(trainingdata.dtypes.values.astype(str)) # always use pandas dtypes
    variablecolumns=list(trainingdata.columns)
  else:
    variabletypes=None
    variablecolumns=None
  return [variabletypes,variablecolumns]

def _get_extension_from_filepath(Filepath):
  Filename = os.path.basename(Filepath)
  file_name, file_extension = os.path.splitext(Filename)
  return file_extension

