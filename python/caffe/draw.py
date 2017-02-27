"""
Caffe network visualization: draw the NetParameter protobuffer.

NOTE: this requires pydot>=1.0.2, which is not included in requirements.txt
since it requires graphviz and other prerequisites outside the scope of the
Caffe.
"""

from collections import OrderedDict
from caffe.proto import caffe_pb2
import pydot

# Internal layer and blob styles.
LAYER_STYLE_DEFAULT = {'shape': 'record', 'fillcolor': '#6495ED',
         'style': 'filled'}
NEURON_LAYER_STYLE = {'shape': 'record', 'fillcolor': '#90EE90',
         'style': 'filled'}
BLOB_STYLE = {'shape': 'octagon', 'fillcolor': '#E0E0E0',
        'style': 'filled'}

def get_pooling_types_dict():
    """Get dictionary mapping pooling type number to type name
    """
    desc = caffe_pb2.PoolingParameter.PoolMethod.DESCRIPTOR
    d = {}
    for k,v in desc.values_by_name.items():
        d[v.number] = k
    return d


def determine_edge_label_by_layertype(layer, layertype):
    """Define edge label based on layer type
    """

    if layertype == 'Data':
        edge_label = 'Batch ' + str(layer.data_param.batch_size)
    elif layertype == 'Convolution':
        edge_label = str(layer.convolution_param.num_output)
    elif layertype == 'InnerProduct':
        edge_label = str(layer.inner_product_param.num_output)
    else:
        edge_label = '""'

    return edge_label


def determine_node_label_by_layertype(layer, layertype, rankdir):
    """Define node label based on layer type
    """

    if rankdir in ('TB', 'BT'):
        # If graph orientation is vertical, horizontal space is free and
        # vertical space is not; separate words with spaces
        separator = ' '
    else:
        # If graph orientation is horizontal, vertical space is free and
        # horizontal space is not; separate words with newlines
        separator = '\n'

    if layertype == 'Convolution':
        # Outer double quotes needed or else colon characters don't parse
        # properly
        node_label = '"%s%s(%s)%skernel size: %d%sstride: %d%spad: %d"' %\
                     (layer.name,
                      separator,
                      layertype,
                      separator,
                      layer.convolution_param.kernel_size,
                      separator,
                      layer.convolution_param.stride,
                      separator,
                      layer.convolution_param.pad)
    elif layertype == 'Pooling':
        pooling_types_dict = get_pooling_types_dict()
        node_label = '"%s%s(%s %s)%skernel size: %d%sstride: %d%spad: %d"' %\
                     (layer.name,
                      separator,
                      pooling_types_dict[layer.pooling_param.pool],
                      layertype,
                      separator,
                      layer.pooling_param.kernel_size,
                      separator,
                      layer.pooling_param.stride,
                      separator,
                      layer.pooling_param.pad)
    else:
        node_label = '"%s%s(%s)"' % (layer.name, separator, layertype)
    return node_label


def choose_color_by_layertype(layertype):
    """Define colors for nodes based on the layer type
    """
    color = '#6495ED'  # Default
    if layertype == 'Convolution':
        color = '#FF5050'
    elif layertype == 'Pooling':
        color = '#FF9900'
    elif layertype == 'InnerProduct':
        color = '#CC33FF'
    return color

def get_param_color(pind):
  """Define colors for edges of shared parameters 
  """
  color_list = ['#F49AC2', '#77DD77', '#779ECB',
  '#AEC6CF', '#DEA5A4', '#CFCFC4']
  if pind >= 6:
    return 'gray'
  return color_list[pind]

def get_pydot_graph(caffe_net, rankdir, label_edges=True, param_edges=True):
  pydot_graph = pydot.Dot(caffe_net.name, graph_type='digraph', rankdir=rankdir)
  pydot_nodes = {}
  pydot_edges = []
  params = OrderedDict()
  for layer in caffe_net.layer:
    name = layer.name
    layertype = layer.type
    node_label = determine_node_label_by_layertype(layer, layertype, rankdir)
    if (len(layer.bottom) == 1 and len(layer.top) == 1 and
        layer.bottom[0] == layer.top[0]):
      # We have an in-place neuron layer.
      pydot_nodes[name + '_' + layertype] = pydot.Node(
          node_label, **NEURON_LAYER_STYLE)
    else:
      layer_style = LAYER_STYLE_DEFAULT
      layer_style['fillcolor'] = choose_color_by_layertype(layertype)
      pydot_nodes[name + '_' + layertype] = pydot.Node(
          node_label, **layer_style)
    for bottom_blob in layer.bottom:
      pydot_nodes[bottom_blob + '_blob'] = pydot.Node(
        '%s' % (bottom_blob), **BLOB_STYLE)
      edge_label = '""'
      pydot_edges.append({'src': bottom_blob + '_blob',
                          'dst': name + '_' + layertype,
                          'label': edge_label})
    for top_blob in layer.top:
      pydot_nodes[top_blob + '_blob'] = pydot.Node(
        '%s' % (top_blob))
      if label_edges:
        edge_label = determine_edge_label_by_layertype(layer, layertype)
      else:
        edge_label = '""'
      pydot_edges.append({'src': name + '_' + layertype,
                          'dst': top_blob + '_blob',
                          'label': edge_label})
    # Check sharing params
    for pind, param in enumerate(layer.param):
      layer_name = name + '_' + layertype
      param_name = param.name
      if not param_name:
        continue
      layer_list = params.get(param_name, []) + [(layer_name, pind)]
      params[param_name] = layer_list

  # Now, add the nodes and edges to the graph.
  for node in pydot_nodes.values():
    pydot_graph.add_node(node)
  for edge in pydot_edges:
    pydot_graph.add_edge(
        pydot.Edge(pydot_nodes[edge['src']], pydot_nodes[edge['dst']],
                   label=edge['label']))
  # Connect shared params with edges
  if param_edges:
    params = filter(lambda x: len(x[1])>1, params.items())
    for pname, layers in params:
      for lid in xrange(1, len(layers)):
        pydot_graph.add_edge(
          pydot.Edge(
            pydot_nodes[layers[lid-1][0]], pydot_nodes[layers[lid][0]],
            dir='none', splines='curved', constraint='false',
            color=';'.join(
              map(get_param_color, (layers[lid-1][1], layers[lid][1]))
              ),
            )
          )
  return pydot_graph

def draw_net(caffe_net, rankdir, ext='png', param_edges=True):
  """Draws a caffe net and returns the image string encoded using the given
  extension.

  Input:
    caffe_net: a caffe.proto.caffe_pb2.NetParameter protocol buffer.
    ext: the image extension. Default 'png'.
  """
  return get_pydot_graph(caffe_net, rankdir,
    param_edges=param_edges).create(format=ext)

def draw_net_to_file(caffe_net, filename, rankdir='LR'):
  """Draws a caffe net, and saves it to file using the format given as the
  file extension. Use '.raw' to output raw text that you can manually feed
  to graphviz to draw graphs.
  """
  ext = filename[filename.rfind('.')+1:]
  with open(filename, 'wb') as fid:
    fid.write(draw_net(caffe_net, rankdir, ext))
