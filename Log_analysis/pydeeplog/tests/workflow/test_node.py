import pytest
from deeplog_trainer.workflow.network import Node, RootNode


def get_data1():
    nodes_data = [dict(value=3, is_start=True, is_end=False, idx=1,
                       parent_idx=0),
                  dict(value=5, is_start=False, is_end=False, idx=2,
                       parent_idx=2),
                  dict(value=5, is_start=False, is_end=False, idx=3,
                       parent_idx=1),
                  dict(value=4, is_start=False, is_end=True, idx=4,
                       parent_idx=1)]
    for node_data in nodes_data:
        yield node_data

@pytest.mark.parametrize("node_data", get_data1())
def test_node_get_methods(mocker, node_data):
    # Mock network class
    mocked_network = \
        mocker.patch('deeplog_trainer.workflow.workflow.Network',
                     autospec=True)
    # Initialise the node with the data in input and test each attribute got the
    # right value
    node = Node(mocked_network, **node_data)
    assert node.get_value() == node_data['value']
    assert node.get_network() == mocked_network
    assert node.get_idx() == node_data['idx']
    assert node.is_start() == node_data['is_start']
    assert node.is_end() == node_data['is_end']
    assert node.get_parents() == [node_data['parent_idx']]

@pytest.mark.parametrize(
    "value, is_start, is_end, idx, parents_idx",
    [(2, True, False, 2, [0, 1]), (5, False, True, 3, [1, 2]),
     (3, False, False, 4, [2, 3])]
)
def test_node_set_methods(mocker, value, is_start, is_end, idx, parents_idx):
    mocked_network = \
        mocker.patch('deeplog_trainer.workflow.workflow.Network',
                     autospec=True)
    node = Node(mocked_network, value)
    node.set_start(is_start)
    assert node.is_start() is is_start
    node.set_end(is_end)
    assert node.is_end() is is_end
    node.set_idx(idx)
    assert node.get_idx() == idx
    root_node = RootNode(mocked_network, idx=0)
    # random node to add to the parents of entry Node
    sample_node = Node(mocked_network, 3, idx=1)
    # Mock method get_node() returning every time it is called, a dictionary
    # with as key the index of the parent, ans as value the corresponding Node
    mocked_network.get_node.side_effect = [{parents_idx[0]: root_node},
                                           {parents_idx[1]: sample_node}]
    node.add_parents(parents_idx)
    assert node.get_parents() == parents_idx

def get_data2():
    nodes = [(dict(value=3, idx=1, parent_idx=0),
             dict(value=5, idx=2, parent_idx=2),
             dict(value=5, idx=3, parent_idx=1),
             dict(value=4, is_end=True, idx=4, parent_idx=1))]
    return nodes

@pytest.mark.parametrize("data_node1, data_node2, data_node3, data_node4",
                         get_data2())
def test_node_add_children(mocker, data_node1, data_node2, data_node3,
                           data_node4):
    mocked_network = \
        mocker.patch('deeplog_trainer.workflow.workflow.Network',
                     autospec=True)
    # Call the mocked class
    instance = mocked_network.return_value
    # Initialise the first node
    node1 = Node(instance, **data_node1)
    # Mock the method get_nodes returning the dictionary with the nodes in input
    nodes = {}
    for data_node in [data_node1, data_node2, data_node3, data_node4]:
        nodes[data_node['idx']] = Node(instance, **data_node)
    instance.get_nodes.return_value = nodes
    # Raise an error if you try to add a child with an unknown index Node
    with pytest.raises(Exception):
        indexes = list(nodes.keys())
        node1.add_child(child_idx=max(indexes)+1)
    # Mock method get_node returning the node with the corresponding index
    my_side_effect = lambda index: nodes[index]
    mocker.patch.object(instance, 'get_node', side_effect=my_side_effect)
    # Add the 2nd and 3rd node to the children of the 1st node
    node1.add_children([data_node2['idx'], data_node3['idx']])
    # Since the 2 nodes have the same value the method add_child must
    # combine node3 and node2 and add only the node with the first child
    # index
    assert node1.get_children() == {data_node2['value']: data_node2['idx']}
    # Raise an error if trying to combine two nodes with different values
    with pytest.raises(Exception):
        node1.combine(Node(instance, **data_node4))
