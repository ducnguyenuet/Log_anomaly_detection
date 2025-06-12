import pytest
from deeplog_trainer.workflow.network import RootNode, Network


def get_data():
    # The function provides a tuple of data and expected values for each test
    return [([], ["root"], [None], [False], [False], [[]]),
            ([dict(value=1, is_start=True, parent_idx="root"), dict(value=5),
              dict(value=5, parent_idx="node-1"),
              dict(value=4, is_end=True, parent_idx="node-1"), dict(value=7)],
             ["node-1", "node-2", "node-3", "node-4", "node-5"],
             [1, 5, 5, 4, 7],
             [True, False, False, False, False],
             [False, False, False, True, False],
             [["root"], [], ["node-1"], ["node-1"], []])]

@pytest.mark.parametrize("nodes_data, expected_new_idx, expected_value, "
                         "expected_start, expected_end, expected_parent_idx",
                         get_data())
def test_build_network(nodes_data, expected_new_idx, expected_value,
                       expected_start, expected_end, expected_parent_idx):
    network = Network()
    assert isinstance(network.get_root_node(), RootNode)
    # Test Network methods in 2 cases: when the class in empty (no node added)
    # and when few nodes are added
    # Add all the nodes to the network
    for i, node_data in enumerate(nodes_data):
        network.add_node(**node_data)
        # Test the node index increase every time a new node is added to the
        # network
        assert network.get_last_index() == expected_new_idx[i]
    assert len(network.get_nodes()) == len(nodes_data) + 1
    # Test Network methods in 2 cases: when the class in empty (no node added)
    # and when few nodes are added
    for i in range(len(expected_new_idx)):
        node = network.get_node(expected_new_idx[i])
        assert node.get_value() == expected_value[i]
        assert node.is_start() == expected_start[i]
        assert node.is_end() == expected_end[i]
        assert node.get_parents() == expected_parent_idx[i]
