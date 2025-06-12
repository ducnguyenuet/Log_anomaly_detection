from deeplog_trainer.workflow.network import Network, RootNode
import copy
import numpy as np


class WorkflowBuilder:
    def __init__(self, logger):
        """
        Attributes
        :param logger: logger function from logging module
        """
        self.logger = logger

    def build_workflows(self, dataset, initial_workflows=None, threshold=0.8,
                        back_steps=1):
        """
        Builds workflows given a dataset of sequences. Also, it is possible to
        provide a set of workflows (initial_workflows) to update them.
        """
        if initial_workflows is None:
            workflows = {
                'network': Network(),
                'data': []
            }
        else:
            workflows = copy.deepcopy(initial_workflows)
        wf_sequences = dataset + [seq for seq in workflows['data']]
        # Remove duplicate sequences
        wf_sequences = [tuple(s) for s in wf_sequences]
        wf_sequences = [list(s) for s in list(dict.fromkeys(wf_sequences))]
        similar_seqs = self._get_similar_sequences(wf_sequences,
                                                   threshold=threshold)
        _ = self._build_all_paths(workflows['network'], similar_seqs,
                                  back_steps=back_steps)

        workflows['data'] = wf_sequences

        return workflows

    def _get_similar_sequences(self, dataset, threshold):
        """
        Find similar sequences using BLEU score.
        """
        self.logger.info('Searching similar sequences...')
        # Long sequences first
        n_items = len(dataset)
        dataset = sorted(dataset, key=len, reverse=True)
        max_length = max([len(s) for s in dataset])
        original_dataset = dataset  # Copy original dataset
        # Create numpy matrix with all sequences and a vector with sequences
        # length to compute the similarity scores
        lengths = np.zeros(n_items, dtype='int32')
        dataset = np.random.rand(n_items, max_length)  # Square matrix
        # Store lengths of each sequence in the original matrix
        for i in range(n_items):
            lengths[i] = len(original_dataset[i])
            dataset[i, :len(original_dataset[i])] = original_dataset[i]

        # Copy matrix to shift rows later
        cp_dataset = copy.deepcopy(dataset)

        # List to store similar sequences
        similar_seqs = []
        for i in range(n_items):
            similar_seqs.append(
                [original_dataset[i]])  # Copy sequences from original dataset
        for offset in range(1, n_items):
            self.logger.info('Searching similar sequences: {} / {}...'.format(
                offset, n_items - 1))

            # To compute the exact BLEU score, we have to use the maximum
            # between the length of s1 and the length of all sequences. However,
            # to speed up, we will simply use max_length
            # copy_lengths = lengths.copy()
            # copy_lengths[lengths < len(s1)] = len(s1)

            # Shift rows one position
            cp_dataset = np.roll(cp_dataset, -1, axis=0)
            dataset = dataset
            cp_dataset = cp_dataset

            # Compute BLEU scores
            scores = np.sum(dataset == cp_dataset, axis=-1) / max_length

            # Store matches
            matches = np.argwhere(scores >= threshold).reshape(-1)
            for match in matches:
                similar_seqs[match].append(
                    original_dataset[(match + offset) % n_items])
        return similar_seqs

    def _find_path(self, network, node_idx, seq, found_workflow=None):
        """
        Finds a path of nodes for a given sequence, recursively. Returns an
        empty list if the path does not exists.
        """
        found_workflow = [] if found_workflow is None else found_workflow
        if len(seq) == 0:
            return found_workflow
        node = network.get_node(node_idx)
        for _, child_idx in node.get_children().items():
            child_node = network.get_node(child_idx)
            if child_node.get_value() == seq[0]:
                iter_found_workflow = self._find_path(network, child_idx,
                                                      seq[1:],
                                                      found_workflow=
                                                      (found_workflow +
                                                       [child_node.get_idx()]))
                if len(iter_found_workflow) > 0:
                    return iter_found_workflow
        return []

    def _build_all_paths(self, network, similar_seqs, back_steps):
        # Build paths between nodes
        self.logger.info('Building workflows...')
        root_node = network.get_root_node()
        root_idx = root_node.get_idx()

        added_workflows = []
        for i, seqs in enumerate(similar_seqs):
            self.logger.info('Building workflows: {} / {}...'.format(
                i + 1, len(similar_seqs)))
            ref_seq = []
            ref_workflows = []
            explored_edges = {}

            # In the group of similar sequences, check if any of them already
            # exists as workflow. If it exists, we will use it as a reference,
            # so we append nodes to the existing path instead of creating a new
            # independent one.
            seqs_to_ignore = []
            n_found = 0
            for k, seq in enumerate(seqs):
                found_workflow = self._find_path(network, root_idx, seq)
                if len(found_workflow) > 0:
                    n_found += 1
                    seqs_to_ignore.append(k)
                    if k == 0:
                        # If first sequence is found, skip group of sequences
                        break

                    ref_seq = seq
                    found_workflow = [found_workflow]
                    new_ref_workflows = list(
                        map(list, zip(*found_workflow)))  # Transpose
                    ref_workflows = self._combine_ref_workflows(
                        ref_workflows, new_ref_workflows)

            if 0 not in seqs_to_ignore:
                # Add the first sequence of the group similar_seqs
                # (if it does not exist yet)
                seqs_to_ignore.append(0)
                iter_added_workflows = self._build_path(
                    network, root_idx, seqs[0], True, back_steps=back_steps,
                    ref_seq=ref_seq, ref_workflows=ref_workflows,
                    explored_edges=explored_edges)

                added_workflows += iter_added_workflows
                new_ref_workflows = list(map(list, zip(*iter_added_workflows)))
                ref_workflows = self._combine_ref_workflows(ref_workflows,
                                                            new_ref_workflows)
            else:
                # If the first sequence (which should be used as reference)
                # already exists, skip
                continue

            ref_seq = seqs[0]
            # Then, add the rest of the sequences in the group similar_seqs
            # using the first added sequence as a reference
            for k, seq in enumerate(seqs[1:]):
                self.logger.info('Building workflows: {} / {} ({} / {})'
                                 '...'.format(i + 1, len(similar_seqs),
                                              k + 2, len(seqs)))

                if k + 1 in seqs_to_ignore:
                    continue

                iter_added_workflows = self._build_path(
                    network, root_idx, seq, True, back_steps=back_steps,
                    ref_seq=ref_seq, ref_workflows=ref_workflows,
                    explored_edges=explored_edges)
                added_workflows += iter_added_workflows
                new_ref_workflows = list(
                    map(list, zip(*iter_added_workflows)))
                ref_workflows = self._combine_ref_workflows(ref_workflows,
                                                            new_ref_workflows)
        return added_workflows

    def _build_path(self, network, parent_idx, seq, is_start, back_steps,
                    ref_seq=None, ref_workflows=None, workflow_path=None,
                    explored_edges=None):
        """
        Recursive method to build workflow path.
        Initial call: _build_path(parent: root node, seq: array with token IDs,
        is_start: True)
        """
        explored_edges = {} if explored_edges else explored_edges
        ref_seq = [] if ref_seq is None else ref_seq
        workflow_path = [] if workflow_path is None else workflow_path
        ref_workflows = [] if ref_workflows is None else ref_workflows
        parent = network.get_node(parent_idx)

        if len(seq) == 0:
            parent.set_end(True)
            return [workflow_path]

        current_value = seq[0]
        added_workflows = []

        if len(ref_seq) > 0 and ref_seq[0] == current_value:
            # Since current value is the same as the reference sequence,
            # next node is based on such reference
            for next_idx in ref_workflows[0]:
                next_idx = parent.add_child(next_idx)
                next_node = network.get_node(next_idx)
                if next_node.get_value() == current_value and not \
                    self._is_edge_explored(network, explored_edges, parent_idx,
                                           next_idx, seq[1:]):
                    self._add_explored_edge(network, explored_edges, parent_idx,
                                            next_idx, seq[1:])
                    added_workflows += self._build_path(
                        network, next_idx, seq[1:], False,
                        back_steps=back_steps, ref_seq=ref_seq[1:],
                        ref_workflows=ref_workflows[1:],
                        workflow_path=(workflow_path + [next_idx]),
                        explored_edges=explored_edges)
        else:
            # Search a similar node (i.e. same value) in the path and add it as
            # a child (i.e. a loop) or, if not exists, create a new node.
            # Additional constraints:
            # - The previous node cannot be the root node.
            # - the previous node cannot be a starting node.
            found_next = False

            for i in range(len(workflow_path)):
                if i > back_steps:
                    # Limit of back steps in the path
                    break

                prev_idx = workflow_path[-(i + 1)]
                prev_node = network.get_node(prev_idx)

                if isinstance(prev_node, RootNode) or prev_node.is_start():
                    continue
                elif prev_node.get_value() == current_value:
                    found_next = True
                    next_idx = prev_idx

                    if self._is_edge_explored(network, explored_edges,
                                              parent_idx, next_idx, seq[1:]):
                        continue

                    next_idx = parent.add_child(next_idx)
                    self._add_explored_edge(network, explored_edges, parent_idx,
                                            next_idx, seq[1:])
                    added_workflows += self._build_path(
                        network, next_idx, seq[1:], False,
                        back_steps=back_steps, ref_seq=ref_seq[1:],
                        ref_workflows=ref_workflows[1:],
                        workflow_path=(workflow_path + [next_idx]),
                        explored_edges=explored_edges)
                    break

            if not found_next:
                # If no previous node has been found in the path, add a new one
                next_idx = network.add_node(seq[0], is_start, False, parent_idx)
                next_idx = parent.add_child(next_idx)
                self._add_explored_edge(network, explored_edges, parent_idx,
                                        next_idx, seq[1:])
                added_workflows += self._build_path(
                    network, next_idx, seq[1:], False, back_steps=back_steps,
                    ref_seq=ref_seq[1:], ref_workflows=ref_workflows[1:],
                    workflow_path=(workflow_path + [next_idx]),
                    explored_edges=explored_edges)
        return added_workflows

    def _is_edge_explored(self, network, explored_edges, from_idx, to_idx, seq):
        from_idx = network.get_node(from_idx).get_idx()
        to_idx = network.get_node(to_idx).get_idx()
        if from_idx not in explored_edges:
            return False
        elif to_idx not in explored_edges[from_idx]:
            return False
        elif seq not in explored_edges[from_idx][to_idx]:
            return False
        else:
            return True

    def _add_explored_edge(self, network, explored_edges, parent_idx, next_idx,
                           seq):
        # Fix idx
        parent_idx = network.get_node(parent_idx).get_idx()
        next_idx = network.get_node(next_idx).get_idx()
        # Add edge
        if parent_idx not in explored_edges:
            explored_edges[parent_idx] = {}
        if next_idx not in explored_edges[parent_idx]:
            explored_edges[parent_idx][next_idx] = []
        if seq not in explored_edges[parent_idx][next_idx]:
            explored_edges[parent_idx][next_idx].append(seq)

    def _combine_ref_workflows(self, a, b):
        """
        Combines two lists of different lengths.
        Example: [[1,2], [3]] and [[1,4]] -> [[1,2,4], [3]]
        """
        if len(a) > len(b):
            result = a
            transfer = b
        else:
            result = b
            transfer = a
        result = [list(set(result[i] + transfer[i])) for i in
                  range(len(transfer))] + result[len(transfer):]
        return result


class WorkflowEvaluator:
    def __init__(self, logger, network: {}):
        """
        Attributes
        :param logger: logger function from logging module
        :param network: Network in json format created with class
        WorkflowBuilder
        """
        self.logger = logger
        self.network = network

    def evaluate(self, dataset):
        self.logger.info('Evaluating workflows...')
        root_node = self.network['root']
        results = [self._evaluate_seq(root_node, seq) for seq in dataset]
        return results

    def _evaluate_seq(self, node: {}, seq):
        if len(seq) == 0:
            return True
        else:
            children = node["children"]
            current_value = seq[0]
            if current_value not in children:
                # Sequence does not exist in the workflows
                return False
            else:
                next_node_idx = children[current_value]
                next_node = self.network[str(next_node_idx)]
                return self._evaluate_seq(next_node, seq[1:])

    @staticmethod
    def compute_scores(matches):
        n_items = len(matches)
        n_correct = sum(matches)
        try:
            accuracy = n_correct / n_items
        except ZeroDivisionError:
            accuracy = 0
        return {
            'n_items': n_items,
            'n_correct': n_correct,
            'accuracy': accuracy
        }
