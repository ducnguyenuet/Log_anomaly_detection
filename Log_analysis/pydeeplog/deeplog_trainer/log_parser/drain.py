from deeplog_trainer import SERIAL_DRAIN_VERSION


class Drain:
    def __init__(self, template_miner):
        self.template_miner = template_miner
        self.root_node = None

    def add_message(self, msg):
        """
        For each log message in input it returns a dictionary with the
        corresponding template, template Id and list of parameters
        """
        msg = msg.strip()
        cluster = self.template_miner.add_log_message(msg)
        template = cluster['template_mined']
        template_id = cluster['cluster_id']
        parameter_list = self.template_miner.get_parameter_list(
            log_template=template, content=msg)
        result = {
            'template_id': template_id,
            'template': template,
            'params': parameter_list
        }
        return result

    def serialize_drain(self):
        masking = []
        for instruction in self.template_miner.config.masking_instructions:
            masking.append({'regex_pattern': instruction.regex_pattern,
                            'mask_with': '<' + instruction.mask_with + '>'})
        serialized = {'version': SERIAL_DRAIN_VERSION,
                      'depth': self.template_miner.drain.depth + 2,
                      'similarity_threshold': self.template_miner.drain.sim_th,
                      'max_children_per_node':
                          self.template_miner.drain.max_children,
                      'delimiters':
                          [' ', *self.template_miner.drain.extra_delimiters],
                      'masking': masking,
                      'root':
                          self._serialize_node(
                              "root", self.template_miner.drain.root_node, 0)
                      }
        return serialized

    def _serialize_node(self, token, node, depth):
        tree_serialized = {
            'depth': depth,
            'key': token,
            'children': {
                token: self._serialize_node(token, child, depth + 1)
                if len(node.key_to_child_node) > 0
                else {}
                for token, child in node.key_to_child_node.items()
            },
            'clusters': [
                {'cluster_id': cluster_id,
                 'log_template_tokens':
                     list(self.template_miner.drain.id_to_cluster[
                         cluster_id].log_template_tokens)}
                for cluster_id in node.cluster_ids]
        }
        return tree_serialized
    
    def set_root_from_dict(self, root_dict):
        """
        Khôi phục lại cây root từ dict (như trong drain.json["root"])
        """
        def build_node(node_dict):
            # Tạo node mới
            node = type('Node', (), {})()
            node.depth = node_dict.get("depth", 0)
            node.key = node_dict.get("key", "")
            node.key_to_child_node = {}  # Sửa ở đây!
            # Nạp children
            for child_key, child_val in node_dict.get("children", {}).items():
                node.key_to_child_node[child_key] = build_node(child_val)
            # Nạp clusters
            node.cluster_ids = []
            for cluster in node_dict.get("clusters", []):
                # Nếu cluster có cluster_id, lưu lại
                if "cluster_id" in cluster:
                    node.cluster_ids.append(cluster["cluster_id"])
            return node

        self.root_node = build_node(root_dict)
        # Gán lại cho template_miner.drain.root_node nếu cần
        if hasattr(self.template_miner, "drain"):
            self.template_miner.drain.root_node = self.root_node
        elif hasattr(self.template_miner, "root_node"):
            self.template_miner.root_node = self.root_node
