"""
Maps lat/lng to an s2cell ID/label from a set of s2 cells

>>> from . import label_mapping
>>> mapping = S2CellMapping(label_mapping.LabelMapping([
...     "8085", "8085c", "80859", "808584", # San Francisco
...     "89c3", "89c24", "89c25", "89c25c", # New York
... ]))
>>> mapping.lat_lng_to_token(37.7953, -122.3939)
'808584'
>>> mapping.lat_lng_to_token(34.0561, -118.2364) is None  # Los Angeles
True
>>> mapping.lat_lng_to_multihot_list(37.7953, -122.3939)
[1, 1, 1, 1, 0, 0, 0, 0]
>>> mapping.lat_lng_to_multihot_list(37.9720, -122.5226)
[1, 1, 1, 0, 0, 0, 0, 0]


The mapping can convert from a predicted token list (multi-label)
to a single predicted CellId.

>>> from . import label_mapping
>>> mapping = S2CellMapping(label_mapping.LabelMapping([
...     "8085", "8085c", "80859", "808584", # San Francisco
...     "89c3", "89c24", "89c25", "89c25c", # New York
... ]))
>>> mapping.token_list_to_prediction(["8085", "8085c", "80859", "808584"]).to_token()
'808584'
>>> # 89c25c (NYC) is not congruent with the rest (SF)
>>> mapping.token_list_to_prediction(["8085", "8085c", "80859", "89c25c"]).to_token()
'80859'


The mapping can organize the labels by (cell) level.

>>> from . import label_mapping
>>> mapping = S2CellMapping(label_mapping.LabelMapping([
...     "8085", "808584", "80857c" # San Francisco
... ]))
>>> mapping.labels_by_level(6)
[0]
>>> mapping.labels_by_level(9)
[1, 2]
"""

import collections

import s2sphere

class S2CellMapping:
    def __init__(self, label_mapping):
        all_cell_tokens = label_mapping.name_to_label.keys()
        self.label_mapping = label_mapping

        self.all_cell_ids = set()
        self.tokens_by_level = collections.defaultdict(list)
        self.min_cell_level = float("inf")
        self.max_cell_level = float("-inf")

        for token in all_cell_tokens:
            cell_id = s2sphere.CellId.from_token(token)
            self.all_cell_ids.add(cell_id.id())
            self.tokens_by_level[cell_id.level()].append(token)
            self.min_cell_level = min(self.min_cell_level, cell_id.level())
            self.max_cell_level = max(self.max_cell_level, cell_id.level())

    @classmethod
    def from_label_mapping(cls, label_mapping):
        return cls(label_mapping)

    def lat_lng_to_token(self, lat, lng):
        s2_cell_id = (
            s2sphere.CellId
            .from_lat_lng(s2sphere.LatLng.from_degrees(lat, lng))
            .parent(self.max_cell_level)
        )
        while s2_cell_id.id() not in self.all_cell_ids:
            if s2_cell_id.level() < self.min_cell_level:
                break
            s2_cell_id = s2_cell_id.parent()

        if s2_cell_id.id() not in self.all_cell_ids:
            # This example can't be labeled
            return None

        token = s2_cell_id.to_token()
        return token

    def lat_lng_to_multihot_list(self, lat, lng):
        s2_cell_id = (
            s2sphere.CellId
            .from_lat_lng(s2sphere.LatLng.from_degrees(lat, lng))
            .parent(self.max_cell_level)
        )
        
        label_list = [0] * len(self.label_mapping)
        while s2_cell_id.level() >= self.min_cell_level:
            if s2_cell_id.id() in self.all_cell_ids:
                label_list[self.label_mapping.get_label(s2_cell_id.to_token())] = 1
            s2_cell_id = s2_cell_id.parent()
        return label_list

    def token_list_to_prediction(self, token_list):
        cell_ids = [s2sphere.CellId.from_token(token) for token in token_list]
        cell_id_set = set(c.id() for c in cell_ids)

        # The best cell is the one that has the most ancestors in the prediction set
        # In the case of a tie, we pick the cell with the lowest level (to reflect uncertainty)
        cell_id_to_parents = collections.defaultdict(int)
        for cell_id in cell_ids:
            query = cell_id.parent()
            while query.level() >= self.min_cell_level:
                if query.id() in cell_id_set:
                    cell_id_to_parents[cell_id] += 1
                query = query.parent()

        best_cell_id = max(cell_ids, key=lambda x: (cell_id_to_parents[x], -x.level()))
        return best_cell_id

    def labels_by_level(self, level):
        return [self.label_mapping.get_label(t) for t in self.tokens_by_level[level]]


if __name__ == "__main__":
    import doctest
    doctest.testmod()