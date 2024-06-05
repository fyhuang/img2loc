"""
Maps lat/lng to an s2cell ID/label from a set of s2 cells

>>> import label_mapping
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
"""

import s2sphere

class S2CellMapping:
    def __init__(self, label_mapping):
        all_cell_tokens = label_mapping.name_to_label.keys()
        self.label_mapping = label_mapping

        self.all_cell_ids = set()
        self.min_cell_level = float("inf")
        self.max_cell_level = float("-inf")

        for token in all_cell_tokens:
            cell_id = s2sphere.CellId.from_token(token)
            self.all_cell_ids.add(cell_id.id())
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


if __name__ == "__main__":
    import doctest
    doctest.testmod()