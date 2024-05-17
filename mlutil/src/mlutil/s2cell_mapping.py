"""
Maps lat/lng to an s2cell ID/label from a set of s2 cells

>>> mapping = S2CellMapping(["8085"])
>>> mapping.lat_lng_to_token(37.7953, -122.3939)
'8085'
>>> mapping.lat_lng_to_token(34.0561, -118.2364) is None
True
"""

import s2sphere

MIN_CELL_LEVEL = 6 # TODO: don't hardcode

class S2CellMapping:
    def __init__(self, all_cell_tokens):
        self.all_cell_ids = set(s2sphere.CellId.from_token(token).id() for token in all_cell_tokens)

    @classmethod
    def from_label_mapping(cls, label_mapping):
        all_cell_tokens = label_mapping.name_to_label.keys()
        return cls(all_cell_tokens)

    def lat_lng_to_token(self, lat, lng):
        s2_cell_id = s2sphere.CellId.from_lat_lng(s2sphere.LatLng.from_degrees(lat, lng))
        while s2_cell_id.id() not in self.all_cell_ids:
            if s2_cell_id.level() < MIN_CELL_LEVEL:
                break
            s2_cell_id = s2_cell_id.parent()

        if s2_cell_id.id() not in self.all_cell_ids:
            # This example can't be labeled
            return None

        token = s2_cell_id.to_token()
        return token


if __name__ == "__main__":
    import doctest
    doctest.testmod()