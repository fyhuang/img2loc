"""
Simple utility class to manage mappings between labels and class names

>>> mapping = LabelMapping()
>>> mapping.add("Blue")
0
>>> mapping.add("Green")
1
>>> mapping.get_label("Blue")
0
>>> mapping.get_label("Green")
1
>>> mapping.get_name(0)
'Blue'
>>> mapping.get_name(1)
'Green'

You can also pass the list of class names at creation:

>>> mapping = LabelMapping(["Red", "Yellow"])
>>> mapping.get_label("Red")
0
>>> mapping.get_name(1)
'Yellow'
"""

import pandas

class LabelMapping:
    def __init__(self, names=None):
        self.label_to_name = {}
        self.name_to_label = {}
        self.next_label = 0

        if names is not None:
            for name in names:
                self.add(name)


    def __len__(self):
        return len(self.label_to_name)

    def get_label(self, name):
        assert type(name) == str
        return self.name_to_label[name]

    def get_name(self, label):
        assert type(label) == int
        return self.label_to_name[label]

    def add(self, name):
        assert type(name) == str
        label = self.next_label
        self.label_to_name[label] = name
        self.name_to_label[name] = label
        self.next_label += 1
        return label


    @classmethod
    def read_csv(cls, path):
        df = pandas.read_csv(path)
        df.sort_values("label", inplace=True)

        mapping = cls()
        for index, row in df.iterrows():
            assigned_label = mapping.add(row["name"])
            assert assigned_label == row["label"]
        return mapping

    def to_csv(self, path):
        sorted_labels = sorted(self.label_to_name.keys())
        df = pandas.DataFrame({
            "label": sorted_labels,
            "name": [self.label_to_name[label] for label in sorted_labels],
        })
        df.to_csv(path)


if __name__ == "__main__":
    import doctest
    doctest.testmod()