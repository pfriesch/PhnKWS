import configparser
from glob import glob

import toml

# file = "TIMIT_MLP_mfcc_basic.cfg"

for file in glob("./*/*.cfg"):

    config = configparser.ConfigParser()
    config.sections()
    assert len(config.read(file)) > 0


    def check_float(val):
        try:
            float(val)
            return True
        except ValueError:
            return False


    def map_basic(val):
        if val == "True":
            return True
        elif val == "False":
            return False
        elif val.isdecimal():
            return int(val)
        elif check_float(val):
            return float(val)
        else:
            return val


    def map_value(section, key, val):
        assert isinstance(val, str)
        if val == "True" or val == "False" or val.isdecimal() or check_float(val):
            return map_basic(val)

        elif key == "fea" or key == "lab" and "\n" in val:
            elems = val.split("\n")

            def to_key_value(elem):
                key, value = elem.split("=", 1)
                return key, map_basic(value)

            if "" not in elems:
                return [dict([to_key_value(elem) for elem in elems])]

            else:

                _list = []

                split_idx = [i for i, e in enumerate(elems) if e == ""]
                _list.append(dict([to_key_value(elem) for elem in elems[:split_idx[0]]]))
                for i in range(len(split_idx) - 1):
                    _list.append(dict([to_key_value(elem) for elem in elems[split_idx[i] + 1:split_idx[i + 1]]]))

                _list.append(dict([to_key_value(elem) for elem in elems[split_idx[-1] + 1:]]))
                return _list



        elif "architecture" in section and "," in val:
            return [map_basic(elem) for elem in val.split(",")]

        elif "model" in section and "\n" in val:
            elems = val.split("\n")

            def to_key_value(elem):
                key, value = elem.split("=", 1)
                return key, map_basic(value)

            return dict([to_key_value(elem) for elem in elems])


        else:
            return val


    as_dict = {key: {elem: map_value(key, elem, config[key][elem]) for elem in config[key]} for key, value in
               config.items()}

    with open(file[:-3] + "toml", "w") as f:
        toml.dump(as_dict, f)
