import numpy as np

class Tools:
    @classmethod
    def save_string2array_dict(cls, data_dict, path_string, metainfo_string):
        with open(path_string, 'w') as file:
            file.write(metainfo_string)
            for key_string, value_array in data_dict.items():
                file.write("%s\t%s\n"%(key_string, ",".join([str(i) for i in value_array])))

    @classmethod
    def load_string2array_float_dict(cls, path_string):
        data_dict = {}
        metainfo_string = ""
        with open(path_string, 'r') as file:
            metainfo_string = file.readline()
            for line in file.readlines():
                [key_string, value_string] = line.split()
                data_dict[key_string] = np.array([float(i) for i in value_string.split(",")])
        return metainfo_string, data_dict

    @classmethod
    def load_string2list_dict(cls, path_string):
        data_dict = {}
        metainfo_string = ""
        with opne(path_string, 'r') as file:
            metainfo_string = file.readline()
            for line in file.readlines():
                [key_string, value_string] = line.split()
                data_dict[key_string] = value_string.split(',')
        return metainfo_string, data_dict

