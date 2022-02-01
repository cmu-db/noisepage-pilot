import numpy as np
import xml.etree.ElementTree as ET
import yaml

# dist is a map of string->list[];
# return a list of tuple (list[], list[]),
# xml namespace & a list of values.
def param_sweep_space(ps_dist):
    ps_space = []
    for name in ps_dist:
        val_list = ps_dist[name]
        if val_list is None or len(val_list) == 0:
            raise Exception('parameter space should not be empty.')
        
        xml_level = list(name.strip().split('.'))

        # linear scale range
        if (val_list[0] == '.range'):
            if len(val_list) not in (3, 4):
                raise Exception('incorrect argument number for linear scale')
            if (val_list == 3):
                val_list = np.arange(val_list[1], val_list[2], 1).tolist()
            else:
                val_list = np.arange(val_list[1], val_list[2], val_list[3]).tolist()

        ps_space.append((xml_level, val_list))

    return ps_space

# inject list of param to certain xml file, given that file path name.
# return an xml etree structure.
def inject_param_xml(file_path, params):
    conf_etree = ET.parse(file_path)
    root = conf_etree.getroot()
    for name_level, val in params:
        cursor = root
        # traverse xml name levels.
        for key in name_level:
            cursor = cursor.find(key)
            if cursor is None:
                break

        if cursor is None:
            print('Warning: fail to inject parameter in conf file,', name_level)
            continue

        cursor.text = str(val)

    conf_etree.write(file_path)

# recursive calling routine on parameter space.
# f is a callback function with closure, it should deside
# what to do with a permutation list of parameters.
def parameter_sweep(ps_space, f, closure=dict()):
    parameter_sweep_r(ps_space, [], f, closure)

def parameter_sweep_r(ps_space, cur_params, f, closure):
    if len(cur_params) == len(ps_space):
        f(cur_params, closure)
        return
    
    name_level, val_list = ps_space[len(cur_params)]
    for val in val_list:
        cur_params.append((name_level, val))
        parameter_sweep_r(ps_space, cur_params, f, closure)
        del cur_params[-1]

if __name__ == '__main__':
    config = yaml.load(open("behavior/default.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    ps = param_sweep_space(config['datagen']['param_sweep'])
    print(ps)

    def f(params):
        print('_'.join([x[0][-1] + '_' + str(x[1]) for x in params]))
        

    parameter_sweep(ps, f)

    import os
    os.remove('test.xml')


            