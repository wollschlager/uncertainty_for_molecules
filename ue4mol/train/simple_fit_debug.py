#!/usr/bin/env python3
import yaml
from ue4mol.train.fit_gp import run

if __name__ == '__main__':
    with open("seml/configs/multgp_fit_md17_dimenetpp.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    _config = {
        'overwrite': 0,
        'db_collection': 'debug-stuff',
    }
    # grid_params = {}
    # if 'grid' in config:
    #     for k in config['grid']:
    #         if config['grid'][k]['type'] == 'choice':
    #             grid_params[k] = config['grid'][k]['options'][0]
    #         elif config['grid'][k]['type'] == 'range':
    #             grid_params[k] = config['grid'][k]['min']
    # #config['fixed']['debug'] = True
    # run(_config=_config, **config['fixed'], **grid_params)
    
    if 'grid' in config and len(config['grid']) > 0:
        grid_params = {}
        for k in config['grid']:
            for j in config['grid'][k]:
                if 'type' in config['grid'][k][j]:    
                    if config['grid'][k][j]['type'] == 'choice':
                        if k in config['fixed']:
                            config['fixed'][k][j] = config['grid'][k][j]['options'][0]    
                        elif k in grid_params:
                            grid_params[k][j] = config['grid'][k][j]['options'][0]
                        else:
                            grid_params[k] = {}
                            grid_params[k][j] = config['grid'][k][j]['options'][0]
                    elif config['grid'][k][j]['type'] == 'range':
                        grid_params[k] = {}
                        grid_params[k][j] = config['grid'][k][j]['min']
                else:
                    if config['grid'][k]['type'] == 'choice':
                        if k in config['fixed']:
                            config['fixed'][k] = config['grid'][k]['options'][0]    
                        else:
                            grid_params[k] = config['grid'][k]['options'][0]
                    elif config['grid'][k]['type'] == 'range':
                        grid_params[k] = {}
                        grid_params[k][j] = config['grid'][k][j]['min']
        #config['fixed']['debug'] = True
        run(_config=_config, **config['fixed'], **grid_params)
    else:
        run(_config=_config, **config['fixed'])
        