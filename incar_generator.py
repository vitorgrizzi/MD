import numpy as np

def create_incar(header, dict_param):
    with open('INCAR.txt', 'w') as f:
        f.write(f'{header}\n')
        f.write('\n')
        for k, v in dict_param.items():
            if type(v) is bool:
                if v:
                    value='.TRUE.'
                else:
                    value='.FALSE.'
            else:
                value=v

            f.write(f'{k.upper()}={value}\n')


create_incar('HPC', {'encut': 700,
                     'ibrion': -1,
                     'ediff': 1e-8,
                     'ncore': 8,
                     'kpar': 16,
                     'ismear': 1,
                     'sigma': 0.1,
                     'GGA': 'PE',
                     'lasph': True,
                     'lwave': False,
                     'lcharg': False,
                     'isym': 0,
                     'lreal': 'Auto',
                     'ldau': True,
                     'ldauu': 4.0,
                     'lmaxmix': 4})
