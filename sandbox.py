from elisa.engine import atm


fpaths = ['10000', '9000', '9000', '9000', '10001', '10000', None, None, '8000', '9000']
fpaths_set = set(fpaths)
fpaths_map = {str(key): list() for key in fpaths}

for idx, key in enumerate(fpaths):
    fpaths_map[str(key)].append(idx)

print(fpaths_map)


