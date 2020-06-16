from os.path import join as pjoin, abspath, dirname

import pandas as pd

from elisa import const, utils

__BASE_PATH__ = pjoin(dirname(dirname(abspath(__file__))), "vh16.orig")
__MH__ = const.METALLICITY_LIST_LD
__HEADER__ = ['xlin', 'qlin', 'xlog', 'ylog', 'qlog', 'xsqrt', 'ysqrt', 'qsqrt']

__TABLE_HEADERS__ = {
    "lin": ["temperature", "gravity", "xlin", "qlin"],
    "log": ["temperature", "gravity", "xlog", "ylog", "qlog"],
    "sqrt": ["temperature", "gravity", "xsqrt", "ysqrt", "qsqrt"],
}

VH_TO_ELISA = {
    'bolometric': 'bolometric',
    'GAIA (2010)': {
        'G': 'GaiaDR2'
    },
    "Bessell": {
        'UX': 'Generic.Bessell.U',
        'B': 'Generic.Bessell.B',
        'V': 'Generic.Bessell.V',
        'R': 'Generic.Bessell.R',
        'I': 'Generic.Bessell.I',
    },
    'Stromgren':
        {
            'b': 'Generic.Stromgren.b',
            'u': 'Generic.Stromgren.u',
            'v': 'Generic.Stromgren.v',
            'y': 'Generic.Stromgren.y',
        },
    'KEPLER': 'Kepler',
    'Sloan DSS':
        {
            'g': 'SLOAN.SDSS.g',
            'i': 'SLOAN.SDSS.i',
            'r': 'SLOAN.SDSS.r',
            'u': 'SLOAN.SDSS.u',
            'z': 'SLOAN.SDSS.z'
        }
}

# __PASSBANDS__ = [
#     'bolometric', 'GaiaDR2', 'Kepler',
#     'Generic.Bessell.U', 'Generic.Bessell.B', 'Generic.Bessell.V', 'Generic.Bessell.R', 'Generic.Bessell.I',
#     'Generic.Stromgren.b', 'Generic.Stromgren.u', 'Generic.Stromgren.v', 'Generic.Stromgren.y',
#     'SLOAN.SDSS.g', 'SLOAN.SDSS.i', 'SLOAN.SDSS.r', 'SLOAN.SDSS.u', 'SLOAN.SDSS.z'
# ]

__PASSBANDS_MAP__ = {
    'bolometric': 'bolometric',
    'GaiaDR2': "gaia", 'Kepler': 'kepler',
    'Generic.Bessell.U': 'bessell', 'Generic.Bessell.B': 'bessell',
    'Generic.Bessell.V': 'bessell', 'Generic.Bessell.R': 'bessell',
    'Generic.Bessell.I': 'bessell',
    'Generic.Stromgren.b': 'stromgren', 'Generic.Stromgren.u': 'stromgren',
    'Generic.Stromgren.v': 'stromgren', 'Generic.Stromgren.y': 'stromgren',
    'SLOAN.SDSS.g': 'sdss', 'SLOAN.SDSS.i': 'sdss', 'SLOAN.SDSS.r': 'sdss', 'SLOAN.SDSS.u': 'sdss',
    'SLOAN.SDSS.z': 'sdss'
}


__PASSBANDS_MAP__ = {
    'bolometric': 'bolometric',
    # 'GaiaDR2': "gaia", 'Kepler': 'kepler',
    # 'Generic.Bessell.U': 'bessell', 'Generic.Bessell.B': 'bessell',
    # 'Generic.Bessell.V': 'bessell', 'Generic.Bessell.R': 'bessell',
    # 'Generic.Bessell.I': 'bessell',
    # 'Generic.Stromgren.b': 'stromgren', 'Generic.Stromgren.u': 'stromgren',
    # 'Generic.Stromgren.v': 'stromgren', 'Generic.Stromgren.y': 'stromgren',
    # 'SLOAN.SDSS.g': 'sdss', 'SLOAN.SDSS.i': 'sdss', 'SLOAN.SDSS.r': 'sdss', 'SLOAN.SDSS.u': 'sdss',
    # 'SLOAN.SDSS.z': 'sdss'
}


def get_vh_filename(metallicity):
    s_mh = utils.numeric_metallicity_to_string(metallicity)
    return f"limcof_bp_{s_mh}.dat"


def get_elisa_filename(metallicity, law, passband):
    s_mh = utils.numeric_metallicity_to_string(metallicity)
    return f"{law}.{passband}.{s_mh}.csv"


def read_file(filename):
    with open(pjoin(__BASE_PATH__, filename), "r") as f:
        return f.read()


def header_line(t, logg, mh):
    t = int(t)
    logg = float(logg)
    mh = f'-{abs(float(mh))}' if mh < 0 else f'+{abs(float(mh))}'
    return f"Teff = {t} K,  log g = {logg},  [M/H] = {mh}"


def remove_parenthesis(record):
    for p in ["(", ")"]:
        record = str(record).replace(p, "")
    return record


def export_all_to_elisa_format(path):
    for law in ["lin", "log", "sqrt"]:
        for passband, band in __PASSBANDS_MAP__.items():
            for mh in const.METALLICITY_LIST_LD:
                pd_records = pd.DataFrame(columns=__TABLE_HEADERS__[law])
                for t in const.CK_TEMPERATURE_LIST_ATM:
                    for g in const.GRAVITY_LIST_LD:
                        obtained_record = get_record(t, g, mh, band)
                        if utils.is_empty(obtained_record):
                            continue
                        for rec in obtained_record:
                            if passband in rec:
                                rec = rec[passband]
                                try:
                                    df = pd.DataFrame(columns=__TABLE_HEADERS__[law])
                                    df[__TABLE_HEADERS__[law][2:]] = rec[__TABLE_HEADERS__[law][2:]]
                                    df[__TABLE_HEADERS__[law][0:2]] = [t, g]

                                    pd_records = pd.concat((pd_records, df))
                                except KeyError:
                                    pass

                tablename = get_elisa_filename(mh, law, passband)
                print(f"saving table {tablename}")
                pd_records.to_csv(pjoin(path, tablename), index=False)


def get_section(data, header):
    section = list()
    ends_on = "Teff = "
    found_section = False

    for line in data.split('\n'):
        line = str(line).strip()
        if line == header:
            found_section = True
            continue

        if found_section and ends_on in line:
            break
        if found_section and not utils.is_empty(line):
            section.append(line)
    return section


def back_parser(passband, records):
    record = records[-8:]
    return {
        passband: pd.DataFrame.from_dict({k: [v] for k, v in zip(__HEADER__, record)})
    }


def parse_row(row):
    placeholder = list()
    for r in row:
        r = str(r).strip()
        if not utils.is_empty(r):
            placeholder.append(remove_parenthesis(r))
    return placeholder


def remove_first_val_if_passband(passband, record):
    if str(record[0]).lower().startswith(str(passband).lower()):
        record = record[1:]
    return record


def get_record(temperature, logg, metallicity, passband):
    filename = get_vh_filename(metallicity)
    data = read_file(filename)
    looking_for = header_line(temperature, logg, metallicity)
    section = get_section(data, looking_for)

    if passband == 'bolometric':
        return get_bolometric(section)
    elif passband == 'stromgren':
        return get_stromgren(section)
    elif passband == 'sdss':
        return get_sdss(section)
    elif passband == 'gaia':
        return get_gaia(section)
    elif passband == 'kepler':
        return get_kepler(section)
    elif passband == 'bessell':
        return get_bessell(section)


def get_bolometric(data):
    bolometric = list()
    for row in data:
        if str(row).startswith('bolometric'):
            splited = str(row).split(" ")
            bolometric = parse_row(splited)
            break
    return [back_parser('bolometric', bolometric)]


def get_sdss(data):
    sdss = list()
    found_record = False
    for row in data:
        if str(row).lower().startswith('hst'):
            break

        if str(row).lower().startswith('sloan dss') or found_record:
            found_record = True
            row = str(row).split(" ")
            row = parse_row(row)
            row = remove_first_val_if_passband('sloan', row)
            row = remove_first_val_if_passband('dss', row)
            sdss.append(back_parser(VH_TO_ELISA["Sloan DSS"][row[0]], row))
    return sdss


def get_bessell(data):
    bessell = list()
    for row in data:
        if str(row).lower().startswith('bessell'):
            row = str(row).split(" ")
            row = parse_row(row)
            row = remove_first_val_if_passband('bessell', row)
            try:
                bessell.append(back_parser(VH_TO_ELISA["Bessell"][row[0]], row))
            except KeyError:
                continue
    return bessell


def get_gaia(data):
    for row in data:
        if str(row).lower().startswith('gaia (2010)    g'):
            row = str(row).split(" ")
            row = parse_row(row)
            return [back_parser('GaiaDR2', row)]


def get_kepler(data):
    for row in data:
        if str(row).lower().startswith('kepler'):
            row = str(row).split(" ")
            row = parse_row(row)
            return [back_parser('Kepler', row)]


def get_stromgren(data):
    stromgren = list()
    found_record = False
    for row in data:
        if str(row).lower().startswith('johnson'):
            break

        if str(row).lower().startswith('stromgren') or found_record:
            found_record = True
            row = str(row).split(" ")
            row = parse_row(row)
            row = remove_first_val_if_passband('stromgren', row)
            stromgren.append(back_parser(VH_TO_ELISA["Stromgren"][row[0]], row))
    return stromgren


def main():
    export_all_to_elisa_format(pjoin(dirname(dirname(abspath(__file__))), "vh16"))


if __name__ == "__main__":
    main()
