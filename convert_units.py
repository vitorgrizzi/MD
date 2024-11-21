
def g_to_kg(m):
    """Converts from g to Kg"""
    return m * 1.0e-3

def ev_to_joules(E):
    """Converts from [eV] to [J]"""
    return E * 1.60218e-19


def kbar_to_pascal(p):
    """Converts from [kBar] to [N/m²] = [Pa]"""
    return p * 1.0e8


def cm3_to_angstrom3(V):
    """Converts from [cm³] to [Å³]"""
    return V * 1.0e24


def angstrom3_to_meter3(V):
    """Converts from [Å³] to [m³]"""
    return V * 1.0e-30


def hartree_to_ev(E):
    """Converts from [H] to [eV]"""
    return E * 27.2114


