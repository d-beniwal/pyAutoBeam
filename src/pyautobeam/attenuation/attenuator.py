"""Attenuator position -> Cu thickness lookup.

Single source of truth for the mapping from the attenuator position index
used in file names (``att<N>``) to the Cu attenuator thickness in mm.

The position index is 0-based to match the beamline file-naming convention
(``att0`` = no attenuator).  Both :mod:`pyautobeam.attenuation.analysis` and
:mod:`pyautobeam.attenuation.auto_attenuate` import from here so the table
never drifts between them.
"""

# Attenuator position (as written in the file name, att<N>) -> Cu thickness (mm)
_POS_THICKNESS = {
    0: 0.00,
    1: 0.50,
    2: 1.00,
    3: 1.50,
    4: 2.00,
    5: 2.39,
    6: 4.78,
    7: 7.14,
    8: 9.53,
    9: 11.91,
    10: 14.30,
    11: 16.66,
}

ALL_ATTENUATOR_POSITIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def att_thickness_from_pos(pos):
    """Map an attenuator position index to Cu thickness in mm.

    Returns ``None`` for positions not in the table.
    """
    return _POS_THICKNESS.get(pos, None)
