import sys

import numpy as np

sys.path.append("../hapod")
import hapod as hp

if __name__ == "__main__":
    s = np.flip(np.linspace(1, 11, 11))
    print(s)

    rmax = hp.get_truncation_rank(s, rank_max=6)
    print(f"rmax {rmax}")
    assert rmax == 6

    rmax = hp.get_truncation_rank(s, magnitude_frac_max=1e-2)
    print(f"rmax {rmax}")
    assert rmax == 11

    rmax = hp.get_truncation_rank(s, res_energy_frac_max=1e-2)
    print(f"rmax {rmax}")
    assert rmax == 8

