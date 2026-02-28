import logging
from typing import Sequence, Union

import sys
sys.path.insert(0, '..')
from pdf.AnalyticPDFParametrization import AnalyticPDFParametrization 
from pdf.PODBasis import PODBasis 
logger = logging.getLogger(__name__)

def PDFParametrization(n, typ, basis=None):
    """
    Interface to all PDF parametrizations.

    Parameters
    ----------
    n : int or sequence of int
        - For AnalyticPDFParametrization (Chebyshev/Bernstein): a single int,
          or a sequence of length 1, whose first element is used as n.
        - For PODBasis: a sequence of ints giving the variation indices. If an
          int is given, it is interpreted as [n].
    typ : str
        One of "Chebyshev", "Bernstein", "PODBasis".
    """

    # Normalise n into either an int (for analytic) or list (for PODBasis)
    if isinstance(n, int):
        n_int = n
        n_list = [n]
    else:
        # treat as sequence
        n_list = list(n)
        if len(n_list) == 0:
            raise ValueError("n must not be empty")
        n_int = n_list[0]

    if typ == "Chebyshev":
        if len(n_list) > 1:
            raise ValueError(
                f"Chebyshev AnalyticPDFParametrization expects a single n, "
                f"got multiple values: {n_list}"
            )
        return AnalyticPDFParametrization(n_int, "Chebyshev")

    elif typ == "Bernstein":
        if len(n_list) > 1:
            raise ValueError(
                f"Bernstein AnalyticPDFParametrization expects a single n, "
                f"got multiple values: {n_list}"
            )
        return AnalyticPDFParametrization(n_int, "Bernstein")

    elif typ == "PODBasis":
        # PODBasis wants the full list of variations
        return PODBasis(variations=n_list, var_set=basis)
    else:
        raise ValueError(f"Unknown PDF parametrization type: {typ!r}")

