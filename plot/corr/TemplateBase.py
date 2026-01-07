import numpy as np

class TemplateBase:
    """
    Base class: "something that can provide templates for central and variations".

    Contract:
      - Templates are always returned as 1D numpy arrays (even "scalar" -> length 1).
      - member=0 is the central prediction.
      - member=1..n-1 are variations (PDF eigendirections etc.).
    """

    def __init__(self, name: str = ""):
        self.name = name

    @property
    def n_members(self) -> int:
        raise NotImplementedError

    def get_template(self, member: int) -> np.ndarray:
        raise NotImplementedError

    def get_central(self) -> np.ndarray:
        return self.get_template(0)
