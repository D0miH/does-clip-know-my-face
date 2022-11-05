from abc import ABC
from typing import Any, List


class DatasetInterface(ABC):
    """
    Wrapper for dataset classes that need to implement certain attributes.
    """

    targets: List[Any]
