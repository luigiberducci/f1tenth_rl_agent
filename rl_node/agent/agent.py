import pathlib
from abc import ABC, abstractmethod
from typing import Dict


class Agent(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def load(self, model: pathlib.Path) -> bool:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_action(self, observation: Dict[str, float], **kwargs) -> Dict[str, float]:
        pass