import logging
from abc import ABC, abstractmethod


class Checker(ABC):
    def __init__(self, col):
        self.col = col

    @abstractmethod
    def check(self, val):
        pass


class MinMaxChecker(Checker):
    def __init__(self, min_val, max_val, col):
        super().__init__(col)
        self.min = min_val
        self.max = max_val

    def check(self, val):
        try:
            val = int(val)
            return self.min <= val <= self.max
        except Exception:
            logging.error(f"Not int: {val}")
            return False


class OptionsChecker(Checker):
    def __init__(self, options, col):
        super().__init__(col)
        self.options = options

    def check(self, option):
        return option in self.options
