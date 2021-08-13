import logging
from abc import ABC, abstractmethod


class Checker(ABC):
    """
    Checker Interface
    """
    def __init__(self, col):
        self.col = col

    @abstractmethod
    def check(self, val):
        pass


class MinMaxChecker(Checker):
    """
    Checker for min-max values
    """
    def __init__(self, min_val, max_val, col):
        super().__init__(col)
        self.min = min_val
        self.max = max_val

    def check(self, val):
        """
        Check if val is a values between min and max
        """
        try:
            val = int(val)
            return self.min <= val <= self.max
        except Exception:
            logging.error(f"Not int: {val}")
            return False


class OptionsChecker(Checker):
    """
    Checker for checking if a val is in a list of options
    """
    def __init__(self, options, col):
        super().__init__(col)
        self.options = options

    def check(self, option):
        """
        Check if the given option is in options
        """
        return option in self.options
