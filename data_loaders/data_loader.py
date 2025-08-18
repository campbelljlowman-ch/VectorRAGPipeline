from abc import ABC, abstractmethod

class DataLoader(ABC):
    def __init__(self):
        """
        Initialize the data loader.
        
        """

    @abstractmethod
    def get_next_chunk(self):
        """
        Load data from the source.
        
        Returns:
            Any: The loaded data.
            Bool: True if more data is available, False otherwise.
        """
        pass
