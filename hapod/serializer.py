import os
import zipfile
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np


class MatrixSerializer(ABC):
    """
    Abstract base class for loading matrices during hapod
    """
    @abstractmethod
    def peek(self, source: Union[np.ndarray, str]) -> Tuple[Tuple[int], np.dtype]:
        """
        Retrieve the shape and dtype of the source array, 
        possibly without loading the entire file in memory

        Args:
            source (Union[np.ndarray, str]): either a np.ndarray, or a string representation of the source

        Returns:
            Tuple[Tuple[int], np.dtype]: shape and dtype of the source
        """
        pass

    @abstractmethod
    def load(self, source: Union[np.ndarray, str]) -> np.ndarray:
        """
        Loads a np.ndarray from the given source

        Args:
            source (Union[np.ndarray, str]): either a np.ndarray, or a string representation of the source

        Returns:
            np.ndarray: the loaded np.ndarray
        """
        pass

    @abstractmethod
    def store(self, X: np.ndarray, basename: str) -> Union[np.ndarray, str]:
        """
        Store the given array using the given basename if needed.
        Return the source identifier that will be used in the future to load back the array.

        Args:
            X (np.ndarray): the array to store
            basename (str): the basename to interpret and possibly modify

        Returns:
            Union[np.ndarray, str]: either the array, if kept in memory, or a string to be fed to the load method
        """
        pass


class InMemorySerializer(MatrixSerializer):
    def peek(self, source: Union[np.ndarray, str]) -> Tuple[Tuple[int], np.dtype]:
        if isinstance(source, np.ndarray):
            return source.shape, source.dtype

        raise TypeError("Source must be a numpy.ndarray.")

    def load(self, source: Union[np.ndarray, str]) -> np.ndarray:
        if isinstance(source, np.ndarray):
            return source

        raise TypeError("Source must be a numpy.ndarray.")

    def store(self, X: np.ndarray, basename: str) -> Union[np.ndarray, str]:
        return X


class NumpySerializer(MatrixSerializer):
    """
    MatrixLoader specialization to handle numpy .npy and .npz files
    """
    def __init__(self, npz_fieldname: Optional[str] = ""):
        """
        Initialization

        Args:
            npz_fieldname (Optional[str], optional): the name of the array to be loaded when handling .npz files. Defaults to None.
        """
        self.npz_fieldname = npz_fieldname

    def peek(self, source: Union[np.ndarray, str]) -> Tuple[Tuple[int], np.dtype]:
        if isinstance(source, np.ndarray):
            return source.shape, source.dtype

        if isinstance(source, str):
            if not os.path.isfile(source):
                raise FileNotFoundError(f"File not found: {source}")

            if source.endswith(".npz"):
                with zipfile.ZipFile(source, "r") as archive:
                    fname = f"{self.npz_fieldname}.npy"
                    if not self.npz_fieldname or fname not in archive.namelist():
                        raise ValueError(f"Field {self.npz_fieldname} not found in the .npz file.")

                    with archive.open(fname) as fin:
                        magic = np.lib.format.read_magic(fin)
                        if magic[0] != 1:
                            raise ValueError(f"Unsupported .npy format version in {fname}")

                        header = np.lib.format.read_array_header_1_0(fin)
                        return header[0], header[2]

            with open(source, "rb") as fin:
                magic = np.lib.format.read_magic(fin)
                if magic[0] != 1:
                    raise ValueError("Unsupported .npy format version")

                header = np.lib.format.read_array_header_1_0(fin)
                return header[0], header[2]

        raise TypeError("Source must be either a string (file path) or a numpy.ndarray.")

    def load(self, source: Union[np.ndarray, str]) -> np.ndarray:
        if isinstance(source, np.ndarray):
            return source

        if isinstance(source, str):
            if not os.path.isfile(source):
                raise FileNotFoundError(f"File not found: {source}")

            if source.endswith(".npz"):
                content = np.load(source)

                fname = f"{self.npz_fieldname}.npy"
                if not self.npz_fieldname or fname not in content:
                    raise ValueError(f"Field {self.npz_fieldname} not found in the .npz file.")

                return content[fname]

            return np.load(source)

        raise TypeError("Source must be either a string (file path) or a numpy.ndarray.")

    def store(self, X: np.ndarray, basename: str) -> Union[np.ndarray, str]:
        fname = None
        if self.npz_fieldname:
            fname = basename + ".npz"
            args = {self.npz_fieldname: X}
            np.savez_compressed(fname, **args)
        else:
            fname = basename + ".npy"
            np.save(fname, X)

        return fname


#TODO: OpenFOAMSerializer
