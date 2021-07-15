import numpy as np
import struct


class BitOps:

    def __init__(self, original):
        self.original = np.array(original)
        self._int_orig = np.array([self.float2int(_float)
                                   for _float in original]).astype("int64")
        # self.mutations = np.empty(len(self.original))
        self.mutations = None
        self._mut_masks = None
        self.vec_int2float = np.vectorize(self.int2float)

    @staticmethod
    def bin2float(b: str) -> float:
        """ Convert binary string to a float.

        Parameters
        ----------
            b : str
                Binary string to transform.

        Returns
        -------
        float

        :Authors:
            Javier Domínguez Gómez  <javier.dominguez@optivamedia.com>
        """
        try:    # solves "OverflowError" almost without damaging distribution
            h = int(b, 2).to_bytes(8, byteorder="big", signed=True)
        except OverflowError:
            h = int(b, 2).to_bytes(8, byteorder="big")
        return struct.unpack('>d', h)[0]

    @staticmethod
    def bin2int(b: str) -> int:
        """ Convert binary string to an int.

        Parameters
        ----------
            b : str
                Binary string to transform.

        Returns
        -------
        int
        """
        try:    # solves "OverflowError" almost without damaging distribution
            h = int(b, 2).to_bytes(8, byteorder="big", signed=True)
        except OverflowError:
            h = int(b, 2).to_bytes(8, byteorder="big")
        return struct.unpack('>q', h)[0]

    @staticmethod
    def float2bin(f: float) -> str:
        """ Convert float to 64-bit binary string.

        Parameters
        ----------
            f : float
                Float number to transform

        Returns
        -------
        str
            binary string representation of f in the IEEE 754 standard

        :Authors:
            Javier Domínguez Gómez  <javier.dominguez@optivamedia.com>
        """
        [d] = struct.unpack(">q", struct.pack(">d", f))
        return f'{d:064b}'

    @staticmethod
    def float2int(f: float) -> int:
        """ Convert float to 64-bit binary integer

        Parameters
        ----------
            f : float
                Float number to transform

        Returns
        -------
        int
            binary integer representation of f (IEEE 754 standard)
        """
        return struct.unpack(">q", struct.pack(">d", f))[0]

    @staticmethod
    def int2float(in_val: int) -> float:
        """ Convert binary integer to a float.

        Parameters
        ----------
            in_val : int
                Binary string to transform.

        Returns
        -------
        float
        """
        return struct.unpack(">d", struct.pack('>q', in_val))[0]

    def _gen_mut_masks(self, n_mut, prob=(0.95, 0.05), length=64):
        n_masks = n_mut * len(self.original)
        str_mut_masks = np.random.choice(["0", "1"], size=n_masks, p=prob)
        for i in range(length - 1):
            next_bit = np.random.choice(["0", "1"], size=n_masks, p=prob)
            str_mut_masks = np.char.add(str_mut_masks, next_bit)
        self._mut_masks = np.array([self.bin2int(_bin)
                                    for _bin in str_mut_masks]).astype("int64")
        self._mut_masks = self._mut_masks.reshape((n_mut, len(self.original)))
        return self._mut_masks

    def mutate(self, n_mut):
        # for i in range(len(self.original)):
        #     self.mutations[i] = self.float2int(self.original[i])
        # self.mutations = np.repeat([self.mutations], n_mut, axis=0)
        self._gen_mut_masks(n_mut)
        self.mutations = self.vec_int2float(self._int_orig ^ self._mut_masks)
        return self.mutations


if __name__ == "__main__":
    bit_ops = BitOps(np.array([0.234, -1.23, 12.625]))
    print(bit_ops.mutate(5))

