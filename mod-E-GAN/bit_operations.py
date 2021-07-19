import numpy as np
import struct


class BitOps:

    def __init__(self, original):
        self.original = np.array(original)
        self._int_orig = np.array([self.float2int(_float)
                                   for _float in original]).astype("int64")
        self.mutations = None
        self._mut_masks = None
        self.vec_int2float = np.vectorize(self.int2float)
        # self.vec_bin2int = np.vectorize(self.bin2int)

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

    def _gen_mut_masks(self, n_mut=5, prob=0.05, length=56, chunk_s=8):
        probs = (1 - prob, prob)
        b_lists = [f'{d:0{chunk_s}b}' for d in range(2 ** chunk_s)]
        p_list = [probs[0] ** d.count("0") * probs[1] ** d.count("1")
                  for d in b_lists]
        n_masks = n_mut * len(self.original)
        str_mut_masks = np.array(["0"*8]).repeat(n_masks)
        for i in range(int(length / chunk_s)):
            next_chunk = np.random.choice(b_lists, size=n_masks, p=p_list)
            str_mut_masks = np.char.add(str_mut_masks, next_chunk)
        self._mut_masks = np.array([self.bin2int(_bin)
                                    for _bin in str_mut_masks]).astype("int64")
        # TODO: try to solve an numpy OverflowError and vectorized version
        # self._mut_masks = self.vec_bin2int(str_mut_masks).astype("int64")
        self._mut_masks = self._mut_masks.reshape((n_mut, len(self.original)))
        return self._mut_masks

    def mutate(self, **kwargs):
        """ Creates mutations of the original array

        Parameters
        ----------
            kwargs:
                n_mut : int
                    number of mutations

                prob : float
                    probability of mutation

                length : int
                    length of the bitstring

                chunk_s : int
                    size of the chunk of bits to be chosen together
        """
        self._gen_mut_masks(**kwargs)
        self.mutations = self.vec_int2float(self._int_orig ^ self._mut_masks)
        np.nan_to_num(self.mutations, copy=False, nan=1.0, posinf=100, neginf=-100)
        return self.mutations


if __name__ == "__main__":
    bit_ops = BitOps(np.array([0.234, -1.23, 12.625]))
    print(bit_ops.mutate(n_mut=10))
    # test_bits = BitOps(np.random.normal(0, 1000, 1000))
    # test_bits.mutate(100)
