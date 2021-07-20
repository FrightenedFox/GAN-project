import numpy as np


class BitOps:
    def __init__(self, original):
        self.original = np.array(original)
        self._int_orig = self.float2int(self.original)
        self.mutations = None
        self._mut_masks = None
        self.in_shape = self.original.shape

    @staticmethod
    def float2int(in_val: np.array) -> np.array:
        """ Convert float to 64-bit binary integer

        Parameters
        ----------
            in_val : numpy.array
                Float number to transform

        Returns
        -------
        np.array
            binary integer representation of f (IEEE 754 standard)
        """
        return np.frombuffer(in_val.astype("float64").tobytes(), dtype="int64")

    @staticmethod
    def int2float(in_val: np.array) -> np.array:
        """ Convert binary integer to a float.

        Parameters
        ----------
            in_val : numpy.array
                Binary string to transform.

        Returns
        -------
        np.array
        """
        return np.frombuffer(in_val.astype("int64").tobytes(), dtype="float64")

    def _gen_mut_masks(self, n_mut=5, prob=0.05, length=56, chunk_s=8):
        probs = (1 - prob, prob)
        num_list = np.arange(2 ** chunk_s, dtype="int64")
        bin_list = [f'{d:0{chunk_s}b}' for d in range(2 ** chunk_s)]
        prob_list = [probs[0] ** d.count("0") * probs[1] ** d.count("1")
                     for d in bin_list]
        n_masks = n_mut * len(self.original)
        self._mut_masks = np.zeros(n_masks, dtype="int64")
        for i in range(int(length / chunk_s)):
            next_chunk = np.random.choice(
                num_list, size=n_masks, p=prob_list
            )
            next_chunk *= 2 ** (i * chunk_s)
            self._mut_masks += next_chunk
        self._mut_masks = self._mut_masks.reshape((n_mut, len(self.original)))
        return self._mut_masks

    def mutate(self, n_mut, **kwargs):
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
        out_shape = [n_mut, *self.in_shape]
        self._gen_mut_masks(n_mut, **kwargs)
        self.mutations = self.int2float(self._int_orig ^ self._mut_masks)
        self.mutations = np.nan_to_num(self.mutations.reshape(out_shape),
                                       nan=1.0, posinf=100, neginf=-100)
        return self.mutations


if __name__ == "__main__":
    bit_ops = BitOps(np.array([0.234, -1.23, 12.625]))
    print(bit_ops.mutate(n_mut=10))
    # test_bits = BitOps(np.random.normal(0, 1000, 100000))
    # test_bits.mutate(n_mut=100)
