import numpy as np


class BitOps:
    """ Generates mutations and selections of the input vector.

    Parameters
    ----------
    original : 1D array-like
        1D list or numpy array input

    Attributes
    ----------
    original : numpy.array
        Input vector represented as the numpy array
    mutations : numpy.array
        An array where all the mutations are stored

    Methods
    -------
    mutate(n_mut, apply_selection=False, n_selections=1, **kwargs)
        Creates mutations of the original input array

    Examples
    --------
    >>> bit_ops = BitOps(np.array([0.234, -1.23, 12.625]))
    >>> print(bit_ops.mutate(n_mut=3, prob=0.05))
    [[ 0.23400001 -1.23       12.68751527]
     [ 0.23399952 -1.23000048 12.70709229]
     [ 0.234      -1.23000191 12.625     ]]

    >>> bit_ops = BitOps(np.array([0.234, -1.23, 12.625]))
    >>> print(bit_ops.mutate(n_mut=3, prob=0.1,
    >>>                      apply_selection=True, n_selections=1))
    [[ 0.98385157 -1.24571748 25.00109875]
     [ 0.2294679  -0.61695307 12.62555313]
     [ 0.23400015 -1.2300276  12.62524414]
     [ 1.0241443  -1.85433307 34.62207248]
     [ 0.18941724 -0.60176597  3.0043392 ]
     [ 0.23375808 -0.6365991  12.62548436]]
    """

    def __init__(self, original):
        self.original = np.array(original)
        self._int_orig = self._float2int(self.original)
        self.mutations = None
        self._mut_masks = None
        self._in_shape = self.original.shape

    @staticmethod
    def _float2int(in_val: np.array) -> np.array:
        """ Convert float numbers to 64-bit integers

        Parameters
        ----------
        in_val : numpy.array
            List of the float numbers to transform

        Returns
        -------
        np.array
        """
        return np.frombuffer(in_val.astype("float64").tobytes(), dtype="int64")

    @staticmethod
    def _int2float(in_val: np.array) -> np.array:
        """ Convert 64-bit integers to float numbers

        Parameters
        ----------
        in_val : numpy.array
            List of the integer numbers to transform

        Returns
        -------
        np.array
        """
        return np.frombuffer(in_val.astype("int64").tobytes(), dtype="float64")

    def _gen_mut_masks(self, n_mut=5, prob=0.05, length=56, chunk_s=8):
        # Generating chunks and calculating their probabilities
        probs = (1 - prob, prob)
        num_list = np.arange(2 ** chunk_s, dtype="int64")
        bin_list = np.array([f'{d:0{chunk_s}b}' for d in range(2 ** chunk_s)])
        ones_count = np.char.count(bin_list, "1")
        prob_list = probs[0] ** (chunk_s - ones_count) * probs[1] ** ones_count

        # Sampling chunks and concatenating them
        n_masks = n_mut * len(self.original)
        self._mut_masks = np.zeros(n_masks, dtype="int64")
        for i in range(int(length / chunk_s)):
            next_chunk = np.random.choice(num_list, size=n_masks, p=prob_list)
            next_chunk *= 2 ** (i * chunk_s)
            self._mut_masks += next_chunk

        self._mut_masks = self._mut_masks.reshape((n_mut, len(self.original)))
        return self._mut_masks

    @staticmethod
    def _pseudo_selection(in_arr, n_selections=1):
        rng = np.random.default_rng()
        skeleton = np.tile(in_arr, n_selections).astype("float64")
        skeleton_1, skeleton_2 = skeleton.copy(), skeleton.copy()
        rng.shuffle(skeleton_1)
        rng.shuffle(skeleton_2)
        skeleton += rng.uniform(
            size=(in_arr.shape[0], 1)
        ) * (skeleton_1 - skeleton_2)
        out_arr = skeleton.reshape(in_arr.shape[0] * n_selections,
                                   *in_arr.shape[1:])
        return out_arr

    def mutate(self, n_mut, apply_selection=False, n_selections=1, **kwargs):
        """ Creates mutations of the original array

        Parameters
        ----------
        n_mut : int
            number of mutations
        apply_selection : bool, optional
            Whether to apply selection over the mutated values. Default False.
        n_selections : int, optional
            number of selections to apply over each mutation. Default 1.
        kwargs:
            prob : float, optional
                Probability of mutation. Default 0.05.

            length : int, optional
                Length of the bitstring. Default 56.

            chunk_s : int, optional
                Size of the single chunk. Default 8.
        """
        out_shape = [n_mut, *self._in_shape]
        self._gen_mut_masks(n_mut, **kwargs)
        self.mutations = self._int2float(self._int_orig ^ self._mut_masks)
        self.mutations = np.nan_to_num(self.mutations, nan=1.0,
                                       posinf=100, neginf=-100)
        self.mutations = self.mutations.reshape(out_shape)
        if apply_selection:
            selection_res = self._pseudo_selection(self.mutations, n_selections)
            self.mutations = np.concatenate((self.mutations, selection_res))
            return self.mutations
        return self.mutations


# TODO: remove this in the final version
if __name__ == "__main__":
    bit_ops_ = BitOps(np.array([0.234, -1.23, 12.625]))
    print(bit_ops_.mutate(n_mut=3, apply_selection=True,
                          n_selections=1, prob=0.01))
    # test_bits = BitOps(np.random.normal(0, 1000, 100000))
    # test_bits.mutate(n_mut=100, apply_selection=True, n_selections=10)
