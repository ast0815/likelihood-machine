"""Utility functions for the work with response matrices."""

from __future__ import division
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from remu import binning
from remu import migration
from remu import plotting

def _block_mahalanobis2(X, mu, inv_cov):
    """Efficiently calculate squared Mahalanobis distance for diagonal block matrix covariances.

    It returns the squared Mahalanobis distance of each block separately.
    To get the total distance, one must sum over these numbers.

    Parameters
    ----------

    X : array_like
        The objects of which the Mahalanobis distance will be calculated.
        Must be of shape ``(n, a, b)``.

    mu : array_like
        The mean values of the distribution. The Mahalanobis distance is
        calculated with respect to these.
        Must be of shape ``(a, b)``.

    inv_cov : array_like
        The inverse of the covariance matrix of the distribution.
        It must be a diagonal block matrix of ``a`` blocks with each block
        of the shape ``(b, b)``. To save space, the off-diagonal 0s are not stored.
        Must be of shape ``(a, b, b)``.

    Returns
    -------

    D_M : ndarray
        The array of squared Mahalanobis distances of shape ``(n, a)``.

    """

    diff = np.asfarray(X) - np.asfarray(mu)
    inv_cov = np.asfarray(inv_cov)
    D_M = np.einsum('...b,...bc,...c', diff, inv_cov, diff)
    return D_M

def mahalanobis_distance(first, second, shape=None, N=None, return_distances_from_mean=False, **kwargs):
    """Calculate the squared Mahalanobis distance of the two matrices for each truth bin.

    Parameters
    ----------

    first, second : ResponseMatrix
        The second ResponseMatrix for the comparison.

    shape : tuple of ints, optional
        The shape of the returned matrix.
        Defaults to ``(#(truth bins),)``.

    N : int, optional
        Number of random matrices to be generated for the calculation.
        This number must be larger than the number of *reco* bins!
        Otherwise the covariances cannot be calculated correctly.
        Defaults to ``#(reco bins) + 100)``.

    return_distances_from_mean : bool, optional
        Also return the ndarray ``distances_from_mean``.

    **kwargs : optional
        Additional keyword arguments are passed through to
        :meth:`generate_random_response_matrices`.

    Returns
    -------

    distance : ndarray
        Array of shape `shape` with the squared Mahalanobis distance
        of the mean difference between the matrices for each truth bin::

            D_M^2( mean(first.random_matrices - second.random_matrices) )

    distances_from_mean : ndarray, optional
        Array of shape ``(N,)+shape`` with the squared Mahalanobis
        distances between the randomly generated matrix differences
        and the mean matrix difference for each truth bin::

            D_M^2( (first.random_matrices - second.random_matrices)
                 - mean(first.random_matrices - second.random_matrices) )

    See also
    --------

    compatibility

    """

    n_reco = first.reco_binning.data_size
    truth_indices = kwargs.pop('truth_indices', list(range(first.truth_binning.data_size)))
    n_truth = len(truth_indices)
    n_bins = n_truth * n_reco
    if N is None:
        N = n_reco + 100

    # Since the detector response is handled completely independently for each truth index,
    # we can calculate the covariance matrices and distances for each one individually.
    distance = []
    distances_from_mean = []
    for i in truth_indices: # TODO: Chunk this when number of reco bins allows it
        indices = [i]

        # Get the *transposed* set of matrices
        first_matrices = first.generate_random_response_matrices(size=N, truth_indices=indices, **kwargs).T

        second_matrices = second.generate_random_response_matrices(size=N, truth_indices=indices, **kwargs).T

        differences = first_matrices - second_matrices

        inv_cov_list = []
        # Actually calculate the inverse covariances
        for j in range(len(indices)):
            cov = np.cov(differences[j])
            inv_cov_list.append(np.linalg.inv(cov))

        null = np.zeros((len(indices), n_reco))
        mean = (first.get_mean_response_matrix_as_ndarray(truth_indices=indices, **kwargs)
            - second.get_mean_response_matrix_as_ndarray(truth_indices=indices, **kwargs)).T

        # Append the distances of the current truth bins to the total
        distance.extend(_block_mahalanobis2([null], mean, inv_cov_list)[0])

        # Calculate distances of throws
        if return_distances_from_mean:
            differences = differences.transpose((2,0,1)) # (truth, reco, N) -> (N, truth, reco)
            distances_from_mean_temp = _block_mahalanobis2(differences, mean, inv_cov_list)
            distances_from_mean_temp = distances_from_mean_temp.transpose() # (N, truth) -> (truth, N)
            distances_from_mean.extend(distances_from_mean_temp)

    # Reshape if asked for
    distance = np.array(distance)
    distances_from_mean = np.array(distances_from_mean).transpose()
    if shape is not None:
        distance = distance.reshape(shape, order='C')
        if return_distances_from_mean:
            distances_from_mean = distances_from_mean.reshape(N+shape, order='C')

    if return_distances_from_mean:
        return distance, distances_from_mean
    else:
        return distance

def _expected_mahalanobis_distance(first, second):
    """Return the expected mahalanobis distance between two matrices.

    Takes shared prior into account.

    Notes
    -----

    The expectation for bins with no true events in either matrix is 0,
    as their prior means are identical.

    When there are no reconstructed events, or no lost events (i.e. 0% or 100%
    efficiency), the difference between the mean efficiency values depends on
    the number of truth events. Depending on how the two matrices are
    generated, this isn't really a random variable.

    The efficiency prior is equivalent to adding two "pseudo observations".

    The smearing prior is kinda equivalent to adding
    min(n_reco_bins, 3**n_reco_variables) events.

    The expectation for bins with lots of reconstructed events in both matrices
    is the number of reconstructed bins, as their priors are negligible.

    We can assume that the priors start losing influence once the number of
    real events is equal to the number of prior events.

    This seems to work reasonably well::


    """

    shape = first.response_binning.bins_shape
    n_reco_bins = shape[0]
    n_reco_vars = len(first.reco_binning.phasespace)

    n_reco_events_1 = first.get_response_entries_as_ndarray(shape=shape).sum(axis=0)
    n_truth_events_1 = first.get_truth_entries_as_ndarray()
    n_lost_events_1 = n_truth_events_1 - n_reco_events_1
    n_reco_events_2 = second.get_response_entries_as_ndarray(shape=shape).sum(axis=0)
    n_truth_events_2 = second.get_truth_entries_as_ndarray()
    n_lost_events_2 = n_truth_events_2 - n_reco_events_2

    prior_events = np.minimum(n_reco_bins, 3**n_reco_vars)
    n_reco_events = np.minimum(n_reco_events_1, n_reco_events_2)
    expectation = n_reco_bins * np.minimum(1., np.asfarray(n_reco_events/(2*prior_events)))

    return expectation

def plot_mahalanobis_distance(first, second, filename=None, plot_expectation=True, **kwargs):
    """Plot the squared Mahalanobis distance ``D_M^2`` between two matrices.

    Parameters
    ----------

    first, second : ResponseMatrix
        The two response matrices for the comparison.
    plot_expectation : bool
        Also plot the expected distance.
    filename :  str, optional
        Save the plot to this location
    **kwargs : optional
        Additional keyword arguments are passed to the plotting function.

    Returns
    -------

    fig : Figure
        The figure that has been plotted on.
    ax : Axes
        The axes that have been plotted into.

    Notes
    -----

    The expected distance is only an estimate based on the statistics in the
    bins. It is not exact and should be treated as a rough guide rather than a
    hard compatibility criterion.

    See also
    --------

    mahalanobis_distance

    """

    from remu import plotting

    if len(first.truth_binning.subbinnings) == 0:
        # Do fancy plots only if there are no subbinnings
        # These mess up the efficiency during subbinning marginalization
        plt = plotting.get_plotter(first.truth_binning)
    else:
        # Otherwise plot bin by bin
        plt = plotting.BinningPlotter(first.truth_binning)

    args = {
        'hatch': None,
        'density': False,
        'margin_function': np.sum,
        }
    args.update(kwargs)

    dist = mahalanobis_distance(first, second)
    if plot_expectation:
        exp = _expected_mahalanobis_distance(first, second)
        plt.plot_array(exp, linestyle='dashed', label="expected", **args)

    plt.plot_array(dist, label=r"$D_M^2$", **args)

    if plot_expectation:
        plt.legend()

    if filename is not None:
        plt.savefig(filename)

    return plt.figax

def compatibility(first, second, N=None, return_all=False,
                  truth_indices=None, min_quality = 0.95, **kwargs):
    """Calculate the compatibility between this and another response matrix.

    Basically, this checks whether the point of "the matrices are
    identical" is an outlier in the distribution of matrix differences as
    defined by the statistical uncertainties of the matrix elements. This
    is done using the Mahalanobis distance as the test statistic. If the
    point "the matrices are identical" is not a reasonable part of the
    distribution, it is not reasonable to assume that the true matrices are
    identical.

    Parameters
    ----------

    second : ResponseMatrix
        The second response matrix.

    N : int, optional
        Number of random matrices to be generated for the calculation.
        This number must be larger than the number of *reco* bins!
        Otherwise the covariances cannot be calculated correctly.
        Defaults to ``#(reco bins) + 100)``.

    return_all : bool, optional
        If ``False``, return only `null_prob_count`, and `null_prob_chi2`.

    truth_indices : list of ints, optional
        Only use the given truth indices to calculate the compatibility. If
        this is not specified, only indices with a minimum "quality" are used.
        This quality requires enough statistics in the bins to make the
        difference between the mean matrices not be dominated by the shared
        prior.

    Returns
    -------

    null_prob_count : float
        The Bayesian p-value evaluated by counting the expected number of
        random matrix differences more extreme than the mean difference.

    null_prob_chi2 : float
        The Bayesian p-value evaluated by assuming a chi-square distribution
        of the squares of Mahalanobis distances.

    null_distance : float, optional
        The squared Mahalanobis distance of the mean differences between the
        two matrices::

            D_M^2( mean(first.random_matrices - second.random_matrices) )

    distances : ndarray, optional
        The set of squared Mahalanobis distances between randomly generated
        matrix differences and the mean matrix difference::

            D_M^2( (first.random_matrices - second.random_matrices)
                 - mean(first.random_matrices - second.random_matrices) )

    df : int, optional
        Degrees of freedom of the assumed chi-squared distribution of the
        squared Mahalanobis distances. This is equal to the number of matrix
        elements that are considered for the calculation::

            df = len(truth_indices) * #(reco_bins in matrix)

    Notes
    -----

    The distribution of matrix differences is evaluated by generating ``N``
    random response matrices from both compared matrices and calculating
    the (n-dimensional) differences. The resulting set of matrix
    differences defines the mean ``mean(differences)`` and the covariance
    matrix ``cov(differences)``. The covariance in turn defines a metric
    for the Mahalanobis distance ``D_M(x)`` on the space of matrix
    differences, where ``x`` is a set of matrix element differences.

    The distance between the mean difference and the Null hypothesis, that
    the two true matrices are identical, is the ``null_distance``::

        null_distance = D_M(0 - mean(differences)) = D_M(mean(differences))

    The compatibility between the matrices is now defined as the Bayesian
    probability that the true difference between the matrices is more
    extreme (has a larger distance from the mean difference) than the
    Null hypothesis. For this, we can just evaluate the set of
    matrix differences that was used to calculate the covariance matrix::

        distances = D_M(differences - mean(differences))
        null_prob_count = np.sum(distances >= null_distance) / distances.size

    It will be 1 if the mean difference between the matrices is 0, and tend
    to 0 when the mean difference between the matrices is far from 0. "Far"
    in this case is determined by the uncertainty, i.e. the covariance, of
    the difference determination.

    In the case of normal distributed differences, the distribution of
    squared Mahalanobis distances becomes chi-squared distributed. The
    numbers of degrees of freedom of that distribution is the number of
    variates, i.e. the number of response matrix elements that are being
    considered. This can be used to calculate a theoretical value for the
    compatibility::

        df = len(truth_indices) * #(reco_bins)
        null_prob_chi2 = chi2.sf(null_distance**2, df)

    Since the distribution of differences is not necessarily Gaussian, this
    is only an estimate. Its advantage is that it is less dependent on the
    number of randomly drawn matrices.

    See also
    --------

    mahalanobis_distance

    """

    n_reco = first.reco_binning.data_size

    if truth_indices is None:
        # If nothing else is specified, only consider truth bins that
        # are considered high enough quality, i.e. the expectation value
        # is close to the high stats limit.

        exp = _expected_mahalanobis_distance(first, second)
        truth_indices = list(sorted(np.argwhere(exp >= min_quality*n_reco).flat))

        if len(truth_indices) == 0:
            raise RuntimeError("No bins with required quality!")

    n_truth = len(truth_indices)
    n_bins = n_truth * n_reco

    # Get the distances for all truth bins
    null_distance, distances = mahalanobis_distance(first, second, N=N,
        truth_indices=truth_indices, return_distances_from_mean=True, **kwargs)

    # Sum up truth bins to total distance
    null_distance = null_distance.sum(axis=-1)
    distances = distances.sum(axis=-1)

    # Calculate theoretical p-value
    null_prob_chi2 = stats.chi2.sf(null_distance, n_bins)
    # Calculate MC p-value
    null_prob_count = float(np.sum(distances >= null_distance)) / distances.size

    if return_all:
        return null_prob_count, null_prob_chi2, null_distance, distances, n_bins
    else:
        return null_prob_count, null_prob_chi2

def plot_compatibility(first, second, filename=None, **kwargs):
    """Plot the compatibility of the two matrices.

    Parameters
    ----------

    first, second : ResponseMatrix
        Two instances of :class:`.ResponseMatrix` for comparison.
    filename : string : optional
        The filename where the plot will be saved.
    **kwargs : optional
        Additional keyword arguments are passed to :func:`compatibility`.

    Returns
    -------

    fig : Figure
        The figure that was used for plotting.
    ax : Axis
        The axis that was used for plotting.

    See also
    --------

    compatibility

    """

    prob_count, prob_chi2, dist, distances, df = compatibility(first, second, return_all=True, **kwargs)

    fig, ax = plt.subplots()
    ax.set_xlabel(r"$D_M^2$")
    nbins = max(min(distances.size // 100, 100), 5)
    ax.hist(distances, nbins, density=True, histtype='stepfilled', label="actual distribution, C=%.3f"%(prob_count,))
    x = np.linspace(df-3*np.sqrt(2*df), df+5*np.sqrt(2*df), 50)
    ax.plot(x, stats.chi2.pdf(x, df), label="$\chi^2$ distribution, C=%.3f"%(prob_chi2,))
    ax.axvline(dist, label="null distance", color='k', linestyle='dashed')
    ax.legend(loc='best', framealpha=0.5)
    if filename is not None:
        fig.savefig(filename)

    return fig, ax

def _merge_suggestions(binning, data_index):
    """Suggest possible bin mergers to improve binning statistics.

    Returns
    -------

    suggestions : list of dict
        Each suggestion is a dict of the form::

            {
            'first_indices': [int, ...],
            'second_indices': [int, ...],
            'binning': Binning,
            'function': function,
            }

    Notes
    -----

    Each suggestions includes a modified Binning, as well as two lists of data
    indices. These are the bins that were merged one-to-one. The returned
    function can do the same merging on an array.

    """

    if data_index is None:
        # Get data index with lowest number of entries
        data_index = np.argmin(binning.entries_array)

    i_bin_min = binning.get_data_bin_index(data_index)

    if i_bin_min in binning.subbinnings:
        return _merge_suggestions_from_subbinnings(binning, i_bin_min, data_index)
    elif len(binning.subbinnings) == 0:
        return _merge_suggestions_without_subbinnings(binning, i_bin_min)
    else:
        raise RuntimeError("Lowest statistics entry is in bin %d of Binning with subbinnings: %r"%(i_bin_min, binning))

def _merge_suggestions_from_subbinnings(binning, i_bin, i_data):
    subbinning = binning.subbinnings[i_bin]
    data_offset = binning.get_bin_data_index(i_bin)
    data_size = subbinning.data_size
    subsuggestions = _merge_suggestions(subbinning, i_data-data_offset)
    suggestions = []
    for sug in subsuggestions:
        # Fix bin numbers
        first = [ data_offset + x for x in sug['first_indices'] ]
        second = [ data_offset + x for x in sug['second_indices'] ]
        new_subbinning = sug['binning']
        # Create a new binning
        marginalized_binning = binning.marginalize_subbinnings([i_bin])
        if new_subbinning.data_size == 1:
            # No need to re-add the subbinning
            new_binning = marginalized_binning
        else:
            new_binning = marginalized_binning.insert_subbinning(i_bin, new_subbinning.clone())
        # Wrap the merging function
        fun0 = sug['function']
        def fun(array, binning=binning.clone(dummy=True), marginalized_binning=marginalized_binning, fun0=fun0,
                data_offset=data_offset, data_size=data_size):
            new_array = binning.marginalize_subbinnings_on_ndarray(array)
            ins = fun0(array[data_offset:data_offset+data_size])
            return marginalized_binning.insert_subbinning_on_ndarray(new_array, i_bin, ins)

        suggestions.append({
            'first_indices': first,
            'second_indices': second,
            'binning': new_binning,
            'function': fun,
            })

    return suggestions

def _merge_suggestions_without_subbinnings(bng, i_bin):
    if isinstance(bng, binning.LinearBinning):
        return _linear_binning_merge_suggestions(bng, i_bin)
    elif isinstance(bng, binning.RectilinearBinning):
        return _rectilinear_binning_merge_suggestions(bng, i_bin)
    else:
        raise RuntimeError("Not able to suggest bin merging for Binning: %r"%(bng))

def _linear_binning_merge_suggestions(binning, i_bin):
    # Suggest merging with the left or right neighbouring bins
    suggestions = []
    if i_bin > 0:
        def fun(array, i_bin=i_bin):
            arr = np.delete(array, [i_bin], axis=0)
            arr[i_bin-1] += array[i_bin]
            return arr
        suggestions.append({
            'first_indices': [i_bin-1,],
            'second_indices': [i_bin,],
            'binning': binning.remove_bin_edges([i_bin]),
            'function': fun,
            })
    if i_bin < binning.nbins-1:
        def fun(array, i_bin=i_bin):
            arr = np.delete(array, [i_bin], axis=0)
            arr[i_bin] += array[i_bin]
            return arr
        suggestions.append({
            'first_indices': [i_bin,],
            'second_indices': [i_bin+1,],
            'binning': binning.remove_bin_edges([i_bin+1]),
            'function': fun,
            })
    return suggestions

def _all_bin_numbers(binning, i_var, i_bin):
    ret = np.arange(binning.nbins)
    ret.shape = binning.bins_shape
    ret = ret[(slice(None),)*i_var + (i_bin, Ellipsis)]
    return ret.flatten()

def _rectilinear_binning_merge_suggestions(binning, i_bin):
    # Suggest merging with the neighbouring bins in all directions
    suggestions = []
    bin_tuple = binning.get_bin_index_tuple(i_bin)
    shape = binning.bins_shape

    for i_var, j_bin in enumerate(bin_tuple):
        size = shape[i_var]
        if j_bin > 0:
            # Merge with lower bin
            def fun(array, i_var=i_var,  j_bin=j_bin):
                array = np.reshape(array, shape + array.shape[1:])
                arr = np.delete(array, [j_bin], axis=i_var)
                arr[(slice(None),)*(i_var) + (j_bin-1, Ellipsis)] += array[(slice(None),)*(i_var) + (j_bin, Ellipsis)]
                return arr.reshape((np.prod(arr.shape[:len(shape)]), ) + array.shape[len(shape):])
            suggestions.append({
                'first_indices': _all_bin_numbers(binning, i_var, j_bin-1),
                'second_indices': _all_bin_numbers(binning, i_var, j_bin),
                'binning': binning.remove_bin_edges({binning.variables[i_var]: [j_bin]}),
                'function': fun,
                })
        if j_bin < size-1:
            # Merge with higher bin
            def fun(array, i_var=i_var, j_bin=j_bin):
                array = np.reshape(array, shape + array.shape[1:])
                arr = np.delete(array, [j_bin], axis=i_var)
                arr[(slice(None),)*(i_var) + (j_bin, Ellipsis)] += array[(slice(None),)*(i_var) + (j_bin, Ellipsis)]
                return arr.reshape((np.prod(arr.shape[:len(shape)]), ) + array.shape[len(shape):])
            suggestions.append({
                'first_indices': _all_bin_numbers(binning, i_var, j_bin),
                'second_indices': _all_bin_numbers(binning, i_var, j_bin+1),
                'binning': binning.remove_bin_edges({binning.variables[i_var]: [j_bin+1]}),
                'function': fun,
                })
    return suggestions

def improve_stats(response_matrix, data_index=None):
    """Reduce the statistical uncertainty by merging some bins in the truth binning.

    Parameters
    ----------

    response_matrix : ResponseMatrix
    data_index : int, optional
        Improve the stats at this truth binning data index. Defaults to lowest
        entries bin.

    Returns
    -------

    new_response_matrix : ResponseMatrix

    Warnings
    --------

    The resulting matrix will have the nuisance/impossible indices set to
    ``[]``!

    Notes
    -----

    Depending on the truth binning, one or more bins will be merged. The bin
    corresponding to `data_index` will be among them. The "direction" of the
    merge (i.e. which neighbouring bin to merge it with) is decided by the
    compatibility of the sets of to-be-merged bins. I.e. the algorithm tries to
    minimize the response difference between the merged bins.

    """

    truth_binning = response_matrix.truth_binning

    # Get merge suggestions
    suggestions = _merge_suggestions(truth_binning, data_index)

    # Judge merge suggestions by compatibility of merged bins
    comp = []
    for sug in suggestions:
        first_indices = sug['first_indices']
        second_indices = sug['second_indices']

        # Create temporary ResponseMatrix objects,
        # consisting only of the bins that are to be merged

        # Need a dummy truth binning for this
        n_bins = len(first_indices)
        temp_truth_binning = binning.LinearBinning('__', np.arange(n_bins+1))

        temp_reco_binning = response_matrix.reco_binning.clone()
        temp_reco_binning.reset()

        response_matrix_1 = migration.ResponseMatrix(temp_reco_binning.clone(), temp_truth_binning.clone())
        response_matrix_2 = migration.ResponseMatrix(temp_reco_binning, temp_truth_binning)

        # Set truth and response values to the bin values
        shape = response_matrix.response_binning.bins_shape

        response_matrix_1.set_truth_values_from_ndarray(response_matrix.get_truth_values_as_ndarray()[first_indices])
        response_matrix_1.set_truth_entries_from_ndarray(response_matrix.get_truth_entries_as_ndarray()[first_indices])
        response_matrix_1.set_truth_sumw2_from_ndarray(response_matrix.get_truth_sumw2_as_ndarray()[first_indices])
        response_matrix_1.set_response_values_from_ndarray(response_matrix.get_response_values_as_ndarray(shape=shape)[:,first_indices])
        response_matrix_1.set_response_entries_from_ndarray(response_matrix.get_response_entries_as_ndarray(shape=shape)[:,first_indices])
        response_matrix_1.set_response_sumw2_from_ndarray(response_matrix.get_response_sumw2_as_ndarray(shape=shape)[:,first_indices])
        response_matrix_1.set_reco_values_from_ndarray(response_matrix.get_response_values_as_ndarray(shape=shape)[:,first_indices].sum(axis=-1))
        response_matrix_1.set_reco_entries_from_ndarray(response_matrix.get_response_entries_as_ndarray(shape=shape)[:,first_indices].sum(axis=-1))
        response_matrix_1.set_reco_sumw2_from_ndarray(response_matrix.get_response_sumw2_as_ndarray(shape=shape)[:,first_indices].sum(axis=-1))

        response_matrix_2.set_truth_values_from_ndarray(response_matrix.get_truth_values_as_ndarray()[second_indices])
        response_matrix_2.set_truth_entries_from_ndarray(response_matrix.get_truth_entries_as_ndarray()[second_indices])
        response_matrix_2.set_truth_sumw2_from_ndarray(response_matrix.get_truth_sumw2_as_ndarray()[second_indices])
        response_matrix_2.set_response_values_from_ndarray(response_matrix.get_response_values_as_ndarray(shape=shape)[:,second_indices])
        response_matrix_2.set_response_entries_from_ndarray(response_matrix.get_response_entries_as_ndarray(shape=shape)[:,second_indices])
        response_matrix_2.set_response_sumw2_from_ndarray(response_matrix.get_response_sumw2_as_ndarray(shape=shape)[:,second_indices])
        response_matrix_2.set_reco_values_from_ndarray(response_matrix.get_response_values_as_ndarray(shape=shape)[:,second_indices].sum(axis=-1))
        response_matrix_2.set_reco_entries_from_ndarray(response_matrix.get_response_entries_as_ndarray(shape=shape)[:,second_indices].sum(axis=-1))
        response_matrix_2.set_reco_sumw2_from_ndarray(response_matrix.get_response_sumw2_as_ndarray(shape=shape)[:,second_indices].sum(axis=-1))

        # Calculate the mahalanobis distance
        D = mahalanobis_distance(response_matrix_1, response_matrix_2).sum()
        comp.append(D)

    # Choose the suggestion with the highest compatibility
    choice = suggestions[np.argmin(comp)]

    # Create the new ResponseMatrix
    new_response_matrix = migration.ResponseMatrix(response_matrix.reco_binning.clone(), choice['binning'])
    # Reco and truth values are already set from the binnings
    # Now need to merge the right response values
    fun = choice['function']
    arr = response_matrix.get_response_values_as_ndarray(shape=shape)
    arr = np.transpose(arr)
    arr = fun(arr)
    new_response_matrix.response_binning.set_values_from_ndarray(arr.T)
    arr = response_matrix.get_response_entries_as_ndarray(shape=shape)
    arr = np.transpose(arr)
    arr = fun(arr)
    new_response_matrix.response_binning.set_entries_from_ndarray(arr.T)
    arr = response_matrix.get_response_sumw2_as_ndarray(shape=shape)
    arr = np.transpose(arr)
    arr = fun(arr)
    new_response_matrix.response_binning.set_sumw2_from_ndarray(arr.T)

    return new_response_matrix

class _ResponsePlotter(plotting.CartesianProductBinningPlotter):
    """Thin wrapper class that defines better axis labels.

    See also
    --------

    .CartesianProductBinningPlotter

    """

    def get_axis_label(self, j_binning):
        """Return the default label for the axis."""
        if j_binning == 0:
            return "Reco Bin #"
        elif j_binning == 1:
            return "Truth Bin #"
        else:
            return "Binning %d Bin #"%(j_binning,)

def plot_mean_response_matrix(response_matrix, filename=None, **kwargs):
    """Plot the smearing and efficiency.

    Parameters
    ----------

    response_matrix : ResponseMatrix
        The thing to plot.
    filename : string : optional
        The filename where the plot will be saved.
    **kwargs : optional
        Additional keyword arguments are passed to the plotting function.

    Returns
    -------

    fig : Figure
        The figure that was used for plotting.
    ax : Axis
        The axis that was used for plotting.

    """

    resp = response_matrix.get_mean_response_matrix_as_ndarray().flatten()
    plt = _ResponsePlotter(response_matrix.response_binning,
                           x_axis_binnings=[1], y_axis_binnings=[0])

    args = {
        'hatch': None,
        'density': False,
        'margin_function': np.sum,
        }
    args.update(kwargs)
    plt.plot_array(resp, **args)
    if filename is not None:
        plt.savefig(filename)
    return plt.figax

def plot_in_bin_variation(response_matrix, filename=None, **kwargs):
    """Plot the maximum in-bin variation vor each truth bin.

    This plots will contain the minimum, maximum, and median marginalization of
    these maximum numbers.

    Parameters
    ----------

    response_matrix : ResponseMatrix
        The thing to plot.
    filename : string : optional
        The filename where the plot will be saved.
    **kwargs : optional
        Additional keyword arguments are passed to the plotting function.

    Returns
    -------

    fig : Figure
        The figure that was used for plotting.
    ax : Axis
        The axis that was used for plotting.

    See also
    --------

    .ResponseMatrix.get_in_bin_variation_as_ndarray

    """

    shape = (response_matrix.reco_binning.data_size, response_matrix.truth_binning.data_size)
    inbin = response_matrix.get_in_bin_variation_as_ndarray(normalize=False, shape=shape)
    inbin = np.max(inbin, axis=0)

    if len(response_matrix.truth_binning.subbinnings) == 0:
        # Do fancy plots only if there are no subbinnings
        # These mess up the efficiency during subbinning marginalization
        plt = plotting.get_plotter(response_matrix.truth_binning)
        funlabs = [
            (np.min, "min"),
            (np.max, "max"),
            (np.median, "median"),
            ]
    else:
        # Otherwise plot bin by bin
        plt = plotting.BinningPlotter(response_matrix.truth_binning)
        funlabs = [
            (np.median, None),
            ]

    for fun, lab in funlabs:
        args = {
            'hatch': None,
            'density': False,
            'margin_function': fun,
            'label': lab,
            }
        args.update(kwargs)
        plt.plot_array(inbin, **args)

    plt.legend()

    if filename is not None:
        plt.savefig(filename)

    return plt.figax

def plot_relative_in_bin_variation(response_matrix, filename=None, **kwargs):
    """Plot the maximum in-bin variation relative to statistical uncertainty.

    This plots will contain the minimum, maximum, and median marginalization of
    these maximum numbers.

    Parameters
    ----------

    response_matrix : ResponseMatrix
        The thing to plot.
    filename : string : optional
        The filename where the plot will be saved.
    **kwargs : optional
        Additional keyword arguments are passed to the plotting function.

    Returns
    -------

    fig : Figure
        The figure that was used for plotting.
    ax : Axis
        The axis that was used for plotting.

    See also
    --------

    .ResponseMatrix.get_in_bin_variation_as_ndarray

    """

    shape = (response_matrix.reco_binning.data_size, response_matrix.truth_binning.data_size)
    inbin = response_matrix.get_in_bin_variation_as_ndarray(normalize=True, shape=shape)
    inbin = np.max(inbin, axis=0)

    if len(response_matrix.truth_binning.subbinnings) == 0:
        # Do fancy plots only if there are no subbinnings
        # These mess up the efficiency during subbinning marginalization
        plt = plotting.get_plotter(response_matrix.truth_binning)
        funlabs = [
            (np.min, "min"),
            (np.max, "max"),
            (np.median, "median"),
            ]
    else:
        # Otherwise plot bin by bin
        plt = plotting.BinningPlotter(response_matrix.truth_binning)
        funlabs = [
            (np.median, None),
            ]

    for fun, lab in funlabs:
        args = {
            'hatch': None,
            'density': False,
            'margin_function': fun,
            'label': lab,
            }
        args.update(kwargs)
        plt.plot_array(inbin, **args)

    plt.legend()

    if filename is not None:
        plt.savefig(filename)

    return plt.figax

def plot_statistical_uncertainty(response_matrix, filename=None, **kwargs):
    """Plot the maximum sqrt(statistical variance) of each truth bin.

    This plots will contain the minimum, maximum, and median marginalization of
    these maximum numbers.

    Parameters
    ----------

    response_matrix : ResponseMatrix
        The thing to plot.
    filename : string : optional
        The filename where the plot will be saved.
    **kwargs : optional
        Additional keyword arguments are passed to the plotting function.

    Returns
    -------

    fig : Figure
        The figure that was used for plotting.
    ax : Axis
        The axis that was used for plotting.

    See also
    --------

    .ResponseMatrix.get_statistical_variance_as_ndarray

    """

    shape = (response_matrix.reco_binning.data_size, response_matrix.truth_binning.data_size)
    stat = np.sqrt(response_matrix.get_statistical_variance_as_ndarray(shape=shape))
    stat = np.max(stat, axis = 0)

    if len(response_matrix.truth_binning.subbinnings) == 0:
        # Do fancy plots only if there are no subbinnings
        # These mess up the efficiency during subbinning marginalization
        plt = plotting.get_plotter(response_matrix.truth_binning)
        funlabs = [
            (np.min, "min"),
            (np.max, "max"),
            (np.median, "median"),
            ]
    else:
        # Otherwise plot bin by bin
        plt = plotting.BinningPlotter(response_matrix.truth_binning)
        funlabs = [
            (np.median, None),
            ]

    for fun, lab in funlabs:
        args = {
            'hatch': None,
            'density': False,
            'margin_function': fun,
            'label': lab,
            }
        args.update(kwargs)
        plt.plot_array(stat, **args)

    plt.legend()

    if filename is not None:
        plt.savefig(filename)

    return plt.figax

def plot_mean_efficiency(response_matrix, filename=None, nuisance_value=0.0, **kwargs):
    """Plot mean efficiencies for all truth bins.

    This ignores the statistical uncertainties of the bin entries. The plot
    will contain the minimum, maximum, and median marginalization of these
    mean efficiencies.

    Parameters
    ----------

    response_matrix : ResponseMatrix
        The thing to plot.
    filename : string : optional
        The filename where the plot will be saved.
    nuisance_value : float, optional
        Nuisance bins are set to this value.
    **kwargs : optional
        Additional keyword arguments are passed to the plotting function.

    Returns
    -------

    fig : Figure
        The figure that was used for plotting.
    ax : Axis
        The axis that was used for plotting.

    """

    nuisance_indices = response_matrix.nuisance_indices

    shape = (response_matrix.reco_binning.data_size, response_matrix.truth_binning.data_size)
    eff = response_matrix.get_mean_response_matrix_as_ndarray(shape=shape)
    eff = np.sum(eff, axis=0)
    eff[nuisance_indices] = nuisance_value

    if len(response_matrix.truth_binning.subbinnings) == 0:
        # Do fancy plots only if there are no subbinnings
        # These mess up the efficiency during subbinning marginalization
        plt = plotting.get_plotter(response_matrix.truth_binning)
        funlabs = [
            (np.min, "min"),
            (np.max, "max"),
            (np.median, "median"),
            ]
    else:
        # Otherwise plot bin by bin
        plt = plotting.BinningPlotter(response_matrix.truth_binning)
        funlabs = [
            (np.median, None),
            ]

    for fun, lab in funlabs:
        args = {
            'hatch': None,
            'density': False,
            'margin_function': fun,
            'label': lab,
            }
        args.update(kwargs)
        plt.plot_array(eff, **args)

    plt.legend()

    if filename is not None:
        plt.savefig(filename)

    return plt.figax
