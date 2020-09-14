import numpy as np
from lmfit import Model, Parameters
from scipy import integrate
from scipy.integrate import simps


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def fitccl(
    data, keys, guess, vary, constraints_min, constraints_max, numfit_min=None, numfit_max=None
):
    """Made for fitting of ccl date where 1 peak is expected. Allows for combination of gaussian, lorentzian and linear model combination


    :param data: dictionary after peak fining
    :param keys: name of the measurement in the data dict (i.e. M123)
    :param guess: initial guess for the fitting, if none, some values are added automatically in order (see below)
    :param vary: True if parameter can vary during fitting, False if it to be fixed
    :param numfit_min: minimal value on x axis for numerical integration - if none is centre of gaussian minus 3 sigma
    :param numfit_max: maximal value on x axis for numerical integration - if none is centre of gaussian plus 3 sigma
    :param constraints_min: min constranits value for fit
    :param constraints_max: max constranits value for fit

    :return data dict with additional values

    order for guess, vary, constraints_min, constraints_max
    [Gaussian centre, Gaussian sigma, Gaussian amplitude, Lorentzian centre, Lorentzian sigma, Lorentzian amplitude, background slope, background intercept]
    examples:
    guess = [None, None, 100, None, None, None, 0, None]
    vary = [True, True, True, True, False, True, True,  True]
    constraints_min = [23, None, 50, None, None, None, 0, 0]
    constraints_min = [80, None, 1000, None, None, None, 0, 100]
    """

    if len(data["Measurements"][str(keys)]["peak_indexes"]) != 1:
        print("NO PEAK or more than 1 peak")
        return

    x = list(data["Measurements"][str(keys)]["omega"])
    y = list(data["Measurements"][str(keys)]["counts"])
    peak_index = data["Measurements"][str(keys)]["peak_indexes"]
    peak_height = data["Measurements"][str(keys)]["peak_heights"]
    print("before", constraints_min)
    guess[0] = x[int(peak_index)] if guess[0] is None else guess[0]
    guess[1] = 0.1 if guess[1] is None else guess[1]
    guess[2] = float(peak_height / 10) if guess[2] is None else float(guess[2])
    guess[3] = x[int(peak_index)] if guess[3] is None else guess[3]
    guess[4] = 2 * guess[1] if guess[4] is None else guess[4]
    guess[5] = float(peak_height / 10) if guess[5] is None else float(guess[5])
    guess[6] = 0 if guess[6] is None else guess[6]
    guess[7] = np.median(x) if guess[7] is None else guess[7]
    constraints_min[0] = np.min(x) if constraints_min[0] is None else constraints_min[0]
    constraints_min[3] = np.min(x) if constraints_min[3] is None else constraints_min[3]
    constraints_max[0] = np.max(x) if constraints_max[0] is None else constraints_max[0]
    constraints_max[3] = np.max(x) if constraints_max[3] is None else constraints_max[3]
    print("key", keys)

    print("after", constraints_min)

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def gaussian(x, g_cen, g_width, g_amp):
        """1-d gaussian: gaussian(x, amp, cen, wid)"""
        return (g_amp / (np.sqrt(2.0 * np.pi) * g_width)) * np.exp(
            -((x - g_cen) ** 2) / (2 * g_width ** 2)
        )

    def lorentzian(x, l_cen, l_width, l_amp):
        """1d lorentzian"""
        return (l_amp / (1 + ((1 * x - l_cen) / l_width) ** 2)) / (np.pi * l_width)

    def background(x, slope, intercept):
        """background"""
        return slope * x + intercept

    mod = Model(gaussian) + Model(lorentzian) + Model(background)
    params = Parameters()
    params.add_many(
        ("g_cen", x[int(peak_index)], bool(vary[0]), np.min(x), np.max(x), None, None),
        ("g_width", guess[1], bool(vary[1]), constraints_min[1], constraints_max[1], None, None),
        ("g_amp", guess[2], bool(vary[2]), constraints_min[2], constraints_max[2], None, None),
        ("l_cen", guess[3], bool(vary[3]), np.min(x), np.max(x), None, None),
        ("l_width", guess[4], bool(vary[4]), constraints_min[4], constraints_max[4], None, None),
        ("l_amp", guess[5], bool(vary[5]), constraints_min[5], constraints_max[5], None, None),
        ("slope", guess[6], bool(vary[6]), constraints_min[6], constraints_max[6], None, None),
        ("intercept", guess[7], bool(vary[7]), constraints_min[7], constraints_max[7], None, None),
    )

    result = mod.fit(y, params, x=x)
    print("Chi-sqr: ", result.chisqr)

    comps = result.eval_components()

    gauss_3sigmamin = find_nearest(
        x, result.params["g_cen"].value - 3 * result.params["g_width"].value
    )
    gauss_3sigmamax = find_nearest(
        x, result.params["g_cen"].value + 3 * result.params["g_width"].value
    )
    numfit_min = gauss_3sigmamin if numfit_min is None else find_nearest(x, numfit_min)
    numfit_max = gauss_3sigmamax if numfit_max is None else find_nearest(x, numfit_max)
    it = -1

    while numfit_max == numfit_min:
        it = it + 1
        numfit_min = find_nearest(
            x, result.params["g_cen"].value - 3 * (1 + it / 10) * result.params["g_width"].value
        )
        numfit_max = find_nearest(
            x, result.params["g_cen"].value + 3 * (1 + it / 10) * result.params["g_width"].value
        )

    if x[numfit_min] < np.min(x):
        numfit_min = gauss_3sigmamin
        print("Minimal integration value outside of x range")
    elif x[numfit_min] >= x[numfit_max]:
        numfit_min = gauss_3sigmamin
        print("Minimal integration value higher than maximal")
    else:
        pass
    if x[numfit_max] > np.max(x):
        numfit_max = gauss_3sigmamax
        print("Maximal integration value outside of x range")
    elif x[numfit_max] <= x[numfit_min]:
        numfit_max = gauss_3sigmamax
        print("Maximal integration value lower than minimal")
    else:
        pass
    print(result.params["g_width"].value)
    print(result.params["g_cen"].value)
    num_int_area = simps(y[numfit_min:numfit_max], x[numfit_min:numfit_max])
    num_int_bacground = integrate.quad(
        background,
        x[numfit_min],
        x[numfit_max],
        args=(result.params["slope"].value, result.params["intercept"].value),
    )

    d = {}
    for pars in result.params:
        d[str(pars)] = (result.params[str(pars)].value, result.params[str(pars)].vary)

    d["export_fit"] = False
    d["int_area"] = num_int_area
    d["int_background"] = num_int_bacground
    d["full_report"] = result.fit_report()
    data["Measurements"][str(keys)]["fit"] = d

    return data
