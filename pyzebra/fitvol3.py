import numpy as np
from lmfit import Model, Parameters
from scipy.integrate import simps
import matplotlib.pyplot as plt
import uncertainties as u
from lmfit.models import GaussianModel
from lmfit.models import VoigtModel
from lmfit.models import PseudoVoigtModel


def bin_data(array, binsize):
    if isinstance(binsize, int) and 0 < binsize < len(array):
        return [
            np.mean(array[binsize * i : binsize * i + binsize])
            for i in range(int(np.ceil(len(array) / binsize)))
        ]
    else:
        print("Binsize need to be positive integer smaller than lenght of array")
        return array


def create_uncertanities(y, y_err):
    # create array with uncertanities for error propagation
    combined = np.array([])
    for i in range(len(y)):
        part = u.ufloat(y[i], y_err[i])
        combined = np.append(combined, part)
    return combined


def find_nearest(array, value):
    # find nearest value and return index
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# predefined peak positions
# peaks = [6.2,  8.1, 9.9, 11.5]
peaks = [23.5, 24.5]
# peaks = [24]
def fitccl(scan, variable="om", peak_type="gauss", binning=None):

    x = list(scan[variable])
    y = list(scan["Counts"])
    peak_centre = np.mean(x)
    if binning is None or binning == 0 or binning == 1:
        x = list(scan["om"])
        y = list(scan["Counts"])
        y_err = list(np.sqrt(y)) if scan.get("sigma", None) is None else list(scan["sigma"])
        print(scan["peak_indexes"])
        if not scan["peak_indexes"]:
            peak_centre = np.mean(x)
        else:
            centre = x[int(scan["peak_indexes"])]
    else:
        x = list(scan["om"])
        if not scan["peak_indexes"]:
            peak_centre = np.mean(x)
        else:
            peak_centre = x[int(scan["peak_indexes"])]
        x = bin_data(x, binning)
        y = list(scan["Counts"])
        y_err = list(np.sqrt(y)) if scan.get("sigma", None) is None else list(scan["sigma"])
        combined = bin_data(create_uncertanities(y, y_err), binning)
        y = [combined[i].n for i in range(len(combined))]
        y_err = [combined[i].s for i in range(len(combined))]

    def background(x, slope, intercept):
        """background"""
        return slope * (x - peak_centre) + intercept

    def gaussian(x, center, g_sigma, amplitude):
        """1-d gaussian: gaussian(x, amp, cen, wid)"""
        return (amplitude / (np.sqrt(2.0 * np.pi) * g_sigma)) * np.exp(
            -((x - center) ** 2) / (2 * g_sigma ** 2)
        )

    def lorentzian(x, center, l_sigma, amplitude):
        """1d lorentzian"""
        return (amplitude / (1 + ((1 * x - center) / l_sigma) ** 2)) / (np.pi * l_sigma)

    def pseudoVoigt1(x, center, g_sigma, amplitude, l_sigma, fraction):
        """PseudoVoight peak with different widths of lorenzian and gaussian"""
        return (1 - fraction) * gaussian(x, center, g_sigma, amplitude) + fraction * (
            lorentzian(x, center, l_sigma, amplitude)
        )

    mod = Model(background)
    params = Parameters()
    params.add_many(
        ("slope", 0, True, None, None, None, None), ("intercept", 0, False, None, None, None, None)
    )
    for i in range(len(peaks)):
        if peak_type == "gauss":
            mod = mod + GaussianModel(prefix="p%d_" % (i + 1))
            params.add(str("p%d_" % (i + 1) + "amplitude"), 20, True, 0, None, None)
            params.add(str("p%d_" % (i + 1) + "center"), peaks[i], True, None, None, None)
            params.add(str("p%d_" % (i + 1) + "sigma"), 0.2, True, 0, 5, None)
        elif peak_type == "voigt":
            mod = mod + VoigtModel(prefix="p%d_" % (i + 1))
            params.add(str("p%d_" % (i + 1) + "amplitude"), 20, True, 0, None, None)
            params.add(str("p%d_" % (i + 1) + "center"), peaks[i], True, None, None, None)
            params.add(str("p%d_" % (i + 1) + "sigma"), 0.2, True, 0, 3, None)
            params.add(str("p%d_" % (i + 1) + "gamma"), 0.2, True, 0, 5, None)
        elif peak_type == "pseudovoigt":
            mod = mod + PseudoVoigtModel(prefix="p%d_" % (i + 1))
            params.add(str("p%d_" % (i + 1) + "amplitude"), 20, True, 0, None, None)
            params.add(str("p%d_" % (i + 1) + "center"), peaks[i], True, None, None, None)
            params.add(str("p%d_" % (i + 1) + "sigma"), 0.2, True, 0, 5, None)
            params.add(str("p%d_" % (i + 1) + "fraction"), 0.5, True, -5, 5, None)
        elif peak_type == "pseudovoigt1":
            mod = mod + Model(pseudoVoigt1, prefix="p%d_" % (i + 1))
            params.add(str("p%d_" % (i + 1) + "amplitude"), 20, True, 0, None, None)
            params.add(str("p%d_" % (i + 1) + "center"), peaks[i], True, None, None, None)
            params.add(str("p%d_" % (i + 1) + "g_sigma"), 0.2, True, 0, 5, None)
            params.add(str("p%d_" % (i + 1) + "l_sigma"), 0.2, True, 0, 5, None)
            params.add(str("p%d_" % (i + 1) + "fraction"), 0.5, True, 0, 1, None)
    # add parameters

    result = mod.fit(
        y, params, weights=[np.abs(1 / y_err[i]) for i in range(len(y_err))], x=x, calc_covar=True
    )

    comps = result.eval_components()

    reportstring = list()
    for keys in result.params:
        if result.params[keys].value is not None:
            str2 = np.around(result.params[keys].value, 3)
        else:
            str2 = 0
        if result.params[keys].stderr is not None:
            str3 = np.around(result.params[keys].stderr, 3)
        else:
            str3 = 0
        reportstring.append("%s = %2.3f +/- %2.3f" % (keys, str2, str3))

    reportstring = "\n".join(reportstring)

    plt.figure(figsize=(20, 10))
    plt.plot(x, result.best_fit, "k-", label="Best fit")

    plt.plot(x, y, "b-", label="Original data")
    plt.plot(x, comps["background"], "g--", label="Line component")
    for i in range(len(peaks)):
        plt.plot(
            x,
            comps[str("p%d_" % (i + 1))],
            "r--",
        )
        plt.fill_between(x, comps[str("p%d_" % (i + 1))], alpha=0.4, label=str("p%d_" % (i + 1)))
    plt.legend()
    plt.text(
        np.min(x),
        np.max(y),
        reportstring,
        fontsize=9,
        verticalalignment="top",
    )
    plt.title(str(peak_type))

    plt.xlabel("Omega [deg]")
    plt.ylabel("Counts [a.u.]")
    plt.show()

    print(result.fit_report())
