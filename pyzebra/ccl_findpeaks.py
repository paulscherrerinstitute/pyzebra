import numpy as np
import scipy as sc
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def ccl_findpeaks(
    data, keys, int_threshold=0.8, prominence=50, smooth=False, window_size=7, poly_order=3
):

    """function iterates through the dictionary created by load_cclv2 and locates peaks for each measurement
    args:   data (dictionary from load_cclv2),

            int_threshold - fraction of threshold_intensity/max_intensity, must be positive num between 0 and 1
                        i.e. will only detect peaks above 75% of max intensity

            prominence - defines a drop of values that must be between two peaks, must be positive number
                        i.e. if promimence is 20, it will detect two neigbouring peaks of 300 and 310 intesities,
                        if none of the itermediate values are lower that 290

            smooth  - if true, smooths data by savitzky golay filter, if false - no smoothing

            window_size - window size for savgol filter, must be odd positive integer

            poly_order =  order of the polynomial used in savgol filter, must be positive integer smaller than
            window_size returns: dictionary with following structure:
                                D{M34{  'num_of_peaks': 1,              #num of peaks
                                        'peak_indexes': [20],           # index of peaks in omega array
                                        'peak_heights': [90.],          # height of the peaks (if data vere smoothed
                                                                        its the heigh of the peaks in smoothed data)
    """
    meas = data["Measurements"][keys]

    if type(data) is not dict and data["file_type"] != "ccl":
        print("Data is not a dictionary or was not made from ccl file")

    if not 0 <= int_threshold <= 1:
        int_threshold = 0.8
        print(
            "Invalid value for int_threshold, select value between 0 and 1, new value set to:",
            int_threshold,
        )

    if isinstance(window_size, int) is False or (window_size % 2) == 0 or window_size <= 1:
        window_size = 7
        print(
            "Invalid value for window_size, select positive odd integer, new value set to!:",
            window_size,
        )

    if isinstance(poly_order, int) is False or window_size < poly_order:
        poly_order = 3
        print(
            "Invalid value for poly_order, select positive integer smaller than window_size, new value set to:",
            poly_order,
        )

    if isinstance(prominence, (int, float)) is False and prominence < 0:
        prominence = 50
        print("Invalid value for prominence, select positive number, new value set to:", prominence)

    omega = meas["om"]
    counts = np.array(meas["Counts"])
    if smooth is True:
        itp = interp1d(omega, counts, kind="linear")
        absintensity = [abs(number) for number in counts]
        lowest_intensity = min(absintensity)
        counts[counts < 0] = lowest_intensity
        smooth_peaks = savgol_filter(itp(omega), window_size, poly_order)

    else:
        smooth_peaks = counts

    indexes = sc.signal.find_peaks(
        smooth_peaks, height=int_threshold * max(smooth_peaks), prominence=prominence
    )
    meas["num_of_peaks"] = len(indexes[0])
    meas["peak_indexes"] = indexes[0]
    meas["peak_heights"] = indexes[1]["peak_heights"]
    meas["smooth_peaks"] = smooth_peaks  # smoothed curve

    return data
