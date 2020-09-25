plt.figure(figsize=(20, 10))
plt.plot(x, y, "b-", label="Original data")
plt.plot(x, comps["gaussian"], "r--", label="Gaussian component")
plt.fill_between(x, comps["gaussian"], facecolor="red", alpha=0.4)
plt.plot(x, comps["background"], "g--", label="Line component")
plt.fill_between(x, comps["background"], facecolor="green", alpha=0.4)

plt.fill_between(
    x[numfit_min:numfit_max],
    y[numfit_min:numfit_max],
    facecolor="yellow",
    alpha=0.4,
    label="Integrated area", )

plt.plot(x, result.best_fit, "k-", label="Best fit")
plt.title(
    "%s \n Gaussian: centre = %9.4f, sigma = %9.4f, area = %9.4f \n"
    "background: slope = %9.4f, intercept = %9.4f \n"
    "Int. area = %9.4f +/- %9.4f \n"
    "fit area = %9.4f +/- %9.4f \n"
    "ratio((fit-int)/fit)  = %9.4f"
    % (
        keys,
        result.params["g_cen"].value,
        result.params["g_width"].value,
        result.params["g_amp"].value,
        result.params["slope"].value,
        result.params["intercept"].value,
        int_area.n,
        int_area.s,
        result.params["g_amp"].value,
        result.params["g_amp"].stderr,
        (result.params["g_amp"].value - int_area.n) / result.params["g_amp"].value

    )
)

plt.legend(loc="best")
plt.show()