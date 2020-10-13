import math

import numpy as np
from numba import njit
from scipy.optimize import curve_fit

import pyzebra

try:
    from matplotlib import pyplot as plt
except ImportError:
    print("matplotlib is not available")

pi_r = 180 / np.pi


def z4frgn(wave, ga, nu):
    """CALCULATES DIFFRACTION VECTOR IN LAB SYSTEM FROM GA AND NU

    Args:
        WAVE,GA,NU

    Returns:
        Z4
    """
    ga_r = ga / pi_r
    nu_r = nu / pi_r
    z4 = [0.0, 0.0, 0.0]
    z4[0] = (np.sin(ga_r) * np.cos(nu_r)) / wave
    z4[1] = (np.cos(ga_r) * np.cos(nu_r) - 1.0) / wave
    z4[2] = (np.sin(nu_r)) / wave

    return z4


@njit(cache=True)
def phimat(phi):
    """BUSING AND LEVY CONVENTION ROTATION MATRIX FOR PHI OR OMEGA

    Args:
        PHI

    Returns:
        DUM
    """
    ph_r = phi / pi_r

    dum = np.zeros(9).reshape(3, 3)
    dum[0, 0] = np.cos(ph_r)
    dum[0, 1] = np.sin(ph_r)
    dum[1, 0] = -dum[0, 1]
    dum[1, 1] = dum[0, 0]
    dum[2, 2] = 1

    return dum


def z1frnb(wave, ga, nu, om):
    """CALCULATE DIFFRACTION VECTOR Z1 FROM GA, OM, NU, ASSUMING CH=PH=0

    Args:
        WAVE,GA,NU,OM

    Returns:
        Z1
    """
    z4 = z4frgn(wave, ga, nu)
    dum = phimat(phi=om)
    dumt = np.transpose(dum)
    z3 = dumt.dot(z4)

    return z3


@njit(cache=True)
def chimat(chi):
    """BUSING AND LEVY CONVENTION ROTATION MATRIX FOR CHI

    Args:
        CHI

    Returns:
        DUM
    """
    ch_r = chi / pi_r

    dum = np.zeros(9).reshape(3, 3)
    dum[0, 0] = np.cos(ch_r)
    dum[0, 2] = np.sin(ch_r)
    dum[1, 1] = 1
    dum[2, 0] = -dum[0, 2]
    dum[2, 2] = dum[0, 0]

    return dum


@njit(cache=True)
def z1frz3(z3, chi, phi):
    """CALCULATE Z1 = [PHI]T.[CHI]T.Z3

    Args:
        Z3,CH,PH

    Returns:
        Z1
    """
    dum1 = chimat(chi)
    dum2 = np.transpose(dum1)
    z2 = dum2.dot(z3)

    dum1 = phimat(phi)
    dum2 = np.transpose(dum1)
    z1 = dum2.dot(z2)

    return z1


def z1frmd(wave, ga, om, chi, phi, nu):
    """CALCULATE DIFFRACTION VECTOR Z1 FROM CH, PH, GA, OM, NU

    Args:
        CH, PH, GA, OM, NU

    Returns:
        Z1
    """
    z3 = z1frnb(wave, ga, nu, om)
    z1 = z1frz3(z3, chi, phi)

    return z1


@njit(cache=True)
def det2pol(ddist, gammad, nud, x, y):
    """CONVERTS FROM DETECTOR COORDINATES TO POLAR COORDINATES

    Args:
        x,y detector position
        dist, gamma, nu of detector

    Returns:
        gamma, nu polar coordinates
    """
    xnorm = 128
    ynorm = 64
    xpix = 0.734
    ypix = 1.4809

    xobs = (x - xnorm) * xpix
    yobs = (y - ynorm) * ypix
    a = xobs
    b = ddist * np.cos(yobs / ddist)
    z = ddist * np.sin(yobs / ddist)
    d = np.sqrt(a * a + b * b)

    gamma = gammad + np.arctan2(a, b) * pi_r
    nu = nud + np.arctan2(z, d) * pi_r

    return gamma, nu


def eqchph(z1):
    """CALCULATE CHI, PHI TO PUT THE VECTOR Z1 IN THE EQUATORIAL PLANE

    Args:
        z1

    Returns:
        chi, phi
    """
    if z1[0] != 0 or z1[1] != 0:
        ph = np.arctan2(z1[1], z1[0])
        ph = ph * pi_r
        d = np.sqrt(z1[0] * z1[0] + z1[1] * z1[1])
        ch = np.arctan2(z1[2], d)
        ch = ch * pi_r
    else:
        ph = 0
        ch = 90
        if z1[2] < 0:
            ch = -ch

    ch = 180 - ch
    ph = 180 + ph

    return ch, ph


def dandth(wave, z1):
    """CALCULATE D-SPACING (REAL SPACE) AND THETA FROM LENGTH OF Z

    Args:
        wave, z1

    Returns:
        ds, th
    """
    ierr = 0
    dstar = np.sqrt(z1[0] * z1[0] + z1[1] * z1[1] + z1[2] * z1[2])

    if dstar > 0.0001:
        ds = 1 / dstar
        sint = wave * dstar / 2
        if np.abs(sint) <= 1:
            th = np.arcsin(sint) * pi_r
        else:
            ierr = 2
            th = 0
    else:
        ierr = 1
        ds = 0
        th = 0

    return ds, th, ierr


def angs4c(wave, z1, ch2, ph2):
    """CALCULATE 2-THETA, OMEGA (=THETA), CHI, PHI TO PUT THE
       VECTOR Z1 IN THE BISECTING DIFFRACTION CONDITION

    Args:
        wave, z1, ch2, ph2

    Returns:
        tth, om, ch, ph
    """
    ch2, ph2 = eqchph(z1)
    ch = ch2
    ph = ph2
    ds, th, ierr = dandth(wave, z1)
    if ierr == 0:
        om = th
        tth = th * 2
    else:
        tth = 0
        om = 0
        ch = 0
        ph = 0

    return tth, om, ch, ph, ierr


def fixdnu(wave, z1, ch2, ph2, nu):
    """CALCULATE A SETTING CH,PH,GA,OM TO PUT THE DIFFRACTED BEAM AT NU.
       PH PUTS THE DIFFRACTION VECTOR Z1 INTO THE CHI CIRCLE (AS FOR
       BISECTING GEOMETRY), CH BRINGS THE VECTOR TO THE APPROPRIATE NU
       AND OM THEN POSITIONS THE BEAM AT GA.

    Args:
        wave, z1, ch2, ph2

    Returns:
        tth, om, ch, ph
    """
    tth, om, ch, ph, ierr = angs4c(wave, z1, ch2, ph2)
    theta = om
    if ierr != 0:
        ch = 0
        ph = 0
        ga = 0
        om = 0
    else:
        if np.abs(np.cos(nu / pi_r)) > 0.0001:
            cosga = np.cos(tth / pi_r) / np.cos(nu / pi_r)
            if np.abs(cosga) <= 1:
                ga = np.arccos(cosga) * pi_r
                z4 = z4frgn(wave, ga, nu)
                om = np.arctan2(-z4[1], z4[0]) * pi_r
                ch2 = np.arcsin(z4[2] * wave / (2 * np.sin(theta / pi_r))) * pi_r
                ch = ch - ch2
                ch = ch - 360 * np.trunc((np.sign(ch) * 180 + ch) / 360)
            else:
                ierr = -2
                ch = 0
                ph = 0
                ga = 0
                om = 0
        else:
            if theta > 44.99 and theta < 45.01:
                ga = 90
                om = 90
                ch2 = np.sign(nu) * 45
                ch = ch - ch2
                ch = ch - 360 * np.trunc((np.sign(ch) * 180 + ch) / 360)
            else:
                ierr = -1
                ch = 0
                ph = 0
                ga = 0
                om = 0

    return ch, ph, ga, om


# for test run:
#   angtohkl(wave=1.18,ddist=616,gammad=48.66,om=-22.80,ch=0,ph=0,nud=0,x=128,y=64)


def angtohkl(wave, ddist, gammad, om, ch, ph, nud, x, y):
    """finds hkl-indices of a reflection from its position (x,y,angles) at the 2d-detector

    Args:
        gammad, om, ch, ph, nud, xobs, yobs

    Returns:

    """
    # define ub matrix if testing angtohkl(wave=1.18,ddist=616,gammad=48.66,om=-22.80,ch=0,ph=0,nud=0,x=128,y=64) against f90:
    #    ub = np.array([-0.0178803,-0.0749231,0.0282804,-0.0070082,-0.0368001,-0.0577467,0.1609116,-0.0099281,0.0006274]).reshape(3,3)
    ub = np.array(
        [0.04489, 0.02045, -0.2334, -0.06447, 0.00129, -0.16356, -0.00328, 0.2542, 0.0196]
    ).reshape(3, 3)
    print(
        "The input values are: ga=",
        gammad,
        ", om=",
        om,
        ", ch=",
        ch,
        ", ph=",
        ph,
        ", nu=",
        nud,
        ", x=",
        x,
        ", y=",
        y,
    )

    ga, nu = det2pol(ddist, gammad, nud, x, y)

    print(
        "The calculated actual angles are: ga=",
        ga,
        ", om=",
        om,
        ", ch=",
        ch,
        ", ph=",
        ph,
        ", nu=",
        nu,
    )

    z1 = z1frmd(wave, ga, om, ch, ph, nu)

    print("The diffraction vector is:", z1[0], z1[1], z1[2])

    ubinv = np.linalg.inv(ub)

    h = ubinv[0, 0] * z1[0] + ubinv[0, 1] * z1[1] + ubinv[0, 2] * z1[2]
    k = ubinv[1, 0] * z1[0] + ubinv[1, 1] * z1[1] + ubinv[1, 2] * z1[2]
    l = ubinv[2, 0] * z1[0] + ubinv[2, 1] * z1[1] + ubinv[2, 2] * z1[2]

    print("The Miller indexes are:", h, k, l)

    ch2, ph2 = eqchph(z1)
    ch, ph, ga, om = fixdnu(wave, z1, ch2, ph2, nu)

    print(
        "Bisecting angles to put reflection into the detector center: ga=",
        ga,
        ", om=",
        om,
        ", ch=",
        ch,
        ", ph=",
        ph,
        ", nu=",
        nu,
    )


def ang2hkl(wave, ddist, gammad, om, ch, ph, nud, ub, x, y):
    """Calculate hkl-indices of a reflection from its position (x,y,angles) at the 2d-detector
    """
    ga, nu = det2pol(ddist, gammad, nud, x, y)
    z1 = z1frmd(wave, ga, om, ch, ph, nu)
    ubinv = np.linalg.inv(ub)
    hkl = ubinv @ z1

    return hkl


def gauss(x, *p):
    """Defines Gaussian function

    Args:
        A - amplitude, mu - position of the center, sigma - width

    Returns:
        Gaussian function
    """
    A, mu, sigma = p
    return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))


def box_int(file, box):
    """Calculates center of the peak in the NB-geometry angles and Intensity of the peak

    Args:
        file name, box size [x0:xN, y0:yN, fr0:frN]

    Returns:
        gamma, omPeak, nu polar angles, Int and data for 3 fit plots
    """

    dat = pyzebra.read_detector_data(file)

    sttC = dat["pol_angle"][0]
    om = dat["rot_angle"]
    nuC = dat["tlt_angle"][0]
    ddist = dat["ddist"]

    # defining indices
    x0, xN, y0, yN, fr0, frN = box

    # omega fit
    om = dat["rot_angle"][fr0:frN]
    cnts = np.sum(dat["data"][fr0:frN, y0:yN, x0:xN], axis=(1, 2))

    p0 = [1.0, 0.0, 1.0]
    coeff, var_matrix = curve_fit(gauss, range(len(cnts)), cnts, p0=p0)

    frC = fr0 + coeff[1]
    omF = dat["rot_angle"][math.floor(frC)]
    omC = dat["rot_angle"][math.ceil(frC)]
    frStep = frC - math.floor(frC)
    omStep = omC - omF
    omP = omF + omStep * frStep
    Int = coeff[1] * abs(coeff[2] * omStep) * math.sqrt(2) * math.sqrt(np.pi)
    # omega plot
    x_fit = np.linspace(0, len(cnts), 100)
    y_fit = gauss(x_fit, *coeff)
    plt.figure()
    plt.subplot(131)
    plt.plot(range(len(cnts)), cnts)
    plt.plot(x_fit, y_fit)
    plt.ylabel("Intensity in the box")
    plt.xlabel("Frame N of the box")
    label = "om"
    # gamma fit
    sliceXY = dat["data"][fr0:frN, y0:yN, x0:xN]
    sliceXZ = np.sum(sliceXY, axis=1)
    sliceYZ = np.sum(sliceXY, axis=2)

    projX = np.sum(sliceXZ, axis=0)
    p0 = [1.0, 0.0, 1.0]
    coeff, var_matrix = curve_fit(gauss, range(len(projX)), projX, p0=p0)
    x = x0 + coeff[1]
    # gamma plot
    x_fit = np.linspace(0, len(projX), 100)
    y_fit = gauss(x_fit, *coeff)
    plt.subplot(132)
    plt.plot(range(len(projX)), projX)
    plt.plot(x_fit, y_fit)
    plt.ylabel("Intensity in the box")
    plt.xlabel("X-pixel of the box")

    # nu fit
    projY = np.sum(sliceYZ, axis=0)
    p0 = [1.0, 0.0, 1.0]
    coeff, var_matrix = curve_fit(gauss, range(len(projY)), projY, p0=p0)
    y = y0 + coeff[1]
    # nu plot
    x_fit = np.linspace(0, len(projY), 100)
    y_fit = gauss(x_fit, *coeff)
    plt.subplot(133)
    plt.plot(range(len(projY)), projY)
    plt.plot(x_fit, y_fit)
    plt.ylabel("Intensity in the box")
    plt.xlabel("Y-pixel of the box")

    ga, nu = pyzebra.det2pol(ddist, sttC, nuC, x, y)

    return ga[0], omP, nu[0], Int
