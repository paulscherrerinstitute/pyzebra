import numpy as np
from numba import njit

pi_r = 180 / np.pi


@njit(cache=True)
def z4frgn(wave, ga, nu):
    """CALCULATES DIFFRACTION VECTOR IN LAB SYSTEM FROM GA AND NU

    Args:
        WAVE,GA,NU

    Returns:
        Z4
    """
    ga_r = ga / pi_r
    nu_r = nu / pi_r
    z4 = [np.sin(ga_r) * np.cos(nu_r), np.cos(ga_r) * np.cos(nu_r) - 1.0, np.sin(nu_r)]

    return np.array(z4) / wave


@njit(cache=True)
def phimat_T(phi):
    """TRANSPOSED BUSING AND LEVY CONVENTION ROTATION MATRIX FOR PHI OR OMEGA

    Args:
        PHI

    Returns:
        DUM_T
    """
    ph_r = phi / pi_r

    dum = np.zeros((3, 3))
    dum[0, 0] = np.cos(ph_r)
    dum[1, 0] = np.sin(ph_r)
    dum[0, 1] = -dum[1, 0]
    dum[1, 1] = dum[0, 0]
    dum[2, 2] = 1

    return dum


@njit(cache=True)
def z1frnb(wave, ga, nu, om):
    """CALCULATE DIFFRACTION VECTOR Z1 FROM GA, OM, NU, ASSUMING CH=PH=0

    Args:
        WAVE,GA,NU,OM

    Returns:
        Z1
    """
    z4 = z4frgn(wave, ga, nu)
    z3 = phimat_T(phi=om).dot(z4)

    return z3


@njit(cache=True)
def chimat_T(chi):
    """TRANSPOSED BUSING AND LEVY CONVENTION ROTATION MATRIX FOR CHI

    Args:
        CHI

    Returns:
        DUM_T
    """
    ch_r = chi / pi_r

    dum = np.zeros((3, 3))
    dum[0, 0] = np.cos(ch_r)
    dum[2, 0] = np.sin(ch_r)
    dum[1, 1] = 1
    dum[0, 2] = -dum[2, 0]
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
    z2 = chimat_T(chi).dot(z3)
    z1 = phimat_T(phi).dot(z2)

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


def ang2hkl(wave, ddist, gammad, om, ch, ph, nud, ub_inv, x, y):
    """Calculate hkl-indices of a reflection from its position (x,y,angles) at the 2d-detector"""
    ga, nu = det2pol(ddist, gammad, nud, x, y)
    z1 = z1frmd(wave, ga, om, ch, ph, nu)
    hkl = ub_inv @ z1

    return hkl


def ang2hkl_1d(wave, ga, om, ch, ph, nu, ub_inv):
    """Calculate hkl-indices of a reflection from its position (angles) at the 1d-detector"""
    z1 = z1frmd(wave, ga, om, ch, ph, nu)
    hkl = ub_inv @ z1

    return hkl


def ang_proc(wave, ddist, gammad, om, ch, ph, nud, x, y):
    """Utility function to calculate ch, ph, ga, om"""
    ga, nu = det2pol(ddist, gammad, nud, x, y)
    z1 = z1frmd(wave, ga, om, ch, ph, nu)
    ch2, ph2 = eqchph(z1)
    ch, ph, ga, om = fixdnu(wave, z1, ch2, ph2, nu)

    return ch, ph, ga, om


def gauss(x, *p):
    """Defines Gaussian function

    Args:
        A - amplitude, mu - position of the center, sigma - width

    Returns:
        Gaussian function
    """
    A, mu, sigma = p
    return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))
