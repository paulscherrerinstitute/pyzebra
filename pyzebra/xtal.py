import  numpy as np

def z4frgn(wave,ga,nu):
    """CALCULATES DIFFRACTION VECTOR IN LAB SYSTEM FROM GA AND NU

    Args:
        WAVE,GA,NU

    Returns:
        Z4
    """
    sin = np.sin
    cos = np.cos
    pir = 180/np.pi
    gar = ga/pir
    nur = nu/pir
    z4 = [0., 0., 0.]
    z4[0]=( sin(gar)*cos(nur)    )/wave
    z4[1]=( cos(gar)*cos(nur)-1. )/wave
    z4[2]=( sin(nur)             )/wave

    return z4
