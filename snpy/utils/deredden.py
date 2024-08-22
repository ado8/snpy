"""
This modules implements several reddening laws:  extinction as a function of
wavelength. They are the following:

- ccm: Cardelli, Clayton & Mathis (1989)
- fm:  Fitzpatrick (1999).
- fm07_full:  The fully-parametrized Fitzpatrick and Masa (2007) function.
- fm07: Average Fitzpatric & Masa (2007) for MW.
- nataf: Experiemental version of CCM from David Nataf.
"""
import numpy as np
from dust_extinction.parameter_averages import (CCM89, D22, F04, F19, F99, G16,
                                                GCC09, M14, O94, VCG04)
from scipy.interpolate import UnivariateSpline, interp1d


def poly(x, c):
    ret = 0
    for i in range(len(c)):
        ret += c[i] * np.power(x, i)
    return ret


def ccm(wave, strict_ccm=0):
    """Returns the Cardelli, Clayton, and Mathis (CCM) reddening curve.
   
   Args:
      wave (float array): wavelength in Angstroms
      strict_ccm (bool): If True, return original CCM (1989), othewise
                         apply updates from O'Donnel (1994)
                        
   Returns:
      2-tupe (a,b):
      The coeffients such that :math:`A_\lambda/A_V = a + b/R_V`
   """
    x = 10000.0 / wave
    # Convert to inverse microns
    a = 0.0 * x
    b = 0.0 * x

    # **************************************************************
    good = np.greater(x, 0.3) * np.less(x, 1.1)  # Infrared
    if np.any(good):
        a = np.where(good, 0.574 * np.power(x, 1.61), a)
        b = np.where(good, -0.527 * np.power(x, 1.61), b)

    # ****************************************************************
    good = np.greater_equal(x, 1.1) * np.less(x, 3.3)  # Optical/NIR
    if np.any(good):  # Use new constants from O'Donnell (1994)
        y = x - 1.82
        if strict_ccm:
            c1 = [
                1.0,
                0.17699,
                -0.50447,
                -0.02427,
                0.72085,  # Original
                0.01979,
                -0.77530,
                0.32999,
            ]  # coefficients
            c2 = [
                0.0,
                1.41338,
                2.28305,
                1.07233,
                -5.38434,  # from CCM89
                -0.62251,
                5.30260,
                -2.09002,
            ]
        else:
            # New coefficents from O'Donnell (1994)
            c1 = [1.0, 0.104, -0.609, 0.701, 1.137, -1.718, -0.827, 1.647, -0.505]
            c2 = [0.0, 1.952, 2.908, -3.989, -7.985, 11.102, 5.491, -10.805, 3.347]

        poly = 0.0
        for i in range(len(c1)):
            poly = poly + c1[i] * np.power(y, i)
        a = np.where(good, poly, a)
        poly = 0.0
        for i in range(len(c2)):
            poly = poly + c2[i] * np.power(y, i)
        b = np.where(good, poly, b)

    # ******************************************************************
    good = np.greater_equal(x, 3.3) * np.less(x, 8)  # Mid-UV
    if np.any(good):
        y = 1.0 * x
        good1 = np.greater(y, 5.9)
        F_a = y * 0.0
        F_b = y * 0.0
        if np.any(good1) > 0:
            y1 = y - 5.9
            F_a = -0.04473 * np.power(y1, 2) - 0.009779 * np.power(y1, 3)
            F_b = 0.2130 * np.power(y1, 2) + 0.1207 * np.power(y1, 3)
            F_a = np.where(good1, F_a, 0.0)
            F_b = np.where(good1, F_b, 0.0)

        a = np.where(
            good,
            1.752 - 0.316 * y - (0.104 / (np.power(y - 4.67, 2) + 0.341)) + F_a,
            a,
        )
        b = np.where(
            good,
            -3.090 + 1.825 * y + (1.206 / (np.power(y - 4.67, 2) + 0.263)) + F_b,
            b,
        )

    #   *******************************

    good = np.greater_equal(x, 8) * np.less_equal(x, 11)  # Far-UV
    if np.any(good):
        y = x - 8.0
        c1 = [-1.073, -0.628, 0.137, -0.070]
        c2 = [13.670, 4.257, -0.420, 0.374]
        poly = 0.0 * y
        for i in range(len(c1)):
            poly = poly + c1[i] * np.power(y, i)
        a = np.where(good, poly, a)
        poly = 0.0 * y
        for i in range(len(c2)):
            poly = poly + c2[i] * np.power(y, i)
        b = np.where(good, poly, b)

    return (a, b)


def fm(wave, R_V=3.1, avglmc=False, lmc2=False):
    """
   Deredden a flux vector using the Fitzpatrick (1999) parameterization.
   The R-dependent Galactic extinction curve is that of Fitzpatrick & Massa
   (Fitzpatrick, 1999, PASP, 111, 63; astro-ph/9809387 ).    
   Parameterization is valid from the IR to the far-UV (3.5 microns to 0.1
   microns).    UV extinction curve is extrapolated down to 912 Angstroms.
   
   Args:
       wave (float array):  wavelength in Angstroms
       R_V (float): The ratio of total to selective extionction
       avglmc (bool): If True, the default fit parameters c1,c2,c3,c4,gamma,x0 
              are set to the average values determined for reddening in the 
              general Large Magellanic Cloud (LMC) field by Misselt et al. 
              (1999, ApJ, 515, 128)
       lmc2 (bool): if True, the fit parameters are set to the values determined
                for the LMC2 field (including 30 Dor) by Misselt et al.
   Returns:
      foat array: :math:`R_\lambda = A_\lambda/E(B-V)`

   """
    x = 10000.0 / wave.astype(np.float64)  # Convert to inverse microns
    curve = x * 0.0

    # Set default values of c1,c2,c3,c4,gamma and x0 parameters
    if lmc2:
        x0, gamma, c4, c3, c1, c1 = (4.626, 1.05, 0.42, 1.92, 1.31, -2.16)
    elif avglmc:
        x0, gamma, c4, c3, c2, c1 = (4.596, 0.91, 0.64, 2.73, 1.11, -1.28)
    else:
        x0, gamma, c3, c4 = (4.596, 0.99, 3.23, 0.41)
        c2 = -0.824 + 4.717 / R_V
        c1 = 2.030 - 3.007 * c2

    # Compute UV portion of A(lambda)/E(B-V) curve using FM fitting function and
    # R-dependent coefficients

    xcutuv = 10000.0 / 2700.0
    xspluv = 10000.0 / np.array([2700.0, 2600.0])
    iuv = np.greater_equal(x, xcutuv)
    N_UV = sum(iuv)
    iopir = np.logical_not(iuv)
    Nopir = sum(iopir)
    if N_UV > 0:
        xuv = np.concatenate([xspluv, x[iuv]])
    else:
        xuv = xspluv

    yuv = c1 + c2 * xuv
    xuv2 = np.power(xuv, 2)
    yuv = yuv + c3 * xuv2 / (np.power(xuv2 - x0 ** 2, 2) + xuv2 * gamma ** 2)
    mxuv = np.where(xuv < 5.9, 5.9, xuv)
    yuv = yuv + c4 * (
        0.5392 * np.power(mxuv - 5.9, 2) + 0.05644 * np.power(mxuv - 5.9, 3)
    )
    yuv = yuv + R_V
    yspluv = yuv[0:2]  # save spline points

    if N_UV > 0:
        curve[iuv] = yuv[2:]  # remove spline points

    # Compute optical portion of A(lambda)/E(B-V) curve
    # using cubic spline anchored in UV, optical, and IR

    xsplopir = np.concatenate(
        [[0], 10000.0 / np.array([26500.0, 12200.0, 6000.0, 5470.0, 4670.0, 4110.0])]
    )
    ysplir = np.array([0.0, 0.26469, 0.82925]) * R_V / 3.1
    ysplop = np.array(
        [
            poly(R_V, [-4.22809e-01, 1.00270, 2.13572e-04]),
            poly(R_V, [-5.13540e-02, 1.00216, -7.35778e-05]),
            poly(R_V, [7.00127e-01, 1.00184, -3.32598e-05]),
            poly(R_V, [1.19456, 1.01707, -5.46959e-03, 7.97809e-04 - 4.45636e-05]),
        ]
    )

    ysplopir = np.concatenate([ysplir, ysplop])

    if Nopir > 0:
        spl = UnivariateSpline(
            np.concatenate([xsplopir, xspluv]), np.concatenate([ysplopir, yspluv]), s=0,
        )
        curve[iopir] = spl(x[iopir])

    return curve


def fm07_full(wave, x0, gamma, c1, c2, c3, c4, c5, o1, o2, o3, R, iscale, ipower=1.84):
    """The fully-parametrized Fitzpatrick & Massa reddening curve.
   
   Args:
       wave (float array):  wavelength in Angstroms
       x0 (float): centroid of 2200 A bump (microns)
       gamma (float): width of 2200 bump (microns)
       c1 (float): Intercept of linear UV extinction
       c2 (float): Slope of linear UV extinction
       c3 (float): Strength of 2200 A bump
       c4 (float): UV curvature
       c5 (float): Onset of UV curvatrue (microns)
       o1,o2,o3 (float): optical spline points
       R (float): A_V/E(B_V)
       iscale: NIR scale
       ipower: power-law index for NIR.

   Returns:
      foat array: E(V-\lambda)/E(B-V)

   """
    x = 10000.0 / wave.astype(np.float64)  # Convert ot inverse microns
    curve = x * 0.0
    xcutuv = 10000.0 / 2700.0
    iuv = np.greater_equal(x, xcutuv)
    N_UV = sum(iuv)
    iopir = np.logical_not(iuv)
    Nopir = sum(iopir)

    # Compute UV portion of A(lambda)/E(B-V) curve using FM fitting function and
    # R-dependent coefficients
    xspluv = 10000.0 / np.array([2700.0, 2600.0])
    x1, x2, x3 = (10000.0 / 3300, 10000.0 / 4000, 10000.0 / 5430)
    if N_UV > 0:
        xuv = np.concatenate([xspluv, x[iuv]])
    else:
        xuv = xspluv

    yuv = c1 + c2 * xuv
    xuv2 = np.power(xuv, 2)
    yuv = yuv + c3 * xuv2 / (np.power(xuv2 - x0 ** 2, 2) + xuv2 * gamma ** 2)
    yuv = yuv + np.greater(xuv, c5) * c4 * np.power(xuv - c5, 2)
    yspluv = yuv[0:2]  # save spline points

    if N_UV > 0:
        curve[iuv] = yuv[2:]  # remove spline points

    # Compute optical portion of A(lambda)/E(B-V) curve
    # using cubic spline anchored in UV, optical, and IR

    xsplir = np.array([0, 0.25, 0.50, 0.75, 1.0])
    ysplir = iscale * np.power(xsplir, ipower) - R
    xsplop = np.array([x3, x2, x1])
    ysplop = np.array([o3, o2, o1])
    xspl = np.concatenate([xsplir, xsplop, xspluv])
    yspl = np.concatenate([ysplir, ysplop, yspluv])

    if Nopir > 0:
        spl = UnivariateSpline(xspl, yspl, k=3, s=0)
        curve[iopir] = spl(x[iopir])
    return curve


def fm07(wave, R_V=3.1):
    """Average reddening law from Fitzpatrick & Massa (2007). This includes
   a variation with R(V) and uses the correlation between R(V) and
   k_NIR to make a one-parameter curve (keeping all other curve parameters
   fixed. Caveats abound.  Read the paper!
   
   Args:
      wave (float array): wavelength in Angstroms
      R_V (float): ratio of total-to-selective absorption.
      
   Returns:
      R_V (float array): A_V/E(B-V)
   """
    # Set default values of fixed constants
    x0, gamma, c1, c2, c3, c4, c5 = (4.592, 0.922, -0.175, 0.807, 2.991, 0.319, 6.097)
    O1, O2, O3 = (2.055, 1.322, 0.000)
    ipower = 1.84
    iscale = -0.83 + 0.63 * R_V
    ElEv = fm07_full(
        wave, x0, gamma, c1, c2, c3, c4, c5, O1, O2, O3, R_V, iscale, ipower
    )

    return ElEv + R_V


def nataf(wave, R_V=3.1, strict_ccm=False):
    """CCM modified by David Nataf, private communication.

   Args:
      wave (float array): wavelength in Angstroms
      strict_ccm (bool): If True, return original CCM (1989), othewise
                         apply updates from O'Donnel (1994)
                        
   Returns:
      2-tupe (a,b):
      The coeffients such that :math:`A_\lambda/A_V = a + b/R_V`
   """
    x = 10000.0 / wave
    # Convert to inverse microns
    a = 0.0 * x
    b = 0.0 * x

    # **************************************************************
    good = np.greater(x, 0.3) * np.less(x, 1.1)  # Infrared
    if len(np.nonzero(good)) > 0:
        alpha = 1.61 - 0.67 * (R_V - 3.1)
        a = np.where(good, 0.574 * np.power(x, alpha), a)
        b = np.where(good, -0.527 * np.power(x, alpha), b)

    # ****************************************************************
    good = np.greater_equal(x, 1.1) * np.less(x, 3.3)  # Optical/NIR
    if len(np.nonzero(good)) > 0:  # Use new constants from O'Donnell (1994)
        y = x - 1.82
        if strict_ccm:
            c1 = [
                1.0,
                0.17699,
                -0.50447,
                -0.02427,
                0.72085,  # Original
                0.01979,
                -0.77530,
                0.32999,
            ]  # coefficients
            c2 = [
                0.0,
                1.41338,
                2.28305,
                1.07233,
                -5.38434,  # from CCM89
                -0.62251,
                5.30260,
                -2.09002,
            ]
        else:
            # New coefficents from O'Donnell (1994)
            c1 = [1.0, 0.104, -0.609, 0.701, 1.137, -1.718, -0.827, 1.647, -0.505]
            c2 = [0.0, 1.952, 2.908, -3.989, -7.985, 11.102, 5.491, -10.805, 3.347]

        poly = 0.0
        for i in range(len(c1)):
            poly = poly + c1[i] * np.power(y, i)
        a = np.where(good, poly, a)
        poly = 0.0
        for i in range(len(c2)):
            poly = poly + c2[i] * np.power(y, i)
        b = np.where(good, poly, b)

    # ******************************************************************
    good = np.greater_equal(x, 3.3) * np.less(x, 8)  # Mid-UV
    if np.any(good):
        y = 1.0 * x
        good1 = np.greater(y, 5.9)
        F_a = y * 0.0
        F_b = y * 0.0
        if np.any(good1):
            y1 = y - 5.9
            F_a = -0.04473 * np.power(y1, 2) - 0.009779 * np.power(y1, 3)
            F_b = 0.2130 * np.power(y1, 2) + 0.1207 * np.power(y1, 3)
            F_a = np.where(good1, F_a, 0.0)
            F_b = np.where(good1, F_b, 0.0)

        a = np.where(
            good,
            1.752 - 0.316 * y - (0.104 / (np.power(y - 4.67, 2) + 0.341)) + F_a,
            a,
        )
        b = np.where(
            good,
            -3.090 + 1.825 * y + (1.206 / (np.power(y - 4.67, 2) + 0.263)) + F_b,
            b,
        )

    #   *******************************

    good = np.greater_equal(x, 8) * np.less_equal(x, 11)  # Far-UV
    if np.any(good):
        y = x - 8.0
        c1 = [-1.073, -0.628, 0.137, -0.070]
        c2 = [13.670, 4.257, -0.420, 0.374]
        poly = 0.0 * y
        for i in range(len(c1)):
            poly = poly + c1[i] * np.power(y, i)
        a = np.where(good, poly, a)
        poly = 0.0 * y
        for i in range(len(c2)):
            poly = poly + c2[i] * np.power(y, i)
        b = np.where(good, poly, b)

    return (a, b)


def unred_fm(wave, flux, ebv, R_V=3.1, z=0, avglmc=False, lmc2=False):
    """Deredden by the Fitzpatrick (1999) Law"""
    R_lambda = fm(wave, R_V, avglmc=avglmc, lmc2=lmc2)
    A_lambda = ebv * R_lambda
    unred_flux = flux * np.power(10.0, 0.4 * A_lambda)
    return unred_flux


def unred_fm07(wave, flux, ebv, R_V=3.1, z=0):
    """Deredden by the Fitzpatrick & Massa (2007) Law"""
    R_lambda = fm07(wave, R_V)
    A_lambda = ebv * R_lambda
    unred_flux = flux * np.power(10.0, 0.4 * A_lambda)
    return unred_flux


def get_dust_ext_model(redlaw):
    redlawlow = redlaw.lower()

    if redlawlow in ("ccm", "ccm89"):
        ext_model = CCM89
    elif redlawlow == "o94":
        ext_model = O94
    elif redlawlow in ("fm", "f99"):
        ext_model = F99
    elif redlawlow == "fm04":
        ext_model = F04
    elif redlawlow == "vcg04":
        ext_model = VCG04
    elif redlawlow == "gcc09":
        ext_model = GCC09
    elif redlawlow == "m14":
        ext_model = M14
    elif redlawlow == "g16":
        ext_model = G16
    elif redlawlow == "f19":
        ext_model = F19
    elif redlawlow == "d22":
        ext_model = D22
    # elif redlaw == "nataf":
    #     a, b = nataf(wave, strict_ccm)
    #     # note CCM gives A_lambda/A_V = a + b/Rv
    #     # therefore, A_lambda = E(B-V)*R_V*(a + b/Rv) = ebv*(a*R_V + b)
    #     A_lambda = ebv * (a * R_V + b)
    #     unred_flux = flux * np.power(10.0, 0.4 * A_lambda)
    else:
        raise ValueError("Unkwown reddening law %s" % redlaw)
    return ext_model


def unred(wave, flux, ebv, R_V=3.1, z=0, redlaw="O94"):
    """
   de-redden (or redden, if you set ebv < 0) a spectrum using color
   excess E(B-V).  Optionally, you can redshift the spectrum. You can
   choose between the Cardelli, Clayton and Mathis, (redlaw='ccm'),
   Fitzpatrick (1999) law (redlaw='fm'), . If
   using ccm, you can either use the default CCM modified by O'Donnel (1994)
   or use the script CCM by specifying strict_ccm=True.

   Args:
      wave (float array): wavelenths in Angstroms
      flux (float array): flux in arbitrary units
      ebv (float): color excess E(B-V)
      R_V (float): ratio of total-to-selective absorption
      z (float): redshift the spectrum by this amount before applying
                 reddening law.
      redlaw (str): Choice of particular reddening law. Currently we have:
                   'ccm':  Cardelli, Clayton & Mathis (1989) + O'Donnel (1994)
                   'fm' or 'f99': Fitzpatrick (1999)
                   'fm07': Fitzpatrick & Masa (2007)
      srict_ccm (bool):  If True and redlaw='ccm', ignore changes by
                        O'Donnel (1994)

   Returns:
      3-tuple:  (uflux, a, b)
                uflux: un-reddened flux
                a,b:  The equivalent of the CCM a and b parameters.
   """

    if z > 0:
        wave = wave*(1 + z)
    elif z < 0:
        wave = wave/(1 + z)
    ext_model = get_dust_ext_model(redlaw)
    mask = np.where((10000/wave > ext_model.x_range[0]) & (10000/wave < ext_model.x_range[1]))[0]

    unred_flux = flux[mask] / ext_model(R_V).extinguish(10000 / wave[mask], Ebv=ebv)
    return (unred_flux, ext_model(R_V)(10000 / wave[mask]) / R_V, mask)


def old_unred(wave, flux, ebv, R_V=3.1, z=0, redlaw="ccm", strict_ccm=0):
    """
   de-redden (or redden, if you set ebv < 0) a spectrum using color
   excess E(B-V).  Optionally, you can redshift the spectrum. You can
   choose between the Cardelli, Clayton and Mathis, (redlaw='ccm'),
   Fitzpatrick (1999) law (redlaw='fm'), . If
   using ccm, you can either use the default CCM modified by O'Donnel (1994)
   or use the script CCM by specifying strict_ccm=True.

   Args:
      wave (float array): wavelenths in Angstroms
      flux (float array): flux in arbitrary units
      ebv (float): color excess E(B-V)
      R_V (float): ratio of total-to-selective absorption
      z (float): redshift the spectrum by this amount before applying
                 reddening law.
      redlaw (str): Choice of particular reddening law. Currently we have:
                   'ccm':  Cardelli, Clayton & Mathis (1989) + O'Donnel (1994)
                   'fm' or 'f99': Fitzpatrick (1999)
                   'fm07': Fitzpatrick & Masa (2007)
      srict_ccm (bool):  If True and redlaw='ccm', ignore changes by
                        O'Donnel (1994)

   Returns:
      3-tuple:  (uflux, a, b)
                uflux: un-reddened flux
                a,b:  The equivalent of the CCM a and b parameters.
   """

    if z > 0:
        wave = wave * (1 + z)

    if redlaw == "ccm":
        a, b = ccm(wave, strict_ccm)
        # note CCM gives A_lambda/A_V = a + b/Rv
        # therefore, A_lambda = E(B-V)*R_V*(a + b/Rv) = ebv*(a*R_V + b)
        A_lambda = ebv * (a * R_V + b)
        unred_flux = flux * np.power(10.0, 0.4 * A_lambda)
    elif redlaw == "nataf":
        a, b = nataf(wave, strict_ccm)
        # note CCM gives A_lambda/A_V = a + b/Rv
        # therefore, A_lambda = E(B-V)*R_V*(a + b/Rv) = ebv*(a*R_V + b)
        A_lambda = ebv * (a * R_V + b)
        unred_flux = flux * np.power(10.0, 0.4 * A_lambda)
    elif redlaw == "fm" or redlaw == "f99":
        R_lambda = fm(wave, R_V, avglmc=False, lmc2=False)
        A_lambda = ebv * R_lambda
        unred_flux = flux * np.power(10.0, 0.4 * A_lambda)
        # This is a little trick to return an a and b like CCM
        a = R_lambda / R_V
        b = R_lambda * 0
    elif redlaw == "fm07":
        R_lambda = fm07(wave, R_V)
        A_lambda = ebv * R_lambda
        unred_flux = flux * np.power(10.0, 0.4 * A_lambda)
        # This is a little trick to return an a and b like CCM
        a = R_lambda / R_V
        b = R_lambda * 0
    else:
        raise ValueError("Unkwown reddening law %s" % redlaw)
    return (unred_flux, a, b)


def R_z(wave, z, redlaw, R_V=3.1):
    """Compute host reddening law for effective observed wavelength wave, 
   at redshift z, assuming R_V."""
    wave = wave / (1 + z)
    ext_model = get_dust_ext_model(redlaw)
    R = ext_model(R_V)(10000 / wave)
    a = R / R_V
    return (R, a, 0)
