"""Plot utilities for the DESI ETC.

Requires that matplotlib is installed.
"""
import datetime
import copy # for shallow copies of matplotlib colormaps

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines

import desietc.util


def plot_image_quality(stacks, meta, size=33, zoom=5, pad=2, dpi=128, interpolation='none', maxline=17):
    # Calculate crops to use, without assuming which cameras are present in stacks.
    gsize, fsize = 0, 0
    for name, stack in stacks.items():
        if name.startswith('GUIDE'):
            if stack[0] is not None:
                gsize = len(stack[0])
        else:
            L, R = stack
            if L[0] is not None:
                fsize = len(L[0])
            elif R[0] is not None:
                fsize = len(R[0])
        if gsize > 0 and fsize > 0:
            break
    if gsize == 0:
        gsize = size
    if gsize == 0:
        fsize = size
    gcrop = gsize - size
    fcrop = fsize - size
    # Initialize PSF measurements.
    M = desietc.util.PSFMeasure(gsize)
    # Initialize the figure.
    gz = (gsize - gcrop) * zoom
    fz = (fsize - fcrop) * zoom
    nguide, nfocus = 6, 4
    width = nguide * gz + (nguide - 1) * pad
    height = gz + 2 * (fz + pad)
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False)
    # Fill the background with white.
    ax = plt.axes((0, 0, 1, 1))
    ax.axis('off')
    ax.imshow(np.ones((height, width, 3)))
    # Calculate the fiber diameter in GFA pixels.
    fiber_diam_um = 107
    pixel_size_um = 15
    #plate_scale_x, plate_scale_y = 70., 76. # microns / arcsec
    fiber_diam_pix = fiber_diam_um / pixel_size_um
    # Define helper functions.
    def imshow(ax, D, W, name):
        ax.axis('off')
        d = D.copy()
        d[W == 0] = np.nan
        vmax = np.nanpercentile(d, 99.5)
        ax.imshow(d, interpolation=interpolation, origin='lower', cmap='magma', vmin=-0.05 * vmax, vmax=vmax)
        ax.text(0, 0, name, transform=ax.transAxes, fontsize=10, color='c',
                verticalalignment='bottom', horizontalalignment='left')
        n = int(name[5])
        angle = np.deg2rad(36 * (n - 2))
        ny, nx = D.shape
        assert ny == nx
        xc, yc = 0.12 * ny, ny - 0.12 * ny
        du = 0.02 * ny * np.cos(angle)
        dv = 0.02 * ny * np.sin(angle)
        #ax.add_line(matplotlib.lines.Line2D([xc + du, xc - 3 * du], [yc - dv, yc + 3 * dv], c='c', lw=1, ls='-'))
        #ax.add_line(matplotlib.lines.Line2D([xc - dv, xc + dv], [yc - du, yc + du], c='c', lw=1, ls='-'))
        xpt = np.array([-4 * du, dv, du, -dv, -4 * du])
        ypt = np.array([4 * dv, du, -dv, -du, 4 * dv])
        ax.add_line(matplotlib.lines.Line2D(xpt + xc, ypt + yc, c='c', lw=1, ls='-'))
    # Plot GUIDEn PSFs along the middle row.
    y = (fz + pad) / height
    dy, dx = gz / height, gz / width
    fwhm_vec, ffrac_vec = [], []
    cropped = slice(gcrop // 2, gsize - gcrop // 2)
    for k, n in enumerate((2, 0, 8, 7, 5, 3)):
        x = (k * gz + (k - 1) * pad) / width
        name = 'GUIDE{0}'.format(n)
        if name in stacks:
            D, W = stacks[name]
            if D is not None:
                ax = plt.axes((x, y, dx, dy))
                xy0 = (gsize - gcrop - 1) / 2
                imshow(ax, D[cropped, cropped], W[cropped, cropped], name)
                # Draw an outline of the fiber.
                fiber = matplotlib.patches.Circle((xy0, xy0), 0.5 * fiber_diam_pix, color='c', ls='-', alpha=0.7, fill=False)
                ax.add_artist(fiber)
                # Calculate and display the PSF FWHM and fiberfrac.
                fwhm, ffrac = M.measure(D, W)
                fwhm_vec.append(fwhm if fwhm > 0 else np.nan)
                ffrac_vec.append(ffrac if ffrac > 0 else np.nan)
    # Plot FOCUSn PSFs along the top and bottom rows.
    yL = 0
    yR = (gz + 2 * pad + fz) / height
    x0 = ((fz + pad) // 2) / width
    dy, dx = fz / height, fz / width
    cropped = slice(fcrop // 2, fsize - fcrop // 2)
    for k, n in enumerate((1, 9, -1, 6, 4)):
        x = (k * gz + (k - 1) * pad) / width + x0
        if n < 0:
            xc = x
            continue
        name = 'FOCUS{0}'.format(n)
        if name in stacks:
            L, R = stacks[name]
            if L[0] is not None:
                D, W = L[0][cropped, cropped], L[1][cropped, cropped]
                ax = plt.axes((x, yL, dx, dy))
                imshow(ax, D, W, name + 'L')
            if R[0] is not None:
                D, W = R[0][cropped, cropped], R[1][cropped, cropped]
                ax = plt.axes((x, yR, dx, dy))
                imshow(ax, D, W, name + 'R')

    # Fill upper title region.
    ax = plt.axes((xc, yR, dx, dy))
    ax.axis('off')
    ax.text(0.5, 0.8, str(meta['NIGHT']), transform=ax.transAxes, fontsize=16, color='k',
            verticalalignment='bottom', horizontalalignment='center', fontweight='bold')
    ax.text(0.5, 0.6, '{0:08d}'.format(meta['EXPID']), transform=ax.transAxes, fontsize=16, color='k',
            verticalalignment='bottom', horizontalalignment='center', fontweight='bold')
    if 'PROGRAM' in meta:
        line1 = meta['PROGRAM'].strip()
        if len(line1) > maxline:
            line2 = line1[maxline:2 * maxline].strip()
            line1 = line1[:maxline].strip()
            y = 0.5
        else:
            y = 0.46
            line2 = None
        ax.text(0.5, y, line1, transform=ax.transAxes, fontsize=8, color='gray',
                verticalalignment='bottom', horizontalalignment='center')
        if line2 is not None:
            ax.text(0.5, y - 0.08, line2, transform=ax.transAxes, fontsize=8, color='gray',
                    verticalalignment='bottom', horizontalalignment='center')
    if 'MJD-OBS' in meta:
        localtime = datetime.datetime(2019, 1, 1) + datetime.timedelta(days=meta['MJD-OBS'] - 58484.0, hours=-7)
        ax.text(0.5, 0.26, localtime.strftime('%H:%M:%S'), transform=ax.transAxes, fontsize=12, color='k',
                verticalalignment='bottom', horizontalalignment='center')
        ax.text(0.5, 0.17, 'local = UTC-7', transform=ax.transAxes, fontsize=8, color='gray',
                verticalalignment='bottom', horizontalalignment='center')
    if 'EXPTIME' in meta:
        ax.text(0.5, 0.01, '{0:.1f}s'.format(meta['EXPTIME']), transform=ax.transAxes, fontsize=12, color='k',
                verticalalignment='bottom', horizontalalignment='center')
    # Add airmass/alt, az?

    # Fill lower title region.
    ax = plt.axes((xc, yL, dx, dy))
    ax.axis('off')
    ax.text(0.5, 0.8, 'FWHM', transform=ax.transAxes, fontsize=12, color='gray',
            verticalalignment='bottom', horizontalalignment='center')
    if len(fwhm_vec) > 0:
        ax.text(0.5, 0.6, '{0:.2f}"'.format(np.nanmedian(fwhm_vec)), transform=ax.transAxes, fontsize=20, color='k',
                verticalalignment='bottom', horizontalalignment='center', fontweight='bold')
    ax.text(0.5, 0.3, 'FFRAC', transform=ax.transAxes, fontsize=12, color='gray',
            verticalalignment='bottom', horizontalalignment='center')
    if len(fwhm_vec) > 0:
        ax.text(0.5, 0.1, '{0:.0f}%'.format(100 * np.nanmedian(ffrac_vec)), transform=ax.transAxes, fontsize=20, color='k',
                verticalalignment='bottom', horizontalalignment='center', fontweight='bold')

    # Fill corner regions.
    xmirror = np.linspace(-0.8, 0.8, 15)
    ymirror = 0.1 * xmirror ** 2 - 0.85
    for k, y in enumerate((yL, yR)):
        ax = plt.axes((0, y, x0, dy))
        ax.axis('off')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.plot([-0.6, 0.6], [0.8, -0.8], 'c:', lw=1)
        ax.plot([-0.6, 0.6], [-0.8, 0.8], 'c:', lw=1)
        if k == 0: # L = focus at z4 < 0 (closer to mirror)
            ax.plot([0.3, 0.6], [-0.4, -0.8], 'c-', lw=1)
            ax.plot([-0.6, -0.3], [-0.8, -0.4], 'c-', lw=1)
        else: # R = focus at z4 > 0 (closer to sky)
            ax.plot([-0.3, 0.6], [0.4, -0.8], 'c-', lw=1)
            ax.plot([-0.6, 0.3], [-0.8, 0.4], 'c-', lw=1)
        # Mirror
        ax.plot(xmirror, ymirror, 'c-', lw=3)

    hexpos = [float(Z) for Z in meta['HEXPOS'].split(',')]
    temp = meta.get('TRUSTEMP', None)
    if len(hexpos) == 6:
        ax = plt.axes((1 - x0, yR, x0, dy))
        ax.axis('off')
        ax.text(0.5, 0.85, 'hex z', transform=ax.transAxes, fontsize=10, color='k',
                verticalalignment='bottom', horizontalalignment='center')
        ax.text(0.5, 0.70, '{0:.0f}$\mu$m'.format(hexpos[2]), transform=ax.transAxes, fontsize=8, color='k',
                verticalalignment='bottom', horizontalalignment='center')
        if temp is not None:
            best = 430 + (7 - temp) * 110
            ax.text(0.5, 0.55, 'auto'.format(temp), transform=ax.transAxes, fontsize=10, color='c',
                    verticalalignment='bottom', horizontalalignment='center')
            ax.text(0.5, 0.40, '{0:.0f}$\mu$m'.format(best), transform=ax.transAxes, fontsize=8, color='c',
                    verticalalignment='bottom', horizontalalignment='center')
            ax.text(0.5, 0.25, 'truss', transform=ax.transAxes, fontsize=10, color='c',
                    verticalalignment='bottom', horizontalalignment='center')
            ax.text(0.5, 0.10, '{0:.1f}C'.format(temp), transform=ax.transAxes, fontsize=8, color='c',
                    verticalalignment='bottom', horizontalalignment='center')

    adc1, adc2 = meta.get('ADC1PHI'), meta.get('ADC2PHI')
    EL, HA, DEC = meta.get('MOUNTEL'), meta.get('MOUNTHA'), meta.get('MOUNTDEC')
    if adc1 is not None and adc2 is not None:
        axt = plt.axes((1 - x0, yL, x0, dy))
        axt.axis('off')
        axt.text(0.5, 0.9, 'ADC1 {0:.0f}$^\circ$'.format(adc1), transform=axt.transAxes, fontsize=7,
                 color='k', verticalalignment='bottom', horizontalalignment='center')
        axt.text(0.5, 0.8, 'ADC2 {0:.0f}$^\circ$'.format(adc2), transform=axt.transAxes, fontsize=7,
                 color='k', verticalalignment='bottom', horizontalalignment='center')
        if EL is not None:
            axt.text(0.5, 0.7, 'ELEV {0:.0f}$^\circ$'.format(EL), transform=axt.transAxes, fontsize=7,
                     color='gray', verticalalignment='bottom', horizontalalignment='center')
        ax = plt.axes((1 - x0, yL, x0, x0 * width / height))
        ax.axis('off')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        r = 0.98
        circle = matplotlib.patches.Circle((0, 0), r, color='lightgray', ls='-', fill=False)
        ax.add_artist(circle)
        # Draw the horizon and zenith.
        ax.plot([-r, r], [0, 0], '-', c='lightgray', lw=1)
        ax.plot([0, 0], [0, r], '-', c='lightgray', lw=1)
        # Draw the actual ADC angles.
        for phi in adc1, adc2:
            phi = np.deg2rad(phi)
            ax.plot([0, r * np.sin(phi)], [0, r * np.cos(phi)], 'k-', lw=1)
        if not (EL is None or HA is None or DEC is None):
            # Following slide 20 of DESI-3522.
            PARA, PHI1, PHI2 = desietc.util.ADCangles(EL, HA, DEC)
            axt.text(0.5, 0.6, 'PARA {0:.0f}$^\circ$'.format(PARA), transform=axt.transAxes, fontsize=7,
                     color='c', verticalalignment='bottom', horizontalalignment='center')
            PARA, PHI1, PHI2, EL = np.deg2rad([PARA, PHI1, PHI2, EL])
            # Draw the ADC angles necessary to cancel atmospheric refraction at HA.
            HORIZON = PARA + 0.5 * np.pi
            for phi in PHI1, PHI2:
                u, v = r * np.cos(phi - HORIZON), r * np.sin(phi - HORIZON)
                ax.plot([-u, u], [v, -v], ':', c='gray', lw=1)
            # Draw the elevation angle.
            u, v = r * np.cos(EL), r * np.sin(EL)
            ax.plot([-u, 0, u], [v, 0, v], '-', c='gray', lw=1)
            # Draw a North pointer at the parallactic angle relative to zenith.
            u, v = 0.1 * r * np.sin(PARA), 0.1 * r * np.cos(PARA)
            ax.plot([-u, v, 6 * u, -v, -u], [-v, -u, 6 * v, u, -v], 'c-', lw=2)

    return fig
