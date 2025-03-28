"""Plot utilities for the DESI ETC.

Requires that matplotlib is installed.
"""

import datetime
import time
import copy  # for shallow copies of matplotlib colormaps

try:
    import DOSlib.logger as logging
except ImportError:
    # Fallback when we are not running as a DOS application.
    import logging

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.patheffects
import matplotlib.dates

import desietc.util


def plot_colorhist(D, ax, imshow, mode="reverse", color="w", alpha=0.75):
    """Draw a hybrid colorbar and histogram."""
    ax.axis("off")
    # Extract parameters of the original imshow.
    cmap = imshow.get_cmap()
    vmin, vmax = imshow.get_clim()
    # Get the pixel dimension of the axis to fill.
    fig = plt.gcf()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = int(round(bbox.width * fig.dpi)), int(round(bbox.height * fig.dpi))
    # Draw the colormap gradient.
    img = np.zeros((height, width, 3))
    xgrad = np.linspace(0, 1, width)
    img[:] = cmap(xgrad)[:, :-1]
    # Superimpose a histogram of pixel values.
    counts, _ = np.histogram(D.reshape(-1), bins=np.linspace(vmin, vmax, width + 1))
    hist_height = ((height - 1) * counts / counts[1:-1].max()).astype(int)
    mask = np.arange(height).reshape(-1, 1) < hist_height
    if mode == "color":
        img[mask] = (1 - alpha) * img[mask] + alpha * np.asarray(
            matplotlib.colors.to_rgb(color)
        )
    elif mode == "reverse":
        cmap_r = cmap.reversed()
        for i, x in enumerate(xgrad):
            img[mask[:, i], i] = cmap_r(x)[:-1]
    elif mode == "complement":
        # https://stackoverflow.com/questions/40233986/
        # python-is-there-a-function-or-formula-to-find-the-complementary-colour-of-a-rgb
        hilo = np.amin(img, axis=2, keepdims=True) + np.amax(img, axis=2, keepdims=True)
        img[mask] = hilo[mask] - img[mask]
    else:
        raise ValueError('Invalid mode "{0}".'.format(mode))
    ax.imshow(img, interpolation="none", origin="lower")


def plot_pixels(
    D,
    label=None,
    colorhist=False,
    zoom=1,
    masked_color="cyan",
    imshow_args={},
    text_args={},
    colorhist_args={},
):
    """Plot pixel data at 1:1 scale with an optional label and colorhist."""
    dpi = (
        100  # value only affects metadata in an output file, not appearance on screen.
    )
    ny, nx = D.shape
    width, height = zoom * nx, zoom * ny
    if colorhist:
        colorhist_height = 32
        height += colorhist_height
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False)
    ax = plt.axes((0, 0, 1, zoom * ny / height))
    args = dict(imshow_args)
    for name, default in dict(
        interpolation="none", origin="lower", cmap="plasma_r"
    ).items():
        if name not in args:
            args[name] = default
    # Set the masked color in the specified colormap.
    cmap = copy.copy(matplotlib.cm.get_cmap(args["cmap"]))
    cmap.set_bad(color=masked_color)
    args["cmap"] = cmap
    # Draw the image.
    I = ax.imshow(D, **args)
    ax.axis("off")
    if label:
        args = dict(text_args)
        for name, default in dict(color="w", fontsize=18).items():
            if name not in args:
                args[name] = default
        outline = [
            matplotlib.patheffects.Stroke(linewidth=1, foreground="k"),
            matplotlib.patheffects.Normal(),
        ]
        text = ax.text(0.01, 0.01 * nx / ny, label, transform=ax.transAxes, **args)
        text.set_path_effects(outline)
    if colorhist:
        axcb = plt.axes((0, zoom * ny / height, 1, colorhist_height / height))
        plot_colorhist(D, axcb, I, **colorhist_args)
    return fig, ax


def plot_data(
    D,
    W,
    downsampling=4,
    zoom=1,
    label=None,
    colorhist=False,
    stamps=[],
    preprocess_args={},
    imshow_args={},
    text_args={},
    colorhist_args={},
):
    """Plot weighted image data using downsampling, optional preprocessing, and decorators."""
    # Downsample the input data.
    D, W = desietc.util.downsample_weighted(D, W, downsampling)
    # Preprocess the data for display.
    D = desietc.util.preprocess(D, W, **preprocess_args)
    ny, nx = D.shape
    # Display the image.
    args = dict(imshow_args)
    if "extent" not in args:
        # Use the input pixel space for the extent, without downsampling.
        args["extent"] = [-0.5, nx * downsampling - 0.5, -0.5, ny * downsampling - 0.5]
    fig, ax = plot_pixels(
        D,
        zoom=zoom,
        label=label,
        colorhist=colorhist,
        imshow_args=args,
        text_args=text_args,
        colorhist_args=colorhist_args,
    )
    outline = [
        matplotlib.patheffects.Stroke(linewidth=1, foreground="k"),
        matplotlib.patheffects.Normal(),
    ]
    for k, stamp in enumerate(stamps):
        yslice, xslice = stamp[:2]
        xlo, xhi = xslice.start, xslice.stop
        ylo, yhi = yslice.start, yslice.stop
        rect = plt.Rectangle((xlo, ylo), xhi - xlo, yhi - ylo, fc="none", ec="w", lw=1)
        ax.add_artist(rect)
        if xhi < nx // 2:
            xtext, halign = xhi, "left"
        else:
            xtext, halign = xlo, "right"
        text = ax.text(
            xtext,
            0.5 * (ylo + yhi),
            str(k),
            fontsize=12,
            color="w",
            va="center",
            ha=halign,
        )
        text.set_path_effects(outline)
    return fig, ax


def save_acquisition_summary(
    mjd,
    exptag,
    psf_model,
    psf_stack,
    fwhm,
    ffrac,
    nstars,
    badfit,
    noisy,
    path,
    show_north=True,
    show_fiber=True,
    zoom=5,
    dpi=128,
    cmap="magma",
    masked_color="gray",
):
    """ """
    # Get the size of the PSF model and stack images.
    shapes = [D.shape for D in psf_stack.values() if D is not None]
    if len(shapes) == 0:
        return
    size = shapes[0][1]
    # Get the number of expected in-focus GFAs.
    names = desietc.gfa.GFACamera.guide_names
    ngfa = len(names)
    # Initialize the figure.
    width = size * zoom * ngfa
    height = size * zoom * 2
    fig, axes = plt.subplots(
        2, ngfa, figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False
    )
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    # Prepare a colormap with our custom ivar=0 color.
    cmap = copy.copy(matplotlib.cm.get_cmap(cmap))
    cmap.set_bad(color=masked_color)
    # Get the colormap scale to use for all images.
    model_sum = {name: psf_model[name].sum() for name in psf_model}
    model_max = np.median(
        [psf_model[camera].max() / model_sum[camera] for camera in psf_model]
    )
    vmin, vmax = -0.1 * model_max, 1.0 * model_max
    # Calculate the image extent.
    # Outline text to ensure that it is visible whatever pixels are below.
    outline = [
        matplotlib.patheffects.Stroke(linewidth=1, foreground="k"),
        matplotlib.patheffects.Normal(),
    ]
    # Calculate the fiber diameter to overlay in GFA pixels.
    fiber_diam_um = 107
    pixel_size_um = 15
    radius = 0.5 * fiber_diam_um / pixel_size_um
    center = ((size - 1) / 2, (size - 1) / 2)
    # Loop over cameras.
    default_norm = np.median([s for s in model_sum.values()])
    for i, name in enumerate(names):
        for ax in axes[:, i]:
            ax.axis("off")
            ax.add_artist(
                plt.Rectangle(
                    [0, 0], 1, 1, fc="k", ec="none", transform=ax.transAxes, zorder=-1
                )
            )
        if name in psf_stack and psf_stack[name] is not None:
            data = psf_stack[name][0].copy()
            norm = model_sum.get(name, default_norm)
            data /= norm
            # do not show ivar=0 pixels
            data[psf_stack[name][1] == 0] = np.nan
            axes[0, i].imshow(
                data,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                interpolation="none",
                origin="lower",
            )
            if show_north:
                # Draw an arrow point north in this GFA's pixel basis.
                n = int(name[5])
                angle = np.deg2rad(36 * (n - 2))
                xc, yc = 0.5 * size, 0.16 * size
                du = 0.02 * size * np.cos(angle)
                dv = 0.02 * size * np.sin(angle)
                xpt = np.array([-4 * du, dv, du, -dv, -4 * du])
                ypt = np.array([4 * dv, du, -dv, -du, 4 * dv])
                axes[0, i].add_line(
                    matplotlib.lines.Line2D(xpt + xc, ypt + yc, c="c", lw=1, ls="-")
                )
        if name in psf_model and psf_model[name] is not None:
            data = psf_model[name]
            data /= model_sum[name]
            axes[1, i].imshow(
                psf_model[name],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                interpolation="bicubic",
                origin="lower",
            )
            if show_fiber:
                # Draw an outline of the fiber.
                fiber = matplotlib.patches.Circle(
                    center, radius, color="c", ls="-", lw=1, alpha=0.7, fill=False
                )
                axes[1, i].add_artist(fiber)
    # Generate a text overlay.
    ax = plt.axes((0, 0, 1, 1))
    ax.axis("off")
    if mjd is not None:
        night = desietc.util.mjd_to_night(mjd)
        localtime = desietc.util.mjd_to_date(mjd, utc_offset=-7)
        center = localtime.strftime("%H:%M:%S") + " (UTC-7)"
    else:
        night = "YYYYMMDD"
        center = "HH:MM:SS (UTC-7)"
    left = f"{night}/{exptag}"
    right = f'FWHM={fwhm:.2f}" ({100*ffrac:.1f}%)'
    for x, ha, label in zip(
        (0, 0.5, 1), ("left", "center", "right"), (left, center, right)
    ):
        text = ax.text(
            x, 0, label, color="w", ha=ha, va="bottom", size=10, transform=ax.transAxes
        )
        text.set_path_effects(outline)
    # Add per-GFA labels.
    xtext = (np.arange(ngfa) + 0.5) / ngfa
    for x, name in zip(xtext, names):
        text = ax.text(
            x,
            0.5,
            name,
            color="w",
            ha="center",
            va="center",
            size=8,
            transform=ax.transAxes,
        )
        text.set_path_effects(outline)
        nstar = nstars[name]
        label = f"{nstar} star"
        if nstar > 1:
            label += "s"
        text = ax.text(
            x,
            0.45,
            label,
            color="w",
            ha="center",
            va="center",
            size=7,
            transform=ax.transAxes,
        )
        text.set_path_effects(outline)
        warn_args = dict(size=10, color="c", fontweight="bold")
        if nstar == 0:
            text = ax.text(
                x,
                0.92,
                "NO STARS?",
                ha="center",
                va="top",
                transform=ax.transAxes,
                **warn_args,
            )
            text.set_path_effects(outline)
        elif name in badfit:
            text = ax.text(
                x,
                0.92,
                "BAD PSF?",
                ha="center",
                va="top",
                transform=ax.transAxes,
                **warn_args,
            )
            text.set_path_effects(outline)
        if name in noisy:
            text = ax.text(
                x,
                1,
                "NOISY?",
                ha="center",
                va="top",
                transform=ax.transAxes,
                **warn_args,
            )
            text.set_path_effects(outline)
    # Save the image.
    plt.savefig(path)
    plt.close(fig)


def plot_measurements(buffer, mjd1, mjd2, ymin=0, resolution=1, label=None, ax=None):
    """Plot measurements spanning (mjd1, mjd2) in the specified buffer."""
    ax = ax or plt.gca()
    # Convert from MJD to minutes after mjd1.
    minutes = lambda mjd: (mjd - mjd1) * 720
    # Plot measurements covering (mjd1, mjd2) with some extra padding.
    xlo, xhi = mjd1 - 3 * buffer.padding, mjd2 + 3 * buffer.padding
    padded = buffer.inside(xlo, xhi)
    used = buffer.inside(mjd1 - buffer.padding, mjd2 + buffer.padding)
    extra = padded & ~used
    for sel, color in ((extra, "lightgray"), (used, "b")):
        x = 0.5 * (buffer.entries["mjd1"][sel] + buffer.entries["mjd2"][sel])
        dx = 0.5 * (buffer.entries["mjd2"][sel] - buffer.entries["mjd1"][sel])
        y = buffer.entries["value"][sel]
        dy = buffer.entries["error"][sel]
        ax.errorbar(
            minutes(x), y, xerr=dx * 720, yerr=dy, fmt=".", color=color, ms=2, lw=1
        )
    # Draw the linear interpolation through the selected points.
    ngrid = int(np.ceil((mjd2 - mjd1) / (resolution / buffer.SECS_PER_DAY)))
    x_grid = np.linspace(mjd1, mjd2, ngrid)
    y_grid = buffer.sample_grid(x_grid)
    ax.fill_between(minutes(x_grid), ymin, y_grid, color="b", lw=0, alpha=0.2)
    # Highlight samples used for the trend.
    sel = buffer.inside(mjd2 - buffer.recent, mjd2)
    x = 0.5 * (buffer.entries["mjd1"][sel] + buffer.entries["mjd2"][sel])
    y = buffer.entries["value"][sel]
    ax.plot(minutes(x), y, "r.", ms=4, zorder=10)
    # Extrapolate the trend.
    offset, slope = buffer.trend(mjd2)
    x = np.array([mjd2, xhi])
    y = offset + slope * (x - mjd2)
    ax.fill_between(minutes(x), ymin, y, color="r", lw=0, alpha=0.2)
    # Draw vertical lines to show the (mjd1, mjd2) interval.
    for xv in (mjd1, mjd2):
        ax.axvline(minutes(xv), c="b", ls="--")
    ax.set_xlim(minutes(xlo), minutes(xhi))
    ax.set_ylim(ymin, None)
    ax.set_xlabel(f"Minutes relative to MJD {mjd1:.6f}")
    if label is not None:
        ax.set_ylabel(label)


def mjd_plot(mjd, *args, axis=None, utc_offset=-7, date_format="%H:%M", **kwargs):
    """Replacement for plt.plot or axis.plot where the x-axis value is an MJD interpreted
    as a local datetime.
    """
    axis = axis or plt.gca()
    plot_epoch = matplotlib.dates.date2num(
        datetime.datetime(1858, 11, 17) + datetime.timedelta(hours=utc_offset)
    )
    axis.plot_date(mjd + plot_epoch, *args, **kwargs)
    axis.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(date_format))
    return axis


def plotSkyCentroidFit(data, name, SKY, plot_fine=True, slow=True, save=None):
    """Make a plot comparing the slow and fast centroid fit methods for SkyCam data.

    See DESI-8945 for examples.
    """
    N = len(SKY.slices[name])
    icamera = SKY.sky_names.index(name)
    mask = desietc.sky.fiber_mask[icamera][:N] > 0

    # Run the fast method
    start = time.time()
    SKY.setraw(
        data,
        name,
        fit_centroids=True,
        fast_centroids=True,
        refit=False,
        return_offsets=False,
    )
    elapsed = time.time() - start
    print(
        f"FAST dx {SKY.fit_dx:.2f} dy {SKY.fit_dy:.2f} flux {np.round(SKY.flux[:N][mask],1)} "
        + f"bg {np.round(SKY.bgfit[:N][mask],2)}\nfluxerr {np.round(SKY.fluxerr[:N][mask],3)} elapsed {1000*elapsed:.1f}ms"
    )

    # Plot the chi-square values tabulated on the fixed coarse grid of centroid offsets.
    dx = SKY.centroid_dx
    dy = SKY.centroid_dy
    xpad = (dx[1] - dx[0]) / 2
    ypad = (dy[1] - dy[0]) / 2
    plt.imshow(
        SKY.grid_chisq,
        origin="lower",
        interpolation="none",
        extent=(dx[0] - xpad, dx[-1] + xpad, dy[0] - ypad, dy[-1] + ypad),
    )
    if plot_fine and SKY.grid_fine_chisq is not None:
        # Superimpose the spline interpolation of the 5x5 grid centered on the coarse-grid min chisq.
        fxpad = (SKY.grid_finedx[1] - SKY.grid_finedx[0]) / 2
        fypad = (SKY.grid_finedy[1] - SKY.grid_finedy[0]) / 2
        plt.imshow(
            SKY.grid_fine_chisq,
            origin="lower",
            interpolation="none",
            extent=(
                SKY.grid_finedx[0] - fxpad,
                SKY.grid_finedx[-1] + fxpad,
                SKY.grid_finedy[0] - fypad,
                SKY.grid_finedy[-1] + fypad,
            ),
        )

    plt.xlim(dx[0] - xpad, dx[-1] + xpad)
    plt.ylim(dy[0] - ypad, dy[-1] + ypad)
    plt.plot(SKY.fit_dx, SKY.fit_dy, "rx", ms=10)
    plt.xlabel("Centroid $\Delta x$ [pixels]")
    plt.ylabel("Centroid $\Delta y$ [pixels]")
    plt.text(0.02, 0.02, name, transform=plt.gca().transAxes, fontsize=16)

    if slow:
        # Run the slow method
        start = time.time()
        SKY.setraw(
            data,
            name,
            fit_centroids=True,
            fast_centroids=False,
            refit=False,
            return_offsets=False,
        )
        elapsed = time.time() - start
        print(
            f"SLOW dx {SKY.fit_dx:.2f} dy {SKY.fit_dy:.2f} flux {np.round(SKY.flux[:N][mask],1)} "
            + f"bg {np.round(SKY.bgfit[:N][mask],2)}\nfluxerr {np.round(SKY.fluxerr[:N][mask],3)} elapsed {1000*elapsed:.1f}ms"
        )

        # Plot centroid-offsets calculated by the slow method for each fiber.
        plt.plot(*SKY.spot_offsets.T, "w.", ms=2)
        # Plot the global centroid offset calculated by the slow method as the mean over fibers.
        plt.plot(SKY.fit_dx, SKY.fit_dy, "w+", ms=10)

    plt.tight_layout()
    if save:
        plt.savefig(save)
