# This Python file uses the following encoding: utf-8
import platform
import time
import logging 
import numpy as np
if platform.system() != 'Darwin':
    import cupy as cp
    import cupyx.scipy.ndimage as cps
import cv2
from numba import njit, prange
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

def disk_kernel(r):
    '''
    Create a circular kernel of radius r.

    Parameters:
        r (float): radius of circle

    Returns:
        kernel (numpy.ndarray): a kernel with each entry corresponding
        to the portion of area of a pixel covered by a circle centered
        at the center entry of the kernel with radius 'r'.
    '''
    # compute radius as the distance from the center pixel
    if (r < 0):
        return 0
    elif (r <= 0.5):
        return np.array([[0.0, 0.0, 0.0], [0.0, np.pi * r**2, 0.0], [0.0, 0.0, 0.0]])

    crad = int(np.ceil(r - 0.5))
    x, y = np.meshgrid(np.arange(-crad, crad+1, dtype=np.int32), np.arange(-crad, crad+1, dtype=np.int32))
    maxxy = np.maximum(np.fabs(x), np.fabs(y))
    minxy = np.minimum(np.fabs(x), np.fabs(y))

    # magical arrays
    m1 = np.where(
            r**2 < (maxxy + 0.5)**2 + (minxy - 0.5)**2, 
            minxy - 0.5, 
            np.sqrt(np.maximum(r**2 - (maxxy + 0.5)**2, 0.0))
            #np.sqrt(r**2 - (maxxy + 0.5)**2)
    )

    m2 = np.where(
         r**2 > (maxxy - 0.5)**2 + (minxy + 0.5)**2,
         minxy + 0.5,
         np.sqrt(np.maximum(r**2 - (maxxy - 0.5)**2, 0.0))  # Clip to 0
         #np.sqrt(r**2 - (maxxy - 0.5)**2)
    )

    # a magical set of conditions
    c1 = (maxxy+0.5)**2 + (minxy+0.5)**2 > r**2
    c2 = (maxxy-0.5)**2 + (minxy-0.5)**2 < r**2
    c3 = (minxy == 0) & (maxxy - 0.5 < r) & (maxxy+0.5 >= r)
    cond = (c1 & c2) | c3
    # a grid of magical values
    kernel = 0.5 * r**2 * (np.arcsin(m2/r) - np.arcsin(m1/r))
    kernel += 0.25 * r**2 * ( np.sin(2*np.arcsin(m2/r)) - np.sin(2*np.asin(m1/r)) )
    kernel -= (maxxy - 0.5) * (m2-m1)
    kernel += (m1 - minxy+0.5)
    kernel[~cond] = 0.0
    kernel[~c1] += 1.0 
    kernel[crad, crad] = np.minimum(np.pi * r**2, 0.5 * np.pi)
    if ((crad > 0) and (r > crad - 0.5) and (r**2 < (crad-0.5)**2 + 0.25)):
        m1 = np.sqrt(r**2 - (crad - 0.5)**2)
        m1n = m1/r
        sg0 = 2.0*(r**2 * (0.5 * np.arcsin(m1n) + 0.25 * np.sin(2 * np.arcsin(m1n))) - m1*(crad-0.5) )
        kernel[2*crad, crad] = sg0
        kernel[crad, 2*crad] = sg0
        kernel[crad, 0] = sg0
        kernel[0, crad] = sg0
        kernel[2*crad - 1, crad] -= sg0
        kernel[crad, 2*crad - 1] -= sg0
        kernel[crad, 1] -= sg0
        kernel[1, crad] -= sg0
    kernel[crad, crad] = np.minimum(kernel[crad, crad], 1);
    return kernel 


def distance_to_measure_cpu(measure, stroke_radius=1, dr=0.05):
    '''
    Compute the distance to a measure defined on a pixel grid.
    '''
    start = time.time() 
    rmax = 5*stroke_radius 
    # compute the mass of the heaviest ball and set the mass threshold to that
    last_mass = cv2.filter2D(src=measure, ddepth=-1, kernel=disk_kernel(stroke_radius))
    #m0 = np.min(last_mass[last_mass > np.max(measure)])
    m0 = np.max(last_mass)

    # compute the smallest possible radius that could contain mass m0
    rmin = np.minimum(np.sqrt(m0 / (np.pi * np.max(measure))), 0.5)

    # linearly sample radii
    #radii = np.linspace(rmin, rmax, int((rmax-rmin) / dr) )
    radii = np.geomspace(rmin, rmax, int((rmax-rmin) / dr) )


    # compute all convolution kernels 
    kernels = [disk_kernel(r) for r in radii]

    # wasserstein distance contribution of inner-most disk
    mass = cv2.filter2D(src=measure, ddepth=-1, kernel=kernels[0])
    D = np.zeros_like(measure)
    D = 0.5 * radii[0]**2 * mass

    # find all satisfied pixels
    satisfied = (mass >= m0)
    mass_prev = mass

    # add contribution from subsequent shells until reach mass m0
    for i, kernel in enumerate(kernels[1:], start=1):
        # compute the amount of mass contained in radius 'r' for all pixels
        mass = cv2.filter2D(src=measure, ddepth=-1, kernel=kernel)
        mass_diff = mass - mass_prev
        # set up a mask to only update pixels that haven't met the mass threshold
        update = (mass < m0) & ~satisfied 
        # add wasserstein distance contribution
        D[update] += 0.5 * (radii[i]**2 + radii[i-1]**2) * mass_diff[update]
        satisfied = (mass >= m0) 
        mass_prev = mass

    D /= np.sqrt(m0)
    D[~satisfied] = np.max(D)
    D = np.sqrt(D)

    end = time.time()
    duration = end - start

    print("----- DISTANCE COMPUTATION STATISTICS -----")
    print(f"Finished computing distance field in {duration} seconds")
    print(f"Heaviest ball of radius {stroke_radius}: mass = {m0}")
    print("Checked {} different radii from [{},{}]".format(len(radii), radii[0], radii[-1]))

    return D, m0

def distance_to_measure_gpu(measure, stroke_radius=1, dr=0.05):
   '''
   Compute the distance to a measure defined on a pixel grid.
   '''

   start = time.perf_counter() 
   rmax = 5*stroke_radius 
   # transfer input measure to GPU
   measure_gpu = cp.array(measure) 
   # compute the mass of the heaviest ball and set the mass threshold to that
   last_mass = cv2.filter2D(src=measure, ddepth=-1, kernel=disk_kernel(stroke_radius))
   m0 = np.max(last_mass)

   # compute the smallest possible radius that could contain mass m0
   rmin = np.minimum(np.sqrt(m0 / (np.pi * np.max(measure))), 0.5)
   
   # linearly sample radii
   radii = cp.linspace(rmin, rmax, int((rmax-rmin) / dr) )
   
   # initilize distance matrix on GPU
   D = cp.zeros_like(measure_gpu)

   # compute all convolution kernels 
   kernels = [cp.array(disk_kernel(r)) for r in radii.get()]

   # wasserstein distance contribution of inner-most disk
   mass = cps.convolve(measure_gpu, kernels[0], mode='constant')
   D = 0.5 * radii[0]**2 * mass

   # find all satisfied pixels
   satisfied = (mass >= m0)
   mass_prev = mass

   # add contribution from subsequent shells until reach mass m0
   for i, kernel in enumerate(kernels[1:], start=1):
       # compute the amount of mass contained in radius 'r' for all pixels
       mass = cps.convolve(measure_gpu, kernel, mode='constant')
       mass_diff = mass - mass_prev
       update = (mass < m0) & ~satisfied 
       # add wasserstein distance contribution
       #D[update] += 0.5 * (radii[i]**2 + radii[i-1]**2) * mass_diff[update]
       # gpu optimization (?)
       contrib = 0.5 * (radii[i]**2 + radii[i-1]**2) * mass_diff
       D = cp.where(update, D + contrib, D)
       satisfied |= (mass >= m0)

       #satisfied = (mass >= m0) #| satisfied
       mass_prev = mass

   D /= cp.sqrt(m0)
   D[~satisfied] = cp.max(D)
   D = cp.sqrt(D)


   elapsed_ms = (time.perf_counter() - start) * 1000
   logger.info(f"[DISTANCE] Computed distance function in {elapsed_ms} ms")
   logger.debug(f"[DISTANCE] Heaviest ball of radius {stroke_radius}: mass = {np.max(last_mass)}")
   logger.debug(f"[DISTANCE] Median mass of ball of radius {stroke_radius}: mass = {np.mean(last_mass[last_mass > 0])}")
   logger.debug(f"[DISTANCE] Checked {len(radii)} different radii from [{radii[0]},{radii[-1]}]")
   return D.get(), m0


def distance_to_measure_roi_sparse_cpu_numba(measure, stroke_radius=1, dr=0.05):
    """
    Numba-accelerated distance-to-measure function using sparse ROI and local convolutions.
    Avoids global convolution; operates only at relevant ROI pixels.
    """
    h, w = measure.shape
    rmax = min(3 * stroke_radius, 50)

    # Step 1: Compute m0 (heaviest ball of radius stroke_radius)
    last_mass = cv2.filter2D(measure, -1, disk_kernel(stroke_radius))
    m0 = 0.5* (np.max(last_mass) - np.min(last_mass)) 


    # Plot histogram
    '''
    plt.figure()
    plt.hist(last_mass[last_mass>1e-5].ravel(), bins=50, color='skyblue', edgecolor='black')
    plt.title("Histogram of Mass Values")
    plt.xlabel("Mass")
    plt.ylabel("Frequency")
    plt.show()
    '''

    # Step 2: Estimate rmin
    rmin = min(np.sqrt(m0 / (np.pi * np.max(measure))), 0.5)
    radii = np.linspace(rmin, rmax, int((rmax - rmin) / dr))
    radii_sq = radii ** 2

    # Step 3: Compute ROI mask
    kernel_rmax = disk_kernel(rmax)
    conv_rmax = cv2.filter2D(measure, -1, kernel_rmax)
    roi_mask = (conv_rmax >= m0)
    roi_coords = np.argwhere(roi_mask).astype(np.int32)  # shape (N, 2)

    # Allocate output arrays
    D = np.zeros((h, w), dtype=np.float64)
    satisfied = np.zeros((h, w), dtype=np.bool_)
    mass_prev = np.zeros((h, w), dtype=np.float64)

    # Step 4: Process each radius
    for i in range(len(radii)):
        r = radii[i]
        r2 = radii_sq[i]
        r2_prev = radii_sq[i - 1] if i > 0 else -1.0
        kernel = disk_kernel(r)
        D, satisfied, mass_prev = _process_radius_numba(
            measure, D, satisfied, mass_prev, roi_coords, kernel, m0, r2, r2_prev
        )

        if np.all(satisfied[roi_mask]):
            break

    # Final normalization
    D[~satisfied] = np.max(D)
    D /= np.sqrt(m0)
    D = np.sqrt(D)

    '''
    plt.figure()
    plt.hist(D.ravel(), bins=50, color='red', edgecolor='black')
    plt.title("Histogram of Distance Values")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.show()
    '''
    return D, m0


@njit(parallel=True)
def _process_radius_numba(measure, D, satisfied, mass_prev, roi_coords, kernel, m0, r2, r2_prev):
    h, w = measure.shape
    kH, kW = kernel.shape
    pad_y = kH // 2
    pad_x = kW // 2

    for i in prange(roi_coords.shape[0]):
        y, x = roi_coords[i]
        if satisfied[y, x]:
            continue

        y0 = y - pad_y
        y1 = y + pad_y + 1
        x0 = x - pad_x
        x1 = x + pad_x + 1

        if y0 < 0 or y1 > h or x0 < 0 or x1 > w:
            continue

        local_mass = 0.0
        for dy in range(kH):
            for dx in range(kW):
                yy = y0 + dy
                xx = x0 + dx
                local_mass += measure[yy, xx] * kernel[dy, dx]

        mass_diff = local_mass - mass_prev[y, x]

        if r2_prev < 0:
            D[y, x] = 0.5 * r2 * local_mass
        elif local_mass < m0:
            D[y, x] += 0.5 * (r2 + r2_prev) * mass_diff

        if local_mass >= m0:
            satisfied[y, x] = True

        mass_prev[y, x] = local_mass

    return D, satisfied, mass_prev

def distance_to_measure_gpu_sparse(measure, stroke_radius=1, dr=0.05):
    """
    Compute the distance-to-measure function on the GPU using CuPy,
    while skipping pixels that cannot possibly reach the mass threshold m0.
    """
    start = time.perf_counter()

    # Transfer measure to GPU
    measure_gpu = cp.array(measure, dtype=cp.float32)

    # Clamp rmax
    rmax = min(10 * stroke_radius, 50)

    # Compute m0 from CPU (cv2 is still used here)
    from cv2 import filter2D
    m0 = np.max(filter2D(measure.astype(np.float32), -1, disk_kernel(stroke_radius)))

    # Estimate rmin
    rmin = min(np.sqrt(m0 / (np.pi * np.max(measure))), 0.5)
    radii = cp.array(np.geomspace(rmin, rmax, int((rmax - rmin) / dr)))
    radii_sq = radii**2

    # Initialize arrays
    D = cp.zeros_like(measure_gpu)
    satisfied = cp.zeros_like(measure_gpu, dtype=cp.bool_)
    mass_prev = cp.zeros_like(measure_gpu)

    # Compute roi_mask using disk_kernel(rmax)
    kernel_rmax = cp.array(disk_kernel(rmax), dtype=cp.float32)
    mass_rmax = cps.convolve(measure_gpu, kernel_rmax, mode="constant")
    roi_mask = (mass_rmax >= m0)

    # First radius
    kernel0 = cp.array(disk_kernel(radii[0].item()), dtype=cp.float32)
    mass = cps.convolve(measure_gpu, kernel0, mode="constant")
    D = 0.5 * radii_sq[0] * mass
    satisfied = mass >= m0
    mass_prev = mass

    # Iterate through radii
    for i in range(1, len(radii)):
        kernel = cp.array(disk_kernel(radii[i].item()), dtype=cp.float32)
        mass = cps.convolve(measure_gpu, kernel, mode="constant")
        mass_diff = mass - mass_prev

        contrib = 0.5 * (radii_sq[i] + radii_sq[i - 1]) * mass_diff
        #update_mask = (~satisfied) & roi_mask & (mass < m0)
        update_mask = (~satisfied) & (mass < m0)
        D[update_mask] += contrib[update_mask]

        satisfied = satisfied | (mass >= m0)
        mass_prev = mass

    # Finalize
    D[~satisfied] = cp.max(D)
    D = D / cp.sqrt(m0)
    D = cp.sqrt(D)

    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"[GPU-SPARSE] Distance computation completed in {elapsed:.2f} ms")
    logger.debug(f"[GPU-SPARSE] Checked {len(radii)} radii from {rmin:.2f} to {rmax:.2f}")
    logger.debug(f"[GPU-SPARSE] m0 = {m0:.6f}, max D = {cp.max(D).item():.4f}")

    return D.get(), m0

# if __name__ == "__main__":
#
