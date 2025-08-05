# This Python file uses the following encoding: utf-8
import platform
import time
import logging 
import numpy as np
if platform.system() != 'Darwin':
    import cupy as cp
    import cupyx.scipy.ndimage as cps
import cv2


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


def distance_to_measure_cpu(measure, m0, dr, rmax=None):
    '''
    Compute the distance to a measure defined on a pixel grid.
    '''
    # compute the smallest possible radius that could contain mass m0
    rmin = np.minimum(np.sqrt(m0 / (np.pi * np.max(measure))), 0.5)

    # wasserstein distance contribution of inner-most disk
    mass = cv2.filter2D(src=measure, ddepth=-1, kernel=disk_kernel(rmin))
    D = np.zeros_like(measure)
    D = 0.5 * rmin**2 * mass

    # find all satisfied pixels
    satisfied = (mass >= m0)
    mass_prev = mass

    # add contribution from subsequent shells until reach mass m0
    radius = rmin + dr
    while not np.all(satisfied):
        # stop if we hit the radius limit
        if rmax is not None and radius > rmax:
            logger.warning(f"Reached max radius ({rmax}). Exiting early.")
            break

        kernel = disk_kernel(radius)
        # compute the amount of mass contained in radius 'r' for all pixels
        mass = cv2.filter2D(src=measure, ddepth=-1, kernel=kernel)
        mass_diff = mass - mass_prev
        # set up a mask to only update pixels that haven't met the mass threshold
        update = (mass < m0) & ~satisfied 
        # add wasserstein distance contribution
        D[update] += 0.5 * (radius**2 + (radius-dr)**2) * mass_diff[update]
        satisfied = (mass >= m0) 
        mass_prev = mass
        radius += dr
        # log progress using info level
        percent = 100 * np.count_nonzero(satisfied) / satisfied.size 
        logger.info(f"⏳ Radius: {radius:.2f}, Progress: {percent:.2f}%")

    D /= np.sqrt(m0)
    # if we never reach m0, just apply ceiling
    D[~satisfied] = np.max(D)
    D = np.sqrt(D)

    return D

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
       D[update] += 0.5 * (radii[i]**2 + radii[i-1]**2) * mass_diff[update]
       satisfied = (mass >= m0) #| satisfied
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


# if __name__ == "__main__":
#
