import numpy as np

from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit, minimize

from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft

# from py4DSTEM.process.utils import convert_ellipse_params,convert_ellipse_params_r

'''def process_frame_for_shifts(data,win,use_grad=False,use_double=False):
    if use_grad:     
        if use_double:
            Gx,Gy,_,_ = compute_gradient_maps(data)
            Gx *= win
            Gy *= win
            im_proc = [Gx,Gy]
        else:
            _,_,Gmag,_ = compute_gradient_maps(data)
            Gmag *= win
            im_proc = Gmag
    else:
        im_proc = data
    return im_proc

def compute_shifts(im_proc,im_proc_ref,use_double=True):
	if use_double:
		shifts,_,_ = phase_cross_correlation_double(im_proc_ref[0],im_proc[0],im_proc_ref[1],im_proc[1],
		normalization=None,upsample_factor=10)
	else:
		shifts,_,_ = phase_cross_correlation(im_proc_ref,im_proc,normalization=None,upsample_factor=10)
		
	return shifts

def compute_shifts_array(data_array,im_proc_ref,store_im_proc=True,use_double=True):
    n_frames = data_array.shape[0]
    shift_arr = np.zeros((n_frames,2))
    
    if store_im_proc:
        im_proc_list = []
    else: 
        im_proc_list= None
    
    for i_frame in range(n_frames):
        im_proc = process_frame_for_shifts(data_array[i_frame],use_double=use_double)
        shift_arr[i_frame] = compute_shifts(im_proc,im_proc_ref,use_double=use_double)
        if store_im_proc:
            im_proc_list.append(im_proc)
        
    return shift_arr,im_proc_list'''
	
# Processing

def compute_CoM(I,X,Y):
    CoM_x = np.mean(I*X)/np.mean(I)
    CoM_y = np.mean(I*Y)/np.mean(I)
    return (CoM_x,CoM_y)

def compute_shift_ecc(imdata,Qx,Qy,K_ell,mask=None,filt_sigma=5,thresh=0.25):
# ECC
	Gx,Gy,Gmag,_ = compute_gradient_maps(imdata,thresh=thresh,filt_sigma=filt_sigma)
	if not(mask==None):
		Gmag *= mask
	shifts,_,_ = phase_cross_correlation(K_ell,Gmag,normalization=None,upsample_factor=100)
	shifts = np.array((-shifts[1],-shifts[0]))
	return shifts,Gmag
	
def compute_shift_array_ecc(data_array,Qx,Qy,K_ell,mask=None,store_im_proc=True,thresh=0.25,filt_sigma=5):
    n_frames = data_array.shape[0]
    shift_arr = np.zeros((n_frames,2))
    
    if store_im_proc:
        im_proc_list = []
    else: 
        im_proc_list= None
    
    for i_frame in range(n_frames):        

        im_proc = data_array[i_frame].astype('double')
        shift_arr[i_frame,:],Gmag = compute_shift_ecc(im_proc,Qx,Qy,K_ell,
                                                                 mask=mask,thresh=thresh,filt_sigma=filt_sigma)         
        if store_im_proc:
            im_proc_list.append(Gmag)
        
    return shift_arr,im_proc_list
	
# Phase computation
def convert_defl_to_phase_grad(lamb_elec_nm,defl):
    return defl*(2*np.pi)/lamb_elec_nm

def rotate_coord_list(u,rot_theta):
    u_rot = np.zeros(u.shape)
    u_rot[:,0] = u[:,0]*np.cos(rot_theta) - u[:,1]*np.sin(rot_theta)
    u_rot[:,1] = u[:,0]*np.sin(rot_theta) + u[:,1]*np.cos(rot_theta)
    return u_rot


# Edge finding

def compute_gradient_maps(data,thresh=0.25,filt_sigma=3):
	Kx = np.array([[1,2,1],
		[0,0,0],
		[-1,-2,-1]])/4
	Ky = np.array([[1,0,-1],
		[2,0,-2],
		[1,0,-1]])/4

	data = gaussian_filter(data,filt_sigma)
	Gx = convolve2d(data,Kx,mode='same',boundary='symm')
	Gy = convolve2d(data,Ky,mode='same',boundary='symm')
	Gmag = np.sqrt(Gx**2+Gy**2)
	above_thresh = (Gmag/np.max(Gmag)) > thresh
	Gx *= above_thresh
	Gy *= above_thresh
	Gmag *= above_thresh
    
	return Gx,Gy,Gmag,above_thresh

def generate_annular_mask(shape,center,r_inner,r_outer):
    xm,ym = np.meshgrid(np.arange(shape[0]),np.arange(shape[1]))
    r2m = (xm-center[1])**2 + (ym-center[0])**2
    return (r2m>r_inner**2) * (r2m<r_outer**2)
	
def generate_annular_mask_elliptical(shape,center,a_inner,a_outer,ab_ratio,theta):
	xm,ym = np.meshgrid(np.arange(shape[0]),np.arange(shape[1]))
	r2m = (xm-center[1])**2 + (ym-center[0])**2
	thetam = np.arctan2(ym-center[0],xm-center[1])
	r2_inner_ell = (a_inner*np.cos(thetam-theta))**2 + (ab_ratio*a_inner*np.sin(thetam-theta))**2
	r2_outer_ell = (a_outer*np.cos(thetam-theta))**2 + (ab_ratio*a_outer*np.sin(thetam-theta))**2
	in_mask = (r2m>r2_inner_ell) * (r2m<r2_outer_ell)
	return in_mask


# Ellipse fitting functions borrowed from py4DSTEM version 0.13.7

def convert_ellipse_params(A, B, C):
    """
    Converts ellipse parameters from canonical form (A,B,C) into semi-axis lengths and
    tilt (a,b,theta).
    See module docstring for more info.

    Args:
        A,B,C (floats): parameters of an ellipse in the form:
                             Ax^2 + Bxy + Cy^2 = 1

    Returns:
        (3-tuple): A 3-tuple consisting of:

        * **a**: (float) the semimajor axis length
        * **b**: (float) the semiminor axis length
        * **theta**: (float) the tilt of the ellipse semimajor axis with respect to
          the x-axis, in radians
    """
    val = np.sqrt((A - C) ** 2 + B**2)
    b4a = B**2 - 4 * A * C
    # Get theta
    if B == 0:
        if A < C:
            theta = 0
        else:
            theta = np.pi / 2.0
    else:
        theta = np.arctan2((C - A - val), B)
    # Get a,b
    a = -np.sqrt(-2 * b4a * (A + C + val)) / b4a
    b = -np.sqrt(-2 * b4a * (A + C - val)) / b4a
    a, b = max(a, b), min(a, b)
    return a, b, theta


def convert_ellipse_params_r(a, b, theta):
    """
    Converts from ellipse parameters (a,b,theta) to (A,B,C).
    See module docstring for more info.

    Args:
        a,b,theta (floats): parameters of an ellipse, where `a`/`b` are the
            semimajor/semiminor axis lengths, and theta is the tilt of the semimajor axis
            with respect to the x-axis, in radians.

    Returns:
        (3-tuple): A 3-tuple consisting of (A,B,C), the ellipse parameters in
            canonical form.
    """
    sin2, cos2 = np.sin(theta) ** 2, np.cos(theta) ** 2
    a2, b2 = a**2, b**2
    A = sin2 / b2 + cos2 / a2
    C = cos2 / b2 + sin2 / a2
    B = 2 * (b2 - a2) * np.sin(theta) * np.cos(theta) / (a2 * b2)
    return A, B, C

def gaussian_ring(p, x, y):
    """
    Return the value of the double-sided gaussian function at point (x,y) given
    parameters p, described in detail in the fit_ellipse_amorphous_ring docstring.
    """
    # Unpack parameters
    I, sigma, c_bkgd, x0, y0, A, B, C = p
    a,b,theta = convert_ellipse_params(A,B,C)
    R = np.mean((a,b))
    R2 = R**2
    A,B,C = A*R2,B*R2,C*R2
    r2 = A*(x - x0)**2 + B*(x - x0)*(y - y0) + C*(y - y0)**2
    r = np.sqrt(r2) - R

    return I * np.exp(-r ** 2 / (2 * sigma ** 2)) + c_bkgd


	
# Cross correlation

def compute_cross_correlation(src_freq,target_freq,normalization="phase"):
    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    if normalization == "phase":
        eps = np.finfo(image_product.real.dtype).eps
        image_product /= np.maximum(np.abs(image_product), 100 * eps)
    elif normalization is not None:
        raise ValueError("normalization must be either phase or None")
    cross_correlation = np.fft.ifftn(image_product)
    return cross_correlation,image_product
	
def butterworth(q,q0,n):
    return 1/np.sqrt(1+(q/q0)**(2*n))

def elliptical_butterworth_bandpass(data,x,y,p,dr_filt,n):
    # Unpack parameters
    x0, y0, A, B, C = p
    a,b,theta = convert_ellipse_params(A,B,C)
    R = np.mean((a,b))
    R2 = R**2
    A,B,C = A*R2,B*R2,C*R2
    r2 = A*(x - x0)**2 + B*(x - x0)*(y - y0) + C*(y - y0)**2
    r = np.sqrt(r2) - R
    
    return butterworth(r,dr_filt,n)
	
# NOTE: FIX INCONSISTENT CONVERSION OF IMAGE PRODUCT TO CROSS CORRELATION BETWEEN INITIAL AND UPSAMPLED

def phase_cross_correlation_double(reference_image_1, moving_image_1,
                                   reference_image_2, moving_image_2,
                                   *,
                            upsample_factor=1, space="real",
                            disambiguate=False,
                            return_error=True, reference_mask=None,
                            moving_mask=None, overlap_ratio=0.3,
                            normalization="phase"):

#     # images must be the same shape
#     if reference_image.shape != moving_image.shape:
#         raise ValueError("images must be same shape")

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq_1 = reference_image_1
        target_freq_1 = moving_image_1
        src_freq_2 = reference_image_2
        target_freq_2 = moving_image_2
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        src_freq_1 = np.fft.fftn(reference_image_1)
        target_freq_1 = np.fft.fftn(moving_image_1)
        src_freq_2 = np.fft.fftn(reference_image_2)
        target_freq_2 = np.fft.fftn(moving_image_2)
    else:
        raise ValueError('space argument must be "real" of "fourier"')

    
    cross_correlation_1,image_product_1 = compute_cross_correlation(src_freq_1,target_freq_1,normalization=normalization)
    cross_correlation_2,image_product_2 = compute_cross_correlation(src_freq_2,target_freq_2,normalization=normalization)
    cross_correlation_double = np.abs(cross_correlation_1) + np.abs(cross_correlation_2)
    image_product_double = image_product_1+image_product_2
    
    # Locate maximum
    shape = src_freq_1.shape
    maxima = np.unravel_index(np.argmax(cross_correlation_double),
                              cross_correlation_double.shape)
    midpoint = np.array([np.fix(axis_size / 2) for axis_size in shape])

    float_dtype = image_product_1.real.dtype

    shift = np.stack(maxima).astype(float_dtype, copy=False)
    shift[shift > midpoint] -= np.array(shape)[shift > midpoint]

#     if upsample_factor == 1:
#         if return_error:
#             src_amp = np.sum(np.real(src_freq * src_freq.conj()))
#             src_amp /= src_freq.size
#             target_amp = np.sum(np.real(target_freq * target_freq.conj()))
#             target_amp /= target_freq.size
#             CCmax = cross_correlation[maxima]
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    if upsample_factor > 1:
        # Initial shift estimate in upsampled grid
        upsample_factor = np.array(upsample_factor, dtype=float_dtype)
        shift = np.round(shift * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shift*upsample_factor
        cross_correlation = _upsampled_dft(image_product_double.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj()
        # Locate maximum and map back to original pixel grid
        maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                                  cross_correlation.shape)
        CCmax = cross_correlation[maxima]

        maxima = np.stack(maxima).astype(float_dtype, copy=False)
        maxima -= dftshift

        shift += maxima / upsample_factor

#         if return_error:
#             src_amp = np.sum(np.real(src_freq * src_freq.conj()))
#             target_amp = np.sum(np.real(target_freq * target_freq.conj()))

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq_1.ndim):
        if shape[dim] == 1:
            shift[dim] = 0

#     if disambiguate:
#         if space.lower() != 'real':
#             reference_image = ifftn(reference_image)
#             moving_image = ifftn(moving_image)
#         shift = _disambiguate_shift(reference_image, moving_image, shift)

#     if return_error:
#         # Redirect user to masked_phase_cross_correlation if NaNs are observed
#         if np.isnan(CCmax) or np.isnan(src_amp) or np.isnan(target_amp):
#             raise ValueError(
#                 "NaN values found, please remove NaNs from your "
#                 "input data or use the `reference_mask`/`moving_mask` "
#                 "keywords, eg: "
#                 "phase_cross_correlation(reference_image, moving_image, "
#                 "reference_mask=~np.isnan(reference_image), "
#                 "moving_mask=~np.isnan(moving_image))")

#         return shift, _compute_error(CCmax, src_amp, target_amp),\
#             _compute_phasediff(CCmax)
#     else:
#         warn_return_error()
    return shift,cross_correlation_double,image_product_double
	
# Phase reconstruction

def get_phase_from_CoM(CoMx, CoMy, theta, flip, regLowPass=0.5, regHighPass=100,
                        paddingfactor=2, stepsize=1, n_iter=10, phase_init=None):
    """
    Calculate the phase of the sample transmittance from the diffraction centers of mass.
    A bare bones description of the approach taken here is below - for detailed
    discussion of the relevant theory, see, e.g.::
        Ishizuka et al, Microscopy (2017) 397-405
        Close et al, Ultramicroscopy 159 (2015) 124-137
        Wadell and Chapman, Optik 54 (1979) No. 2, 83-96
    The idea here is that the deflection of the center of mass of the electron beam in
    the diffraction plane scales linearly with the gradient of the phase of the sample
    transmittance. When this correspondence holds, it is therefore possible to invert the
    differential equation and extract the phase itself.* The primary assumption made is
    that the sample is well described as a pure phase object (i.e. the real part of the
    transmittance is 1). The inversion is performed in this algorithm in Fourier space,
    i.e. using the Fourier transform property that derivatives in real space are turned
    into multiplication in Fourier space.
    *Note: because in DPC a differential equation is being inverted - i.e. the
    fundamental theorem of calculus is invoked - one might be tempted to call this
    "integrated differential phase contrast".  Strictly speaking, this term is redundant
    - performing an integration is simply how DPC works.  Anyone who tells you otherwise
    is selling something.
    Args:
        CoMx (2D array): the diffraction space centers of mass x coordinates
        CoMy (2D array): the diffraction space centers of mass y coordinates
        theta (float): the rotational offset between real and diffraction space
            coordinates
        flip (bool): whether or not the real and diffraction space coords contain a
                        relative flip
        regLowPass (float): low pass regularization term for the Fourier integration
            operators
        regHighPass (float): high pass regularization term for the Fourier integration
            operators
        paddingfactor (int): padding to add to the CoM arrays for boundry condition
            handling. 1 corresponds to no padding, 2 to doubling the array size, etc.
        stepsize (float): the stepsize in the iteration step which updates the phase
        n_iter (int): the number of iterations
        phase_init (2D array): initial guess for the phase
    Returns:
        (2-tuple) A 2-tuple containing:
            * **phase**: *(2D array)* the phase of the sample transmittance, in radians
            * **error**: *(1D array)* the error - RMSD of the phase gradients compared
              to the CoM - at each iteration step
    """
    assert isinstance(flip,(bool,np.bool_))
    assert isinstance(paddingfactor,(int,np.integer))
    assert isinstance(n_iter,(int,np.integer))

    # Coordinates
    R_Nx,R_Ny = CoMx.shape
    R_Nx_padded,R_Ny_padded = R_Nx*paddingfactor,R_Ny*paddingfactor

    qx = np.fft.fftfreq(R_Nx_padded)
    qy = np.fft.rfftfreq(R_Ny_padded)
    qr2 = qx[:,None]**2 + qy[None,:]**2

    # Inverse operators
    denominator = qr2 + regHighPass + qr2**2*regLowPass
    _ = np.seterr(divide='ignore')
    denominator = 1./denominator
    denominator[0,0] = 0
    _ = np.seterr(divide='warn')
    f = 1j * -0.25*stepsize
    qxOperator = f*qx[:,None]*denominator
    qyOperator = f*qy[None,:]*denominator

    # Perform rotation and flipping
    if not flip:
        CoMx_rot = CoMx*np.cos(theta) - CoMy*np.sin(theta)
        CoMy_rot = CoMx*np.sin(theta) + CoMy*np.cos(theta)
    if flip:
        CoMx_rot = CoMx*np.cos(theta) + CoMy*np.sin(theta)
        CoMy_rot = CoMx*np.sin(theta) - CoMy*np.cos(theta)

    # Initializations
    phase = np.zeros((R_Nx_padded,R_Ny_padded))
    update = np.zeros((R_Nx_padded,R_Ny_padded))
    dx = np.zeros((R_Nx_padded,R_Ny_padded))
    dy = np.zeros((R_Nx_padded,R_Ny_padded))
    error = np.zeros(n_iter)
    mask = np.zeros((R_Nx_padded,R_Ny_padded),dtype=bool)
    mask[:R_Nx,:R_Ny] = True
    maskInv = mask==False
    if phase_init is not None:
        phase[:R_Nx,:R_Ny] = phase_init

    # Iterative reconstruction
    for i in range(n_iter):

        # Update gradient estimates using measured CoM values
        dx[mask] -= CoMx_rot.ravel()
        dy[mask] -= CoMy_rot.ravel()
        dx[maskInv] = 0
        dy[maskInv] = 0
        
        dx_step = dx
        dy_step = dy

        # Calculate reconstruction update
        update = -np.fft.irfft2( np.fft.rfft2(dx)*qxOperator + np.fft.rfft2(dy)*qyOperator)

        # Apply update
        phase += stepsize*update

        # Measure current phase gradients
        dx = (np.roll(phase,(-1,0),axis=(0,1)) - np.roll(phase,(1,0),axis=(0,1))) / 2.
        dy = (np.roll(phase,(0,-1),axis=(0,1)) - np.roll(phase,(0,1),axis=(0,1))) / 2.

        # Estimate error from cost function, RMS deviation of gradients
        xDiff = dx[mask] - CoMx_rot.ravel()
        yDiff = dy[mask] - CoMy_rot.ravel()
        error[i] = np.sqrt(np.mean((xDiff-np.mean(xDiff))**2 + (yDiff-np.mean(yDiff))**2))

        # Halve step size if error is increasing
        if i>0:
            if error[i] > error[i-1]:
                stepsize /= 2

    phase = phase[:R_Nx,:R_Ny]

    return phase, error, dx_step, dy_step