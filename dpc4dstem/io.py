import numpy as np
import ncempy.io as nio

def load_single_mrc_frame(filepath_input,inds,scan_shape=None):
    # inds = linear index if scalar, multi-coordinate if a shape is given
    # scan_shape = (R_Ny,R_Nx)
    # Requires import ncempy.io as nio
    
    if scan_shape == None:
        ind = inds
    else:
        ind = np.ravel_multi_index(inds,scan_shape)

    with nio.mrc.fileMRC(filepath_input) as f:
        image_data = f.getSlice(ind)
    return image_data

def load_selected_frames_mrc(filepath_input,inds_select):
    arr_select = []
    N = len(inds_select)
    with nio.mrc.fileMRC(filepath_input) as f:
        for i_slice,i_plot in zip(inds_select,range(N)):
            arr_select.append(f.getSlice(i_slice))
    return arr_select
	
def load_scan_line(filepath_input,data_shape,ind_line,scan_dir='row'):
	R_Ny,R_Nx,Q_Ny,Q_Nx = data_shape
    
	if scan_dir == 'row':
		n_frames = R_Nx
	elif scan_dir == 'col':
		n_frames = R_Ny

	data_array = np.zeros((n_frames,Q_Ny,Q_Nx))
    
	with nio.mrc.fileMRC(filepath_input) as f1:
		for i_frame in range(n_frames):
			if scan_dir == 'row':
				ind_row = ind_line
				ind_col = i_frame
			elif scan_dir == 'col':
				ind_row = i_frame
				ind_col = ind_line
                
			data_array[i_frame,:,:] = f1.getSlice(ind_row*R_Nx + ind_col)
	return data_array