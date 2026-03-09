# Movie making
from dpc4dstem.visualize import draw_shifted_ellipse
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

def preprocess_frame_movie(imdata,norm=True,gamma=1,k_thresh=0):
    if norm:
            imdata /= np.mean(imdata)
    imdata[imdata<k_thresh] = 0
    imdata = imdata**gamma
    return imdata


def setup_scan_shifts_movie(data_array,im_proc_array,shift_arr,norm=True,gamma=1,show_ellipse=True):
    
    # Prepare initializing images
    imdata_list = [preprocess_frame_movie(data_array[0],norm=norm,gamma=gamma),
                   im_proc_array[0]]
    
    # Prepare the axes
    fig = plt.figure()
    
    axs = []
    ims = []
    ells = []
    
    for i_ax in range(2):
        ax = fig.add_subplot(1,2,i_ax+1)
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        im = ax.imshow(imdata_list[i_ax],extent=[0,1,1,0])
        ims.append(im)
        
        if show_ellipse:
            shifts = shift_arr[0]
            x_ell,y_ell = draw_shifted_ellipse(shifts,pfit,Q_Nx,Q_Ny)
            ell = ax.plot(x_ell,y_ell,'r--',lw=1.0)
            ells.append(ell)
        
        axs.append(ax)
    
    fig.set_size_inches([9,4.5]) 

    plt.tight_layout()
    
    return fig,axs,ims,ells
    
    
def generate_scan_shifts_movie(data_array,im_proc_array,shift_arr,fname,fps=25,norm=True,gamma=1,dpi=400,show_ellipse=True):
    
    fig,axs,ims,ells = setup_scan_shifts_movie(data_array,im_proc_array,shift_arr,
                                           norm=norm,gamma=gamma,show_ellipse=show_ellipse)

    # Update function
    def update_img(n):
        
        imdata_list = [preprocess_frame_movie(data_array[n],norm=norm,gamma=gamma),
                       im_proc_array[n]]
        
        for i_ax in range(2):
            ims[i_ax].set_data(imdata_list[i_ax])
        
            if show_ellipse:
                shifts = shift_arr[n]
                x_ell,y_ell = draw_shifted_ellipse(shifts,pfit,Q_Nx,Q_Ny)
                ells[i_ax][0].set_xdata(x_ell)
                ells[i_ax][0].set_ydata(y_ell)
        
        return ims,ells
    
    n_frames = data_array.shape[0]
   
    ani = animation.FuncAnimation(fig,update_img,n_frames,interval=1)
    writer = animation.writers['ffmpeg'](fps=fps)

    ani.save(fname,writer=writer,dpi=dpi)
    return ani