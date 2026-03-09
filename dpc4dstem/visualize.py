from dpc4dstem.process import convert_ellipse_params
import numpy as np

from matplotlib import animation
import matplotlib.pyplot as plt

def draw_ellipse(p,N=100):
    x0,y0,A,B,C = p[3:]
    a,b,theta = convert_ellipse_params(A,B,C)
    t = np.linspace(0,2*np.pi,N)
    x = x0 + a*np.cos(theta)*np.cos(t) - b*np.sin(theta)*np.sin(t)
    y = y0 + a*np.sin(theta)*np.cos(t) + b*np.cos(theta)*np.sin(t)
    return x,y

def draw_shifted_ellipse(shifts,pfit,Q_Nx,Q_Ny):
    p_ell = np.zeros(pfit.shape)
    p_ell[:] = pfit[:]
    p_ell[3] += shifts[1]/Q_Nx
    p_ell[4] += shifts[0]/Q_Ny
    x_ell,y_ell = draw_ellipse(p_ell)
    return x_ell,y_ell
	
# plots

def plot_diff_maps(diff_data,titles_diff, figsize_x, figsize_y,clim=None):
    plt.figure(figsize=(figsize_x, figsize_y))
    for i_pane in range(2):
        plt.subplot(2,1,i_pane+1)
        plt.imshow(diff_data[i_pane],cmap='seismic')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar(aspect=8)
        plt.title(titles_diff[i_pane])
        if clim==None:
            plt.clim(np.array([-1,1])*np.max(np.abs(diff_data[i_pane])))
        else:
            plt.clim(clim)
        plt.tight_layout()
	
# movie

def preprocess_frame_movie(imdata,norm=True,gamma=1,k_thresh=0):
    if norm:
            imdata /= np.mean(imdata)
    imdata[imdata<k_thresh] = 0
    imdata = imdata**gamma
    return imdata

def setup_scan_shifts_movie(data_array,im_proc_array,shift_arr,p_ell=None,index=0,ind_test=0,norm=True,gamma=1,show_ellipse=True):
    
    n_frames = data_array.shape[0]
   
    Q_Ny = data_array.shape[1]
    Q_Nx = data_array.shape[2]
	
    # Prepare initializing images
    imdata_list = [preprocess_frame_movie(data_array[ind_test],norm=norm,gamma=gamma),
                   im_proc_array[ind_test]/np.mean(im_proc_array[ind_test])]
    
    # Prepare the axes
    fig = plt.figure()
    
    axs = []
    ims = []
    ells = []
    textlabels = []
    
    for i_ax in range(2):
        ax = fig.add_subplot(1,2,i_ax+1)
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        im = ax.imshow(imdata_list[i_ax])
        ims.append(im)
        
        if i_ax == 0:
            ax.set_title("Raw disk", fontsize=12)
        else:
            ax.set_title("Edge filtered", fontsize=12)

        if show_ellipse:
            shift_ell = (shift_arr[ind_test,0],shift_arr[ind_test,1])
            x_ell,y_ell = draw_ellipse(p_ell,N=100)
            ell = ax.plot(x_ell*Q_Nx,y_ell*Q_Ny,'r--',lw=2.0)
            ells.append(ell)
            textlabel = ax.text(10, Q_Ny-10, f'({index},{ind_test})', fontsize=9, c='white')
            textlabels.append(textlabel)
        
        axs.append(ax)
    
    fig.set_size_inches([9,4.5]) 

    plt.tight_layout()
    
    return fig,axs,ims,ells,textlabels
    
    
def generate_scan_shifts_movie(data_array,im_proc_array,shift_arr,fname,index=0,fps=25,norm=True,gamma=1,dpi=400,show_ellipse=True,p_ell=None,):
    
    fig,axs,ims,ells,textlabels = setup_scan_shifts_movie(data_array,im_proc_array,shift_arr,
                                           p_ell=p_ell,norm=norm,gamma=gamma,show_ellipse=show_ellipse)
    Q_Ny = data_array.shape[1]
    Q_Nx = data_array.shape[2]

    # Update function
    def update_img(n):
        
        imdata_list = [preprocess_frame_movie(data_array[n],norm=norm,gamma=gamma),
                       (im_proc_array[n]/np.mean(im_proc_array[n]))]
        
        for i_ax in range(2):
            ims[i_ax].set_data(imdata_list[i_ax])
        
            if show_ellipse:
                shift_ell = (shift_arr[n,1],shift_arr[n,0])
                x_ell,y_ell = draw_shifted_ellipse(shift_ell,p_ell,Q_Nx,Q_Ny)
                ells[i_ax][0].set_xdata(x_ell*Q_Nx)
                ells[i_ax][0].set_ydata(y_ell*Q_Ny)
                textlabels[i_ax].set_text(f'({index},{n})')
        
        return ims,ells
    
    n_frames = data_array.shape[0]
   
    ani = animation.FuncAnimation(fig,update_img,n_frames,interval=1)
    writer = animation.writers['ffmpeg'](fps=fps)

    ani.save(fname,writer=writer,dpi=dpi)
    return ani