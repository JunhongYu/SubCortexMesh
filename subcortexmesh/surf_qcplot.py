#####################################################################################
###########Plotting native surfaces on top of their volume ##########################

import os
import vtk
import pyvista as pv
import numpy as np
import matplotlib.pyplot as mplpyplot
from matplotlib.widgets import Button, Slider
from typing import Optional, Union
from pathlib import Path

def surf_qcplot(
    volpath: Union[str, Path],
    surfdir: Union[str, Path],
    vol_color_map: Optional[str] = "gray",
    outline_color_map: Optional[str] = "tab20"
    ):
    """Plotting subject surface boundaries on top of matching volume
    
    This function function will plot all surfaces generated with vol2surf() or fslfirst_getsurf() for a given subject (normally stored in "sub_surfaces/sub-[id]/") and plot their boundaries on top of a selected .nii volume. 
    
    In the case of fsaverage-based surfaces: because the surfaces are based on a volume rigidly coregistered to fsaverage, the surfaces will match a volume generated with aseg_getvol(), i.e.  "sub_volumes/sub-[id]/ants_coreg/T1_fsaverage_rigid_coreg.nii.gz" (or any volume likewise coregistered). Note that as per aseg_getvol(), individual regions-of-interest (ROIs) will have been inflated and smoothed by default to minimise graphical artefacts, which will naturally be reflected in the plot: regions will appear with slightly wider boundaries than their original anatomy. Another logical result is that ROI boundaries will appear overlapping, but since the surfaces are processed by SubCortexMesh entirely separately, this has no effect on the metrics values (ROIs overlapping here cannot be mixed up in their calculation by mesh_metrics() or merge_all()).
    
    Parameters
    ----------
    volpath : str, Path
        The path to the volume to be plotted in the background (e.g., in the case of fsaverage, the T1 volume coregistered to fsaverage as part of aseg_getvol() in sub_volume/sub-[id]/ants_coreg/).
    surfdir : str, Path
        The path to the directory where subcortical .vtk meshes have been saved (normally, "sub_surfaces/sub-[id]".
    vol_color_map : str
        Name of the color map to be assigned (in alphabetical order) to the background volume, as listed in matplotlib's colormaps. Default is outline_color_map.
    outline_color_map : str
        Name of the color map to be assigned (in alphabetical order) to the subcortical surface outlines, as listed in matplotlib's colormaps. Default is "tab20".
    """
    
    ###################################################################
    ###################################################################
    
    #Load volume
    vol_reader = vtk.vtkNIFTIImageReader()
    vol_reader.SetFileName(volpath)
    vol_reader.Update()
    vol = pv.wrap(vol_reader.GetOutput())
    #convert volume to numpy array for later manipulation
    vol_arr = vol.point_data[vol.active_scalars_name].reshape(vol.dimensions, order="F")
    
    #Load surfaces and store them in an array
    surf_files = sorted(os.listdir(surfdir))
    surfaces = []
    for fname in surf_files:
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(os.path.join(surfdir, fname))
        reader.Update()
        surfaces.append(pv.wrap(reader.GetOutput()))
    
    #assign colors (tab20) to each mesh (adapted to available meshes)
    cmap = mplpyplot.get_cmap(outline_color_map, len(surfaces))
    mesh_colors = [cmap(i) for i in range(len(surfaces))]
    
    #prepare plot
    fig, axes = mplpyplot.subplots(1, 3, figsize=(15, 5))
    mplpyplot.subplots_adjust(bottom=0.25, right=0.8)  #space for buttons, sliders etc
    #base slice values, will change across slider
    x_mid, y_mid, z_mid  = (vol.dimensions[0] // 2, vol.dimensions[1] // 2, vol.dimensions[2]// 2) #middle slices
    slice_idx = {'sag':x_mid, 'ax':y_mid, 'cor':z_mid}
    #default rotation angle, will change at every click
    current_rotation = {'sag':1, 'ax':0, 'cor':2} 
    
    #mesh rotator: cannot simply be done with rot90 due to array structure (no empty space to rotate along with it)
    #a button callback will later modify its parameters for users to click
    def rotate_points(meshcoords, current_rotation, shape):
        height, width = shape #slice size for rotation reference
        if current_rotation == 1:
            return np.column_stack([meshcoords[:,1], width - meshcoords[:,0]]) #turn 90° leftward (moving axes)
        elif current_rotation == 2:
            return np.column_stack([width - meshcoords[:,0], height - meshcoords[:,1]]) #180°
        elif current_rotation == 3:
            return np.column_stack([height - meshcoords[:,1], meshcoords[:,0]]) #270°
        else:   
            return meshcoords #no rotation yet or back to default angle
    
    #draw slice
    def draw_slice(axis):
        subplot = axes[{'sag':0,'ax':1,'cor':2}[axis]]
        subplot.clear()
        
        for surf, color in zip(surfaces, mesh_colors):
            #extract coordinates of intersecting mesh points
            if axis == 'sag': #x axis
                #load slice at that view
                img = vol_arr[slice_idx[axis],:,:].T #slice converted to 2D Numpy array 
                img_rot = np.rot90(img, current_rotation[axis]) #rotate coordinates 90 degrees for volume slice
                subplot.imshow(img_rot, cmap=vol_color_map, origin="lower") #update plotter
                #increase slider value from origin on that specific axis, relative to voxel space
                origin=(vol.origin[0]+slice_idx['sag']*vol.spacing[0],0,0) 
                #identify mesh coordinates at that slice and rotate separately
                outline = surf.slice((1,0,0),origin=origin) #intersect mesh at this slice
                meshcoords = outline.points[:, [1, 2]] #get updated coords for the moving axes
            elif axis == 'ax': #y axis
                img = vol_arr[:,slice_idx[axis],:].T
                img_rot = np.rot90(img, current_rotation[axis]) 
                subplot.imshow(img_rot, cmap=vol_color_map, origin="lower") 
                origin=(0,vol.origin[1]+slice_idx['ax']*vol.spacing[1],0)
                outline = surf.slice((0,1,0),origin=origin)
                meshcoords = outline.points[:, [0, 2]] 
            elif axis == 'cor': #z axis
                img = vol_arr[:,:,slice_idx[axis]].T
                img_rot = np.rot90(img, current_rotation[axis]) 
                subplot.imshow(img_rot, cmap=vol_color_map, origin="lower") 
                origin=(0,0,vol.origin[2]+slice_idx['cor']*vol.spacing[2])
                outline = surf.slice((0,0,1),origin=origin)
                meshcoords = outline.points[:, [0, 1]]
            
            #rotate mesh if present inside volume slice
            if outline.n_points > 0:
                meshcoords = rotate_points(meshcoords, current_rotation[axis], img.shape)
                subplot.plot(meshcoords[:,0], meshcoords[:,1], '.', color=color, markersize=1)
        
        subplot.set_title({'sag':"Sagittal",'ax':"Axial",'cor':"Coronal"}[axis])
        subplot.axis("off")
        fig.canvas.draw_idle()
    
    #initial slice draw
    draw_slice('sag')
    draw_slice('ax')
    draw_slice('cor')
    
    ###################################################################
    ###################################################################
    
    #rotation button 
    def make_rotate_callback(axis):
        def rotate(event):
            current_rotation[axis] = (current_rotation[axis] + 1) % 4 #4 angles, so value can only be cyclic
            draw_slice(axis)
        return rotate
    #slide slice along axis
    def make_slider_callback(axis):
        def update(val):
            slice_idx[axis] = int(val)
            draw_slice(axis)
        return update
    
    #render actual buttons for each subplot
    #sagittal
    button_sag = Button(mplpyplot.axes([0.12, 0.05, 0.05, 0.05]), "↻")
    button_sag.on_clicked(make_rotate_callback('sag'))
    slider_gui_sag = mplpyplot.axes([0.19, 0.05, 0.13, 0.03], facecolor="lightgrey")
    slider_sag = Slider(slider_gui_sag, 'sag', 0, vol.dimensions[0]-1, valinit=x_mid, valfmt='%d')
    slider_sag.on_changed(make_slider_callback('sag'))
    #axial
    button_ax = Button(mplpyplot.axes([0.35, 0.05, 0.05, 0.05]), "↻")
    button_ax.on_clicked(make_rotate_callback('ax'))
    slider_gui_ax = mplpyplot.axes([0.42, 0.05, 0.13, 0.03], facecolor="lightgrey")
    slider_ax = Slider(slider_gui_ax, 'ax', 0, vol.dimensions[1]-1, valinit=y_mid, valfmt='%d')
    slider_ax.on_changed(make_slider_callback('ax'))
    #coronal
    button_cor = Button(mplpyplot.axes([0.58, 0.05, 0.05, 0.05]), "↻")
    button_cor.on_clicked(make_rotate_callback('cor'))
    slider_gui_cor = mplpyplot.axes([0.65, 0.05, 0.13, 0.03], facecolor="lightgrey")
    slider_cor = Slider(slider_gui_cor, 'cor', 0, vol.dimensions[2]-1, valinit=z_mid, valfmt='%d')
    slider_cor.on_changed(make_slider_callback('cor'))
    
    ###################################################################
    ###################################################################
    
    #Add subcortical legend on the right side
    legend_handles = []
    legend_labels = []
    for fname, color in zip(surf_files, mesh_colors):
        handle = mplpyplot.Line2D([0],[0], color=color, marker='.', linestyle='', markersize=10)
        legend_handles.append(handle)
        legend_labels.append(os.path.splitext(fname)[0])
    
    fig.legend(legend_handles, legend_labels,
            loc='center left', bbox_to_anchor=(0.82, 0.45),
            ncol=1, fontsize='small')
    
    mplpyplot.subplots_adjust(wspace=0.017, bottom=0.067)
    mplpyplot.show()
