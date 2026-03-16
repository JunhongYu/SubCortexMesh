#####################################################################################
######Creates a large mesh collating all aseg subcortices into one big surface#######

import pyvista as pv
import numpy as np
import pandas as pd 
import os
import vtk
from typing import Optional, Union
from pathlib import Path
from subcortexmesh import template_data_fetch

def merge_all(
    inputdir: Union[str, Path],
    toolboxdata: Optional[Union[str, Path]] = None,
    template=str,
    plot_merged=False,
    overwrite=True,
    silent=False,
):
    """Merging all subcortical outputs into a single surface object
    
    This function creates a new mesh merging all subcortical meshes outputted by 
    mesh_metrics() in a given template, for available metrics separately, keeping 
    their vertex-wise values. It will only work if all subcortices have been 
    processed. The merged mesh will be saved along the surface meshes in the 
    directories used as input.
    
    For fslfirst, the cerebella need to have been created in FSL FIRST inside the same
    output directory as run_first_all's, naming them "*R_Cereb_first" and "*L_Cereb_first". 
    It can be done as follows (do the same for `L_Cereb`):
            #generate cort-specific registration matrix for cerebellum
    #    first_flirt [subject T1file] "[subject_directory]/[sub-id]" -cort
    #    run_first -i [subject T1file] \
    #        -t "[subject_directory]/[sub-id_cort.mat]" \
    #        -n 40 \
    #        -o "[subject_directory]/[sub-id]-R_Cereb_first" \
    #        -m "${FSLDIR}/data/first/models_336_bin/intref_puta/R_Cereb.bmv" \
    #        -intref "${FSLDIR}/data/first/models_336_bin/05mm/R_Puta_05mm.bmv"       
    Then simply import them using SubCortexMesh's fslfirst_getsurf() and mesh_metrics().
     
    Parameters
    ----------
    inputdir : str, Path
        The path where the surface-based metrics were outputted (using mesh_metrics()).
        The outputdir will be the inputdir.
    toolboxdata : str, Path, optional
        The path of the "subcortexmesh_data" package data directory. The  default path 
        is assumed to be the user's home directory (pathlib's Path.home()). Users will 
        be prompted to download it if not found.
    template: str
        The name of the template the surfaces are supposed to be matching to. For surfaces
        obtained via aseg_gevtol()/vol2surf(), it is 'fsaverage'. For surfaces
        obtained via fslfirst_getsurf(), it is 'fslfirst'.
    plot_merged: bool
        Whether to plot the resulting merged mesh. Default is False.
    overwrite : bool
        Whether files are to be overwritten or skipped if already made. Default 
        is True.
    silent : bool
        Whether messages about the process are to be printed. Default is False.
    """
    
    ###################################################################
    ###################################################################
    
    #template data is needed
    toolboxdata=template_data_fetch(datapath=toolboxdata, template = template)
    if template=='fsaverage':
        mergedmesh='allaseg'
        nroi=19
    if template=='fslfirst':
        mergedmesh='allfslfirst'
        nroi=17
    
    #Subfunctions
    #mesh loader function
    def load_mesh(path):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        return reader.GetOutput()
    
    #merge all meshes together, storing the ROI labels as an in-vtk array
    def mesh_merger():
        appendFilter = vtk.vtkAppendPolyData()
        for i, mesh in enumerate(mesh_list):
            meshfile=load_mesh(f"{inputdir}/{subid}/{mesh}")
            
            # Add string array with ROI name
            tagArray = vtk.vtkIntArray()
            tagArray.SetName("roi_id")
            tagArray.SetNumberOfTuples(meshfile.GetNumberOfPoints())
            tagArray.FillComponent(0, i)  # i = ROI index
            meshfile.GetPointData().AddArray(tagArray)
            
            _ = meshfile.GetPointData().AddArray(tagArray)
            appendFilter.AddInputData(meshfile)
        
        appendFilter.Update()
        merged_mesh = appendFilter.GetOutput()
        
        if plot_merged:
            vis_merged()
        
        return merged_mesh
    
    ###################################################################
    ###################################################################
    
    #plot function that plots all aseg surfaces together, and gives a slider to space them out from eachother
    def vis_merged():
        #spaced out
        #get all mesh 
        all_meshes = [ load_mesh(os.path.join(inputdir, subid, fname)) for fname in mesh_list ]
        #all vertex coordinates
        all_points = np.vstack([pv.wrap(m).points for m in all_meshes])
        #get centroid across all meshes
        global_centroid = all_points.mean(axis=0) 
        
        plotter = pv.Plotter()
        wrapped_meshes = [pv.wrap(m) for m in all_meshes]
        #get mesh-wise centroids coordinates
        centroids = [wm.points.mean(axis=0) for wm in wrapped_meshes]
        #original points from which the slider will start
        original_points = [wm.points.copy() for wm in wrapped_meshes]
        #append meshes in plot
        for wm in wrapped_meshes:
            _ = plotter.add_mesh(wm, scalars=metric, cmap="viridis")
        #Y flipped as VTK's coord syst not following RAS
        plotter.reset_camera()
        loc, foc, _ = plotter.camera_position
        plotter.camera_position = [loc, foc, (0, -1, 1)]
        #create a slider that increases or decrease distance from global centroid
        def update_distance(distfactor):
            for wm, centroid, orig in zip(wrapped_meshes, centroids, original_points):
                direction = centroid - global_centroid
                norm = np.linalg.norm(direction) #get normals
                if norm > 0:
                    direction = direction / norm
                translation = direction * distfactor 
                wm.points[:] = orig - centroid + (centroid + translation)
            plotter.render()
        
        plotter.add_slider_widget(update_distance, rng=[0, 50], value=0)
        plotter.show()
    
    ###################################################################
    ###################################################################
    
    #run on subject output directories
    #list subjects in the surface subjects directory
    sub_list =[    d for d in os.listdir(inputdir)
        if os.path.isdir(os.path.join(inputdir, d))]
    
    for subid in sub_list:
        
        if not silent: 
            print(f"Creating all-aseg surfaces for {subid}...")
        
        for metric in ['thickness', 'surfarea', 'curvature']:
            
            if not os.path.exists(f"{inputdir}/{subid}/{mergedmesh}_{metric}.vtk") or overwrite:
                
                #listing files for that metric
                mesh_list = [
                    f for f in os.listdir(f"{inputdir}/{subid}")
                    if f"{metric}" in f and not f.startswith(f"{mergedmesh}_{metric}") #explicitly do not list the merged vtk
                ]
                
                if len(mesh_list) > 0:
                    
                    if len(mesh_list) != nroi:
                        if not silent: 
                            print(f"{metric} ignored: all subcortices of this template ({nroi}) must be available to create the template-wide surface.")
                    else:
                        if not silent: 
                            print(f"=> Merging {metric} ...")
                        
                        #force the mesh_list follow templates' ROI order as in the lookup table
                        roi_lookup = pd.read_csv(f"{toolboxdata}/template_data/{template}/surfaces/{mergedmesh}_roi_id.txt",sep="\t")
                        roi_order = roi_lookup['label'].tolist()
                        #reorder mesh_list
                        mesh_list_sorted = []
                        for roi in roi_order:
                            #find filename in mesh_list that starts with ROI label
                            matches = [f for f in mesh_list if f.startswith(roi) and f.endswith(".vtk")]
                            if matches: 
                                mesh_list_sorted.extend(matches)
                        mesh_list = mesh_list_sorted
                        
                        #merge mesh
                        merged_mesh=mesh_merger()
                            
                        #save it
                        writer = vtk.vtkPolyDataWriter()
                        writer.SetFileName(f"{inputdir}/{subid}/{mergedmesh}_{metric}.vtk")
                        writer.SetInputData(merged_mesh)
                        _ = writer.Write()
                else:
                    if not silent: 
                        print(f"No mesh file (.vtk) found at all for {subid}'s surface {metric}.")
            else:
                if not silent: 
                    print(f"=> {metric} already merged")