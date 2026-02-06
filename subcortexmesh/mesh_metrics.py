#####################################################################################
##########extracting thickness and surface area measures in surface meshes########### 

import vtk
from vtk.util import numpy_support
import numpy as np
import pandas as pd
import os
import re
import sys
from sklearn.decomposition import PCA
from typing import Optional, Union
from pathlib import Path

def mesh_metrics(
    inputdir: Union[str, Path],
    outputdir: Union[str, Path],
    toolboxdata: Optional[Union[str, Path]] = None,
    plot_medial_curve=False,
    plot_projection=False,
    native_metrics=False,
    overwrite=True,
    silent=False,
    ):
    """Thickness and surface area computation
    
    This function extract thickness and surface area metrics from subcortical
    surface objects outputted in vol2surf, via the following steps:
    - It renders medial curves, i.e. singular lines spanning the "core" of the
    meshes, between its two polar axes (estimated with principal component analysis)
    - The medial curve is used to align the subject mesh to the template mesh 
    - Thickness is measured as each surface vertex's distance from the curve 
    (i.e., the radial distance between the surface and the center of the mesh)
    - Surface area is measured as the sum of the area of all triangles a given
    vertex belongs to, divided by 3 
    - Optional (native_metrics, disabled by default): meshes with their thickness and
    surface area scalar values can be saved in native subject space, before projection.
    - Thickness and surface values, separately, are "projected" on an empty template
    mesh via a nearest neighbour approach (i.e., each vertex in the template mesh is
    assigned the thickness/surface value of the closest vertex in the subject mesh)
    - Meshes with their thickness and surface area scalar values can be saved in 
    template space.
    
    Two optional arguments can be provided to visualise the medial curve (plot_medial_curve), 
    and later the projection (plot_projection) in order to check the quality of the medial
    curve and projection for every mesh.
    
    Parameters
    ----------
    inputdir : str, Path
        The path where the surface-based metrics were outputted (using mesh_metrics()).
    outputdir : str, Path
        The path where metrics subcortical meshes will be saved (will create a 
        "surface_metrics" directory).
    toolboxdata : str, Path, optional
        The path of the "subcortexmesh_data" package data directory. The  default path 
        is assumed to be the user's home directory (pathlib's Path.home()). Users will 
        be prompted to download it if not found.
    plot_medial_curve: bool
        Whether to plot the mesh and its computed medial curve. Default is False.
    plot_projection: bool
        Whether to plot the subject-space mesh next to its template-space mesh. 
        Default is False.
    native_metrics: bool, optional
        Whether to save the meshes computed in subject space before they are projected 
        to template space (they will contain one scalar for each metric). Default is 
        False. overwrite must be True for this process to be accounted for.
    overwrite : bool
        Whether files are to be overwritten or skipped if already in outputdir. 
        Default is True.
    silent : bool
        Whether messages about the process are to be printed. Default is False.
    """
    
    #template data is needed
    toolboxdata=template_data_fetch(datapath=toolboxdata, template = 'fsaverage')
    
    #list subjects in the surface subjects directory
    sub_list =[    d for d in os.listdir(inputdir)
        if os.path.isdir(os.path.join(inputdir, d))]
    
    #to silence prints
    vtk.vtkObject.GlobalWarningDisplayOff()
    
    #mesh loader function
    def load_mesh(path):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        return reader.GetOutput()
    
    subindex=0
    for subid in sub_list:
        subindex=subindex+1
        if silent==False: print(f"Computing surface metrics for {subid}... [{subindex}/{len(sub_list)}]")
        #preparing subdir
        os.makedirs(f"{outputdir}/surface_metrics/", exist_ok=True)
        subdir = f"{outputdir}/surface_metrics/{subid}" 
        os.makedirs(subdir, exist_ok=True)
        
        #list vtk surfaces created by s2
        mesh_list=[f for f in os.listdir(f"{inputdir}/{subid}") if f.endswith(".vtk")]
        
        if len(mesh_list) > 0:
            for meshfile in mesh_list:
                    
                    #get base name to name surface the same way
                    base = re.sub(r'\.vtk?$', '', meshfile)
                    
                    if (
                        not os.path.exists(f"{subdir}/{base}_thickness.vtk") 
                        or not os.path.exists(f"{subdir}/{base}_surfarea.vtk") 
                        or overwrite==True
                    ):
                        ###################################################################################
                        ###############################medial curve building###############################
                        
                        if silent==False: print(f"=> Computing {base}...")
                        
                        #This function identifies ends of the mesh, will slice the shape along the Z axis (100 time works well)
                        #and identify the centre of every slice. By tracing through each centre, you get a medial curve in the
                        #core of the shape, linking both ends, which can be used as a reference for thickness.
                        if silent==False: print("   Generating a medial curve...")
                        
                        def extract_medial_curve(mesh, n_slices=100):
                            
                            #make array of vertices 3D coordinates
                            points = np.array([mesh.GetPoint(i) for i in range(mesh.GetNumberOfPoints())])
                            
                            #DIRECTIONAL AXIS IDENTIFICATION
                            #sklearn's principal component analysis, helps find the directional axis across the mesh
                            #get vertices that vary the most in the mesh (i.e., the direction across which it extends the most)
                            #this is used to determine the ends of the surface which works well especially for "cylinder/banana" shapes
                            pca = PCA(n_components=3); pca.fit(points)
                            directions = pca.components_[0] #xyz values of the widest identified directional axis (1st component)
                            projected = points @ directions #dot product: quantifies how far/close each vertex is along that directional axis
                            min_val, max_val = projected.min(), projected.max() #ends are the farthest vertices on both sides of that axis
                            
                            #for more vertical structure, the ends are more straight forward ("inferior"-"superior" ends)
                            if ("cerebellum" in base) or ("brain-stem" in base):
                                superior_point = points[np.argmin(points[:,1])]  #idx with lowest Y
                                inferior_point = points[np.argmax(points[:,1])]  #idx with highest Y
                                # axis vector from superior to inferior
                                directions = inferior_point - superior_point
                                directions = directions / np.linalg.norm(directions)  # normalize
                                # project points along this axis
                                projected = points @ directions
                                min_val, max_val = projected.min(), projected.max()
                            
                            slice_range = np.linspace(min_val, max_val, n_slices) #100 points in between where a slice will be made
                            slice_positions = [directions * pos for pos in slice_range] #each position gets assigned a 3D coordinate
                            
                            #CENTROIDS DEFINITION
                            #get centroid of each slice (through which the medial curve will be drawn)
                            centroids = []
                            for position in slice_positions:
                                plane = vtk.vtkPlane() #makes artificial plane
                                #use directional axis so vtk knows how is the slice angled
                                plane.SetOrigin(position); plane.SetNormal(directions) 
                                cutter = vtk.vtkCutter() #create a slice using the plane
                                cutter.SetCutFunction(plane); cutter.SetInputData(mesh); cutter.Update()
                                slice_poly = cutter.GetOutput()
                                #only get centroid if the slice actually cuts any 3D space (otherwise, 0 vertex)
                                if slice_poly.GetNumberOfPoints() > 0:
                                    pts = np.array([slice_poly.GetPoint(i) for i in range(slice_poly.GetNumberOfPoints())])
                                    centroids.append(pts.mean(axis=0))
                              
                            #FLATTENING OF AXIS ENDS
                            #because the ends are usually differently shaped (not clear cut cylinder), 
                            #the slicing tends to go sideways at the ends of the curve, so we stop "drawing" after 80% of the 
                            #curve is complete, the last 10% will be assumed to be going straight along the same "core" direction. 
                            #It's an assumption but it works well for aseg's subcortices
                            #get "core" 80% of the medial curve
                            core = centroids[int(len(centroids) * 0.1):int(len(centroids) * 0.9)] #removes 10% tails of slices
                            #Will extend each end (remaining 10% tails) in a straight line until centroids intersect with mesh
                            box = vtk.vtkOBBTree() #make a "oriented bounding box" that will be used to detect when line reach boundary
                            box.SetDataSet(mesh); box.BuildLocator()
                            def extender(start_pt, direction): 
                                #max_dist in case the line gets drawn outside by error, in which case it would extend infinitely
                                max_dist=50; step=0.5
                                for i in range(1, int(max_dist / step)): #iterates every new line segment added
                                    next_pt = start_pt + direction * (i * step)
                                    points = vtk.vtkPoints() #placeholder for coordinate of the intersection
                                    box.IntersectWithLine(start_pt, next_pt, points, None) #check if line intersects with box
                                    if points.GetNumberOfPoints() > 0: #vertices at that point, if there are some then you've reached the mesh 
                                        return np.array(points.GetPoint(0)) #return intersection's coordinate
                                        
                                return start_pt + direction * max_dist #if it never gets found, will define end as max_dist from start
                              
                            #define the two end directions 
                            dir_start = -directions; ext_start = extender(core[0], -directions);
                            dir_end = directions; ext_end = extender(core[-1], directions)
                            extended_core = [ext_start] + core + [ext_end]
                            
                            #LINE BUILDING
                            #builds 3D line (made of centroids)
                            spline_points = vtk.vtkPoints()
                            for pt in extended_core:
                                spline_points.InsertNextPoint(pt)
                            
                            parametric_spline = vtk.vtkParametricSpline()
                            parametric_spline.SetPoints(spline_points)
                            
                            #smooth curve
                            spline_source = vtk.vtkParametricFunctionSource()
                            spline_source.SetParametricFunction(parametric_spline)
                            spline_source.SetUResolution(25)  #N subdivisions
                            spline_source.Update()
                            
                            core_poly = spline_source.GetOutput()
                            
                            #OUTPUT MEDIAL CURVE
                            medial_curve = core_poly
                            return medial_curve
                        
                        #load mesh of interest
                        subject_mesh = load_mesh(f"{inputdir}/{subid}/{meshfile}")
                        #compute medial_curve
                        subject_medial_curve = extract_medial_curve(subject_mesh, n_slices=100)
                        
                        if plot_medial_curve==True:
                            #script to plot (transparent) surface and its medical curve crossing through
                            vis_medialcurve(subject_mesh, subject_medial_curve, base, subid)
                            
                        ###################################################################################
                        ###################################thickness measure###############################
                        
                        #Thickness is basically radial distance from the medial curve
                        #Distance is calculated in native space
                        #will later be "projected" on the template space after measure
                            
                        if (not os.path.exists(f"{inputdir}/{subid}/{base}_thickness.vtk")) or overwrite==True:
                            if silent==False: print("   Computing thickness...")
                            getting_thickness=True
                            #line to rendered as a 3D tube for VTK's implicit distance filter to actually work
                            tube_filter = vtk.vtkTubeFilter()
                            tube_filter.SetInputData(subject_medial_curve)
                            tube_filter.SetRadius(0.01) #minimum size to mimic the line
                            tube_filter.SetNumberOfSides(20) #polygons to make each segment a round circle
                            tube_filter.CappingOff()
                            tube_filter.Update()
                            medial_curve_tube = tube_filter.GetOutput()
                            
                            def compute_thickness_to_medial(mesh, medial_curve_tube):
                                distance_filter = vtk.vtkImplicitPolyDataDistance()
                                distance_filter.SetInput(medial_curve_tube) #set reference for distances
                                thickness = vtk.vtkFloatArray(); thickness.SetName("thickness") #prepare thickness values scalar
                                for i in range(mesh.GetNumberOfPoints()): #for each vertex
                                    d = distance_filter.EvaluateFunction(mesh.GetPoint(i)) #calculate distance to medial curve
                                    thickness.InsertNextValue(abs(d)) #abs for unsigned thickness
                                return thickness
                            
                            #add thickness values to subject mesh
                            subject_thickness = compute_thickness_to_medial(subject_mesh, medial_curve_tube)
                            _ = subject_mesh.GetPointData().AddArray(subject_thickness)
                        else:
                            if silent==False: print(f"  Thickness already computed.")
                        
                        
                        ###################################################################################
                        ###############################surface area measure################################
                        
                        #Following FreeSurfer's approach:
                        #"the area of a vertex is 1/3 of the area of all triangles that it is a member of, and it should sum to the area of the surface"
                        #https://www.mail-archive.com/freesurfer@nmr.mgh.harvard.edu/msg07738.html
                        #"is surface area represented vertex-wise, with a vertex' adjacent face's area added and divided by 3?
                        #-That is how the ?h.area file is created."
                        #https://www.mail-archive.com/freesurfer@nmr.mgh.harvard.edu/msg29355.html
                        
                        if (not os.path.exists(f"{inputdir}/{subid}/{base}_surfarea.vtk")) or overwrite==True:
                            if silent==False: print("   Computing surface area...")
                            getting_surfarea=True
                            def get_surface_area(subject_mesh):
                                #read vertex coordinates
                                points = subject_mesh.GetPoints()
                                n_points = points.GetNumberOfPoints()
                                #read triangles
                                cells = subject_mesh.GetPolys()
                                cells.InitTraversal()
                                id_list = vtk.vtkIdList()
                                
                                #prepare vertex-wise surfarea array
                                vertexwise_area = np.zeros(n_points)
                                
                                #for each triangle 
                                while cells.GetNextCell(id_list):
                                    
                                    #catch non-triangles and skip (shouldn't be there anyways)
                                    if id_list.GetNumberOfIds() != 3:
                                        continue  
                                    #get vertices of the triangle
                                    v1 = np.array(points.GetPoint(id_list.GetId(0)))
                                    v2 = np.array(points.GetPoint(id_list.GetId(1)))
                                    v3 = np.array(points.GetPoint(id_list.GetId(2)))
                                    
                                    #triangle area
                                    #http://dk81.github.io/dkmathstats_site/linalg-area-paraellogram.html#examples
                                    #calculate two edges distance (xyz difference) vectors for two faces of the triangle
                                    e1=v2-v1; e2=v3-v1
                                    #cross product of the two returns a perpendicular distance vector (forming a parallelogram)
                                    edgespan=np.cross(e1, e2)
                                    #the norm of this vector is the area spanned by the parallelogram
                                    area=np.linalg.norm(edgespan) 
                                    area=area / 2.0 #the triangle is half that of the parallelogram
                                    
                                    #Each connected vertex get added 1/3 of that area
                                    for i in range(3):
                                        vertexwise_area[id_list.GetId(i)] += area / 3.0
                                    
                                #convert to vtkFloatArray
                                vertexwise_area_vtk = numpy_support.numpy_to_vtk(num_array=vertexwise_area, deep=True, array_type=vtk.VTK_FLOAT)
                                vertexwise_area_vtk.SetName("surfarea")
                                    
                                return vertexwise_area_vtk
                            
                            #add surface area array to the mesh
                            _ = subject_mesh.GetPointData().AddArray(get_surface_area(subject_mesh))
                        else:
                            if silent==False: print(f"  Surface area already computed.")
                        
                        
                        ###################################################################################
                        #########################native-space metrics######################################
                        
                        #If you also want the vertex-wise metrics NOT projected to fsaverage space,
                        #you can save the aligned native space mesh (thickness and surfarea scalars attached)
                        if native_metrics==True:
                            if silent==False: print(f"   Saving subject-space mesh to {subid}/native_space/...")
                            os.makedirs(f"{subdir}/native_space/", exist_ok=True)
                            writer = vtk.vtkPolyDataWriter()
                            writer.SetFileName(f"{subdir}/native_space/{base.lower()}_native.vtk")
                            writer.SetInputData(subject_mesh)
                            _ = writer.Write()
                        
                        if silent==False: print(f"   Printing descriptive statistics...")
                        #print native-space statistics summary either way
                        print_stats(subdir, subject_mesh, base)
                        
                        ###################################################################################
                        #######################subject-to-template registration############################
                        
                        #We align the two object's medial curves via rigid rotation (to later improve the native-to-fsaverage values projection)
                        if silent==False: print("   Alignment to fsaverage template...")
                        
                        #prepare the fsaverage template's mesh and curve
                        template_mesh = load_mesh(f"{toolboxdata}/template_data/fsaverage/surfaces/{meshfile}")
                        template_medial_curve = extract_medial_curve(template_mesh, n_slices=100)
                        
                        #function to get vertices to np coordinate arrays (N rows * 3 (xyz) columns)
                        def pts_to_array(polydata):
                            pts = polydata.GetPoints()
                            return np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())], dtype=np.float64)
                        
                        #reduce curve points to equal count
                        #function to reduce number of points in the curve for stability/speed (a maximum of 200 works well)
                        def uniform_subsample(points):
                            if len(points) <= 200:
                                return points
                                
                            idx = np.linspace(0, len(points) - 1, 200).round().astype(int) #evenly spaced along curve
                            return points[idx]
                        
                        #custom procrustres function (no scaling)
                        #aligns subject curve's coordinates without resizing to the template's curve coordinates
                        def rigid_procrustes(X, Y):
                            #each curve imported as array (here X is subject, Y is template)
                            X = np.asarray(X, dtype=np.float64) 
                            Y = np.asarray(Y, dtype=np.float64)
                            #mean centring all coordinate values along the origin to facilitate rotation
                            mx = X.mean(axis=0); my = Y.mean(axis=0) #means of x,y,z coordinates
                            Xc = X - mx; Yc = Y - my
                           
                            #solving the "procrustes problem" with alignment only
                            #basically, it finds an optimal orthogonal matrix with the best rotation of xyz required to align X's points to Y's 
                            cov_matrix = Xc.T @ Yc #gets dot product (covariance) between each possible pair of axes across points (3x3 matrix)
                            U, _, V = np.linalg.svd(cov_matrix) #Singular Value Decomposition (SVD)
                            #SVD basically gets rotation/reflection (across objects X-to-Y for matrix U, Y-to-X for matrix V) based on cov matrix
                            R = V.T @ U.T #the best rotation according to the procrustes solution
                            
                            #check if the matrix' determinant is -1: if so, means the points were mirrored to fit,
                            #but we want to preserve hemispheric "orientation", so signs are flipped back in that case
                            if np.linalg.det(R) < 0:
                                V[-1, :] *= -1
                                R = V.T @ U.T 
                                
                            #get translation vector needed to shift rotated X so its points align with Y's
                            #while X is rotated optimally along the same axis, we also want it in the same space (without offset)
                            #basically the difference between template's centroids and subject's centroids along axis
                            t = my - R @ mx 
                            return R, t
                        
                        def align_subject_by_curve(subject_mesh, template_mesh,
                                                   subject_medial_curve, template_medial_curve):
                            #get curves
                            subj_curv = uniform_subsample(pts_to_array(subject_medial_curve))
                            tmpl_curv = uniform_subsample(pts_to_array(template_medial_curve))
                            #pad or truncate minimally if one curve smaller than 200
                            m = min(len(subj_curv), len(tmpl_curv))
                            subj_curv, tmpl_curv = subj_curv[:m], tmpl_curv[:m]
                            
                            #estimate rigid transform from curves
                            R, t = rigid_procrustes(subj_curv, tmpl_curv)
                            
                            #apply obtained rotation/shift transforms the subject's points
                            subj_aligned = vtk.vtkPolyData()
                            subj_aligned.DeepCopy(subject_mesh)
                            V = pts_to_array(subj_aligned)
                            V_aligned = (R @ V.T).T + t #applies optimal rotation R, and shift t to its coordinates
                            
                            #convert array back to VTK polydata points (different from arrays)
                            vtk_pts = vtk.vtkPoints()
                            vtk_pts.SetNumberOfPoints(len(V_aligned))
                            for i, p in enumerate(V_aligned):
                                vtk_pts.SetPoint(i, float(p[0]), float(p[1]), float(p[2]))
                            subj_aligned.SetPoints(vtk_pts)
                            subj_aligned.Modified()
                            
                            return subj_aligned, R, t
                        
                        aligned_subject_mesh, R, t = align_subject_by_curve(
                            subject_mesh, template_mesh,
                            subject_medial_curve, template_medial_curve)
                        
                        #additional rigid alignment of subject mesh to template mesh at the vertex level 
                        #for optimisation (Iterative Closest Point)
                        icp = vtk.vtkIterativeClosestPointTransform()
                        icp.SetSource(aligned_subject_mesh)
                        icp.SetTarget(template_mesh)
                        icp.GetLandmarkTransform().SetModeToRigidBody()
                        icp.SetMaximumNumberOfIterations(50)
                        icp.Update()
                        transform_filter = vtk.vtkTransformPolyDataFilter()
                        transform_filter.SetInputData(aligned_subject_mesh)
                        transform_filter.SetTransform(icp)
                        transform_filter.Update()
                        aligned_subject_mesh = transform_filter.GetOutput()
                        
                        ########################################################################################
                        ##############################native-to-template projection#############################
                        
                        #We use he Nearest Neighbor method to project native values to template space 
                        #works fairly well as shapes are not too complex and aligned
                        if silent==False: print("   Projecting subject-space data to template space...")
                        
                        def native_to_template(aligned_subject_mesh, template_mesh, scalarname):
                            #setting locator for subject mesh points
                            locator = vtk.vtkPointLocator()
                            locator.SetDataSet(aligned_subject_mesh)
                            locator.BuildLocator()
                            #activated named scalar
                            aligned_subject_mesh.GetPointData().SetActiveScalars(scalarname)
                            
                            #prepare projected scalar array
                            template_scalar_array = vtk.vtkFloatArray()
                            template_scalar_array.SetName(scalarname)
                            
                            #For each vertex in the template mesh, assign scalar value
                            for i in range(template_mesh.GetNumberOfPoints()):
                                pt = template_mesh.GetPoint(i)
                                nearest_id = locator.FindClosestPoint(pt)
                                if nearest_id >= 0 and nearest_id < aligned_subject_mesh.GetPointData().GetScalars().GetNumberOfTuples():
                                    template_scalar_array.InsertNextValue(aligned_subject_mesh.GetPointData().GetScalars().GetValue(nearest_id))
                                else:
                                    template_scalar_array.InsertNextValue(0.0)  #if no projection feasible
                                
                            _ =template_mesh.GetPointData().AddArray(template_scalar_array)
                            
                            #save mesh in template space
                            if silent==False: print(f"   Saving to {subid}/{base.lower()}_{scalarname.lower()}.vtk")
                            saved_mesh = vtk.vtkPolyData(); saved_mesh.DeepCopy(template_mesh) 
                            saved_mesh.GetPointData().SetScalars(template_scalar_array)
                            writer = vtk.vtkPolyDataWriter()
                            writer.SetFileName(f"{subdir}/{base.lower()}_{scalarname.lower()}.vtk")
                            writer.SetInputData(saved_mesh)
                            _ = writer.Write()
                            
                            return template_mesh
                        
                        #project subject's scalars to template space
                        if getting_thickness==True:
                            subject_mesh_fsaverage=native_to_template(aligned_subject_mesh, template_mesh, 'thickness')
                        if getting_surfarea==True:
                            subject_mesh_fsaverage=native_to_template(aligned_subject_mesh, template_mesh, 'surfarea')
                        
                        #########################################################################
                        ##############################plot#######################################
                        
                        if plot_projection==True:
                            #to visualise subject and template's thickness surfaces next to each other
                            if getting_thickness==True:
                                vis_nativetotemplate(aligned_subject_mesh, subject_mesh_fsaverage, 'thickness', base)
                            if getting_surfarea==True:
                                vis_nativetotemplate(aligned_subject_mesh, subject_mesh_fsaverage, 'surfarea', base)
                         
                    else:
                        if silent==False: print(f"=> {base} surface metrics already computed.")
        else:
            if silent==False: print(f"=> No mesh file (.vtk) found at all for {subid}.")


###################################################################
###################################################################
#function to write down summary statistics in a .txt

def print_stats(subdir, mesh, base):
    #one table for each metric separately
    for scalarname in ['thickness', 'surfarea']:
        if mesh.GetPointData().HasArray(scalarname):
            #predefined row and column names 
            labels = ["left-accumbens-area", "right-accumbens-area", "left-amygdala", "right-amygdala",
                "left-caudate", "right-caudate", "left-cerebellum-cortex", "right-cerebellum-cortex",
                "left-hippocampus", "right-hippocampus", "left-pallidum", "right-pallidum",
                "left-putamen", "right-putamen", "left-thalamus", "right-thalamus",
                "left-ventraldc", "right-ventraldc", "brain-stem"]
            columns = ["mean", "sd", "min", "max", "range", "n_vert"]
            
            #load or initialize table
            outfile = f"{subdir}/{scalarname}_stats.txt"
            if os.path.exists(outfile):
                df = pd.read_csv(outfile, sep="\t", index_col="label")
            else:
                df = pd.DataFrame(index=labels, columns=columns) 
            
            #get scalar's stats for that ROI
            arr = mesh.GetPointData().GetArray(scalarname)
            data = np.array([arr.GetValue(i) for i in range(arr.GetNumberOfTuples())])
            stats= [np.mean(data), np.std(data), np.min(data), np.max(data), np.max(data) - np.min(data), 
                    mesh.GetNumberOfPoints()]
            df.loc[base] = stats
            # Save table
            df.to_csv(outfile, sep="\t", index_label="label", float_format="%.3f")

###################################################################
###################################################################
#visualiser to look at the surface with the medial curve crossing through

def vis_medialcurve(mesh, medial_curve, base, subid):
    #surface mesh 
    subject_mapper = vtk.vtkPolyDataMapper()
    subject_mapper.SetInputData(mesh)
    subject_mapper.SetScalarVisibility(False)
    subject_actor = vtk.vtkActor()
    subject_actor.SetMapper(subject_mapper)
    subject_actor.GetProperty().SetColor(1, 0, 0) #red
    subject_actor.GetProperty().SetOpacity(0.3)
    
    #curve mesh
    curve_mapper = vtk.vtkPolyDataMapper()
    curve_mapper.SetInputData(medial_curve)
    curve_actor = vtk.vtkActor()
    curve_actor.SetMapper(curve_mapper)
    curve_actor.GetProperty().SetColor(0, 0, 0) #black
    curve_actor.GetProperty().SetLineWidth(3.0)
    
    #render
    renderer = vtk.vtkRenderer()
    renderer.AddActor(subject_actor)
    renderer.AddActor(curve_actor)
    renderer.SetBackground(1, 1, 1)
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1200, 800)
    #flip Y axis of camera as VTK coord syst not following RAS
    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    camera.SetViewUp(0, -1, 0)
    renderer.ResetCameraClippingRange()
    
    #add interactor for rotation
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    render_window.Render()
    render_window.SetWindowName(f"{subid} - {base} medial curve")
    interactor.Initialize()
    interactor.Start()

###################################################################
###################################################################
#visualiser for the subject and template's surface-based data next to each other and check
#if the projection from native space to template space was successful

def vis_nativetotemplate(aligned_subject_mesh, subject_mesh_fsaverage, scalarname, base):
        #use scalar defined
        aligned_subject_mesh.GetPointData().SetActiveScalars(scalarname)
        subject_mesh_fsaverage.GetPointData().SetActiveScalars(scalarname)
        
        #position shift so meshes are plotted side-by-side
        def mesh_shifter(mesh, z_offset):
            transform = vtk.vtkTransform()
            transform.Translate(0, z_offset, 0)
            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetInputData(mesh)
            transform_filter.SetTransform(transform)
            transform_filter.Update()
            return transform_filter.GetOutput()
        
        #distance between the two meshes - more space needed for cerebellum and brainstem
        if subject_mesh_fsaverage.GetNumberOfPoints() > 8000:
            subject_surf = mesh_shifter(aligned_subject_mesh, -60)
        else:
            subject_surf = mesh_shifter(aligned_subject_mesh, -30)
        
        fsaverage_surf = mesh_shifter(subject_mesh_fsaverage, 0)
        
        #mappers (will assign vertex-wise values if exist)
        def make_mapper(mesh, use_scalars=True):
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(mesh)
            mapper.SetScalarVisibility(use_scalars)
            
            scalars = mesh.GetPointData().GetScalars()
            if scalars:
                mapper.SetScalarRange(scalars.GetRange())
            
            return mapper
        
        subject_mapper = make_mapper(subject_surf, use_scalars=True)
        fsaverage_mapper = make_mapper(fsaverage_surf, use_scalars=True)
        
        #actors
        def make_actor(mapper):
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*(1, 0, 0)) #default color is all red if scalars missing
            return actor
        
        subject_actor = make_actor(subject_mapper)  
        fsaverage_actor = make_actor(fsaverage_mapper)   
        
        #add label when hovering with cursor
        def add_hover_label(interactor, renderer, actor, label_text):
            # Create a text actor (initially hidden)
            text_actor = vtk.vtkTextActor()
            text_actor.SetInput(label_text)
            text_actor.GetTextProperty().SetColor(0, 0, 0)
            text_actor.GetTextProperty().SetFontSize(18)
            text_actor.SetVisibility(False)
            renderer.AddViewProp(text_actor)
            picker = vtk.vtkCellPicker() #Detects hovering
            picker.SetTolerance(0.0005)
            
            def mouse_move_callback(obj, event):
                x, y = interactor.GetEventPosition()
                picker.Pick(x, y, 0, renderer)
                picked_actor = picker.GetActor()
                if picked_actor == actor:
                    # Show label near cursor
                    text_actor.SetPosition(x + 10, y + 10)
                    text_actor.SetVisibility(True)
                else:
                    text_actor.SetVisibility(False)
                
                interactor.Render()
            
            interactor.AddObserver("MouseMoveEvent", mouse_move_callback)
        
        #renderer setup
        renderer = vtk.vtkRenderer()
        renderer.AddActor(subject_actor)
        renderer.AddActor(fsaverage_actor)
        renderer.SetBackground(1, 1, 1)
        #window
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(1200, 800)
        #flip Y axis of camera as VTK coord syst not following RAS
        renderer.ResetCamera()
        camera = renderer.GetActiveCamera()
        camera.SetViewUp(0, -1, 0)
        renderer.ResetCameraClippingRange()
        #interactor to allow rotation
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        
        add_hover_label(interactor, renderer, fsaverage_actor, "FsAverage")
        add_hover_label(interactor, renderer, subject_actor, "Subject")
        
        #start window
        render_window.Render()
        render_window.SetWindowName(f"{base} {scalarname}")
        
        #add color bar for the scalars
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(subject_mapper.GetLookupTable())
        scalar_bar.SetNumberOfLabels(5)
        scalar_bar.SetTitle(aligned_subject_mesh.GetPointData().GetScalars().GetName())
        scalar_bar.GetLabelTextProperty().SetColor(0, 0, 0)
        scalar_bar.GetTitleTextProperty().SetColor(0, 0, 0)
        renderer.AddViewProp(scalar_bar)
        
        interactor.Initialize()
        interactor.Start()
    

