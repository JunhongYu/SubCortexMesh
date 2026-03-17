<img src="figures/hexlogo.png" alt="logo" align="right" style="float:right; height:200px;"/>

# SubCortexMesh

## Surface-based postprocessing of segmented subcortices in Python

This toolbox provides commands for surface based measures of subcortical segmentations from FreeSurfer and FSL FIRST, including thickness, surface area and curvature. Depending on the segmentation algorithm, compatible subcortices are: the **Thalamus**, **Caudate**, **Putamen**, **Pallidum**, **Hippocampus**, **Amygdala**, **Accumbens-area**, **Ventral diencephalon**, **Cerebellum-Cortex**, and the **Brain-Stem**.

## Workflow

The toolbox automatically converts a subjects directory's subcortical segmentation volumes to 3D surface meshes. SubCortexMesh:

-   Rigidly coregisters subject-space volumes to a template space (MNI). It solely does rotation and no interpolation to preserve subject-space dimensions.

-   Extracts individual regions-of-interest (ROIs) as separate volumes

-   Converts each volume to surface meshes using VTK's Discrete Marching Cube protocol, with additional dilating+eroding and smoothing to minimise artefacts (can be disabled or lowered)

-   Measures vertex-wise thickness (radial distance from a generated medial curve), surface area (1/3 of the area of the triangles a vertex is part of) and curvature (mean curvature).

-   Saves descriptive statistics for each ROI in a table for each subject and metric.

-   Aligns subject meshes and projects their vertex-wise values to a standard template-based surface, based on fsaverage for FreeSurfer outputs,[^readme-1] and MNI152 for FSL FIRST outputs[^readme-2]. The native space meshes, with their associated scalar metrics, can also be saved.

[^readme-1]: The fsaverage/MNI305 surface-based templates for each ROI have been produced by using SubCortexMesh's own functions (subseg_getvol(), vol2surf()) with default parameters, on the fsaverage template's own aseg.mgz in FreeSurfer 7.4.1 (\$FREESURFER_HOME/subjects/fsaverage/mri/aseg.mgz).

[^readme-2]: The fslfirst/MNI152 surface-based templates for each ROI have also been produced by using SubCortexMesh's own functions (subseg_getvol(), vol2surf()) with the same parameters (except smoothing=20). The segmentations volumes were obtained using FSL v.6.0.6's own probabilistic models (.bmv files in \$FSLDIR/data/first/models_336_bin/) on the standard MNI 152 1mm T1w volume (\$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz). Cerebellar meshes were produced using [run_first](https://fsl.fmrib.ox.ac.uk/fsl/docs/structural/first.html#advanced-usage), with "-n 40" modes, using the putamen intensities to normalise its intensity sample as recommended by the latter's documentation.

## Installation

SubCortexMesh can be installed via pip from the GitHub directory:

``` bash
pip install git+https://github.com/chabld/SubCortexMesh.git
```

All SubCortexMesh computations are fully executed in Python, most functions mainly relying on the [VTK](https://vtk.org/) *v*9.5.2 module. Once imported in python, the toolbox requires base template data to be downloaded. The command below is triggered by every function that needs the data. It checks for its existence and if it is not found, it will assist users so they can download it in a default or custom path:

``` python
from subcortexmesh import template_data_fetch
toolboxdata=template_data_fetch(template='fsaverage') #or 'fslfirst'
```

## Getting started

### Extracting and converting subcortical volumes

The registration and extraction of segmentation volumes is done on entire output subjects directories. Typically, a subject directory is `$SUBJECTS_DIR` for FreeSurfer's [recon-all pipeline](https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all), and a directory where [run_first_all](https://fsl.fmrib.ox.ac.uk/fsl/docs/structural/first.html#segmentation-with-run_first_all) has sent its output in FSL (to guarantee the success of that process, we ask users to make a main directory containing one subdirectory for each subject). 

In the subjects directory, SubCortexMesh collates all subjects' subcortical segmentation volumes at once ("aseg.mgz" in FreeSurfer,"*all_fast_firstseg" in FSL). 

T1 volumes are required to coregister the segmentation volumes to their respective template (fsaverage/MNI305 for FreeSurfer, MNI152 for FSL). These should be stored as [sub-ID]/mri/T1.mgz for FreeSurfer and expected to be stored in the same "inputdir" path for FSL FIRST (the function will search for files containing the sub-ID and the "*T1w.nii*" suffix for each subject).

SubCortexMesh follows the sub-xxx convention for participant IDs and will only account for folders that contain such IDs. In FSL FIRST, optional [cerebellar surfaces](https://fsl.fmrib.ox.ac.uk/fsl/docs/structural/first.html#advanced-usage) also need to be named *L-Cereb_first.vtk and *R-Cereb_first.vtk.[^readme-3]

[^readme-3]: Cerebellar segmentations can be done with commands such as (do the same for `L_Cereb`):\
    `first_flirt [subject T1file] "[subject_directory]/[sub-id]" -cort`\
    `run_first -i [subject T1file] \ -t "[subject_directory]/[sub-id_cort.mat]" \ -n 40 \ -o "[subject_directory]/[sub-id]-R_Cereb_first" \ -m "${FSLDIR}/data/first/models_336_bin/intref_puta/R_Cereb.bmv" \  -intref "${FSLDIR}/data/first/models_336_bin/05mm/R_Puta_05mm.bmv"` The putamen used as the intensity reference is what is indicated by [FSL FIRST's documentation](https://fsl.fmrib.ox.ac.uk/fsl/docs/structural/first.html#advanced-usage) and 40 just the number of nodes used for most ROIs.
    
Below is the command that extracts all subcortical volumes (the *toolboxdata* argument only needs to be specified if it was not downloaded to the default path, i.e. the user's home directory).

``` python
from subcortexmesh import subseg_getvol
subseg_getvol(
  inputdir="SPRENG_FreeSurfer_subsample/", 
  outputdir="/my_outputdir/",
  #toolboxdata="/user_path/subcortexmesh_data",
  overwrite=False, 
  silent=False)
```

*subseg_getvol()* will create a directory called "sub_volumes" inside the path given (*outputdir*). The path to sub_volumes can then be given to the following command in order to convert the volumes into surfaces:

``` python
from subcortexmesh import vol2surf
vol2surf(
    inputdir="/my_outputdir/sub_volumes",
    outputdir="/my_outputdir/",
    #dilate_erode=True,
    #smoothing=15,
    plot_volnext2surf=True,
    overwrite=False,
    silent=False
    )
```

The *plot_volnext2surf()* argument is False by default, but allows user to check if they are satisfied with the resulting mesh. Below is an example with a left thalamus: A) shows the volume and the surface created from the former superimposed, B) the voxel-based volume, C) the vertex-wise surface (with VTK' [discrete marching cubes](https://vtk.org/doc/nightly/html/classvtkDiscreteMarchingCubes.html) method, no smoothing), D) the surface after dilation+erosion and smoothing.

![](figures/subseg_getvol_vol2surf.png)

Similarly, *vol2surf()* will create a directory called "sub_surfaces" inside the given outputdir.

### Visualising converted surfaces meshes

The surfaces stored in "sub_surfaces" can also be plotted together on top of a subject's corresponding volume with the surf_qcplot() function. Because the surfaces are based on a volume rigidly coregistered to fsaverage, the surfaces will match a volume generated with subseg_getvol(), i.e. "sub_volumes/sub-[id]/ants_coreg/T1_[template]_rigid_coreg.nii.gz" (or any volume likewise coregistered):

``` python
from subcortexmesh import surf_qcplot
surf_qcplot(
    volpath="/my_outputdir/sub_volumes/sub-[id]/ants_coreg/T1_fsaverage_rigid_coreg.nii.gz",
    surfdir="/my_outputdir/sub_surfaces/sub-[id]",
    vol_color_map= "gray",
    outline_color_map = "tab20"
    )
```

![](figures/surf_qcplot.png)

Since regions will have been inflated and smoothed by default to minimise graphical artefacts (e.g. hanging sparse voxels, sharp mesh spikes), the boundaries naturally appear slightly wider than their original anatomy and overlapping (since the surfaces are processed by SubCortexMesh entirely separately, the visual overlap has no effect on later metrics values). In any event, this can be run on surfaces produced with dilate_erode=False.

### Computing surface-based metrics

The path to sub_surfaces/ can then be given to the following command in order to compute mesh-wise metrics:

``` python
from subcortexmesh import mesh_metrics
mesh_metrics(
  inputdir="/my_outputdir/sub_surfaces",
  outputdir="/my_outputdir/", 
  #toolboxdata="/user_path/subcortexmesh_data",
  template='fsaverage',
  metric=['thickness','surfarea','curvature'], #default, can also be one string
  #roilabel=['left-thalamus','right-thalamus'], #if one wants to only compute specific ROIs
  smooth=[0,5,5], #FMHW smoothing for: thickness, surface area, curvature
  plot_medial_curve=True, 
  plot_projection=True, 
  native_meshes=True, 
  overwrite=True, 
  silent=False)
```

The measure will create a "surface_metrics/" directory in the *outputdir*. The *native_meshes* argument gives the option to also save .vtk surface meshes in native space, containing the scalars for each metric. 

mesh_metrics() also saves native summary statistics tables in each subject directory. E.g., surface_metrics/sub-xxx/surfarea_stats.txt:

| label                | mean  | sd    | min   | max   | range | n_vert |
|:---------------------|:------|:------|:------|:------|:------|:-------|
| left-accumbens-area  | 0.683 | 0.205 | 0.158 | 1.242 | 1.084 | 806    |
| right-accumbens-area | 0.677 | 0.205 | 0.184 | 1.146 | 0.962 | 886    |
| left-amygdala        | 0.685 | 0.199 | 0.159 | 1.254 | 1.095 | 1288   |
| ...                  |       |       |       |       |       |        |

The plotting arguments are also False by default and allow users to check the computation process. The medial curve, the line crossing through the centre of the mesh (A) is fundamental to the thickness measure, as it is a measure of radial distance between the medial curve and the outer surface. It is also used to align the subject-space and template-space mesh further. Likewise, plotting the subject surface (B) and its template equivalent (C) shows how accurate the standardisation is.

![](figures/medialcurve_standardization.png)

### Merging all surfaces

It is possible to treat all ROIs as one and merge their vertex-wise surface metrics into one big mesh. This is what the following command accomplishes, subject-per-subject (provided all subcortical meshes have been created, including the cerebella for FSL FIRST, cf. footnote 3):

``` python
from subcortexmesh import merge_all
merge_all(
  inputdir="/my_outputdir/surface_metrics/,
  #toolboxdata="/user_path/subcortexmesh_data",
  template='fsaverage',
  plot_merged=True, 
  overwrite=False, 
  silent=False)
```

The mesh is saved in the sub_surfaces/ subject directories, e.g. allaseg_thickness.vtk for fsaverage based surfaces. The plotter lets you visualise the outcome of the merging:

![](figures/merged.png)
