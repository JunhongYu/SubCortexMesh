#####################################################################################
##########Statistical analyses using random field theory cluster correction##########

import vtk
from vtk.util import numpy_support
import numpy as np
import pyvista as pv
from typing import Optional, Union, Sequence, Tuple
from pathlib import Path
from brainstat.stats.SLM import SLM
from brainstat._typing import ArrayLike
from brainstat.mesh.data import mesh_smooth
from brainspace.mesh import mesh_io

def slm_analysis(
    inputdir: Union[str, Path],
    template: str,
    metric: Union[str, Sequence[str]],
    roilabel: str,
    model: ArrayLike, 
    contrast: ArrayLike, 
    mask: ArrayLike = None,
    correction: Optional[Union[str, Sequence[str]]] = None,
    thetalim: float = 0.01,
    drlim: float = 0.1,
    two_tailed: bool = True,
    cluster_threshold: float = 0.001,
    smooth: Optional[float] = None,
    sub_list: Optional[Sequence[str]] = None
    ):
    
    """Vertex-wise analysis with random field theory cluster correction
    
    This function is a wrapper for BrainStat's SLM (Standard Linear regression models)
    modelling functions, which fit linear or linear mixed models with surface-based 
    values, with random field theory cluster correction. You may refer to their 
    tutorial to understand the model and contrast arguments:
    https://brainstat.readthedocs.io/en/master/python/tutorials/tutorial_1.html
    
    slm_analysis() automatically collates surface-based measures in template space from a 
    surface_metrics/ folder produced by mesh_metrics() (or equivalent folder), and fits 
    them as the "surf" argument in BrainStat. The model returns t-statistics, p-values,
    and Random field theory clusters map for a selected contrast. 
    
    Parameters
    ----------
    inputdir : str, Path
        The surface_metrics/ directory where the surface-based metrics were outputted 
        (using mesh_metrics()) or a directory with the same tree structure (subject folders,
        each with metrics .vtk files inside).
    template: str
        The name of the template the surfaces are supposed to be matching to. For FreeSurfer
        outputs, it is 'fsaverage'. For FSL FIRST, it is 'fslfirst'.
    roilabel: str
        The name of the region-of-interest to analyse (one at a time): 'left-cerebellum-cortex', 'right-cerebellum-cortex', 'left-pallidum', 'right-pallidum', 'left-putamen', 
        'right-putamen', 'left-thalamus', 'right-thalamus','left-amygdala',  'right-amygdala', 'left-hippocampus', 'right-hippocampus', 'left-accumbens-area','right-accumbens-area','left-caudate', 'right-caudate', 'left-ventraldc', 'right-ventraldc', and 'brain-stem'.
    model : brainstat.stats.terms.FixedEffect, brainstat.stats.terms.MixedEffect
        The linear model to be fitted of dimensions (observations, predictors).
    contrast : array-like
        Vector of contrasts from the observations.
    mask: array-like, optional
        A mask containing True for vertices to include in the analysis, by
        default None.
    correction: str, Sequence, optional 
        String or sequence of strings. If it contains “rft” a random field theory multiple comparisons correction will be run. If it contains “fdr” a false discovery rate multiple comparisons correction will be run. Both may be provided. By default None.
    thetalim : float, optional
        Lower limit on variance coefficients in standard deviations, by
        default 0.01.
    drlim : float, optional
        Step of ratio of variance coefficients in standard deviations, by
        default 0.1.
    two_tailed : bool, optional
        Determines whether to return two-tailed or one-tailed p-values. Note
        that multivariate analyses can only be two-tailed, by default True.
    cluster_threshold : float, optional
        P-value threshold or statistic threshold for defining clusters in
        random field theory, by default 0.001.
    p: float
        A float numeric value specifying the p-value to threshold the results (Default is 0.05)
    smooth : float, optional
        The full-maximum half-width (FMHW) value that will be applied for smoothing 
        of the surface-based measure along the surface. Default is None.
    sub_list: str, Sequence
        An optional list of subject IDs to determine whose surface is to be included in the
        analysis. Surfaces whose path does not contain a subject ID from the list will be 
        ignored. If not specified, all surfaces found will be included.
    
    Returns
    -------
    slm_model : 
        BrainStat SLM object containing the statistical outputs.
    """
    
    #collate surface matrices from the outputdir at that metric and specific metric
    mesh_list = list(Path(inputdir).rglob(f'*{roilabel}_{metric}.vtk'))
    
    if len(mesh_list) <= 0:
        raise FileNotFoundError(f"No surface file for {roilabel} {metric} found.")
    
    surf_data=[]
    for mesh in mesh_list:
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(mesh)
        reader.Update()
        surf=reader.GetOutput()
        _ = surf.GetPointData().SetActiveScalars(metric)
        scalar = numpy_support.vtk_to_numpy(surf.GetPointData().GetArray(metric))
        surf_data.append(scalar)
    
    surf_data=np.vstack(surf_data)
    
    #template surface can simply be 1st mesh (as all surfaces are template-based)
    surf_template=mesh_io.read_surface(str(mesh_list[0]))
    
    #exclude non applicable subjects
    if sub_list is not None:
        keep_idx = [i for i, m in enumerate(mesh_list) if any(sub in str(m) for sub in np.asarray(sub_list))]
        surf_data = surf_data[keep_idx, :]
    
    #apply smoothing if applicable
    if smooth:
        surf_data=mesh_smooth(surf_data.copy(), surf_template, 5)
    
    #apply model on surf data
    slm_model = SLM(
        model=model,
        contrast=contrast,
        surf=surf_template,
        mask=mask,
        correction=correction,
        thetalim=thetalim,
        drlim=drlim,
        two_tailed=two_tailed,
        cluster_threshold=cluster_threshold,
        data_dir=None
    )
    slm_model.fit(surf_data)
    
    return slm_model
    
#####################################################################################
##########Statistical analyses using random field theory cluster correction##########

def slm_plot(
    slm, 
    stat: str, 
    threshold: Optional[float] = None, 
    cmap: str = 'RdBu_r', 
    clim: Optional[Tuple[float, float]] = None
    ):
    """
    Plot SLM result maps on their respective surface.
    
    Parameters
    ----------
    slm: SLM object
        BrainStat SLM object containing the statistical outputs.
    stat: str
        Map to plot from the SLM object - 't' for t-statistics map, 'p_fdr' for the false discovery rate-corrected p-value map, 'p_rft' for the random field theory cluster corrected p-values, or 'clusters' for the clusters.
    threshold: float, optional
        Thresholding value in which vertices are to be plotted (below the threshold for p-value
        and clusters maps, or above the set threshold for t-statistics map).
    cmap: str
        Name of the color map to be assigned to the background volume, as listed in matplotlib's colormaps. Default is "RdBu_r".
    clim: Tuple, optional
        Sequence of float stating the minimum and maximum value of the color bar. Default is 
        minimum and maximum value for the t-statitics maps, 0 to 0.05 for the p-value maps. It
        is not applied for the clusters maps as colour values follow cluster IDs.
    """
    
    #load mesh
    mesh = pv.wrap(slm.surf.VTKObject)  
    
    #add scalar values from SLM's maps:
    #t statistics
    if stat == 't':
        scalars = slm.t.squeeze()
        cmap = cmap
        if clim is None:
            clim = [-max(abs(scalars)), max(abs(scalars))]
        title = 't-statistics'
    #p-vals (FDR)
    elif stat == 'p_fdr':
        scalars = slm.Q 
        cmap = cmap
        if clim is None:
            if max(scalars) > 0.05:
                clim = [0, 0.05]
            else:
                clim = [0, max(scalars)]
        title = 'q-value (FDR)'
    #Cluster corrected p-values
    elif stat == 'p_rft':
        scalars = slm.P['pval']['C']
        cmap = cmap
        if clim is None:
            if max(scalars) > 0.05:
                clim = [0, 0.05]
            else:
                clim = [0, max(scalars)]
        title = 'p-value (RFT)'
    
    elif stat == 'clusters':
        if threshold is None:
            threshold=0.05
        
        #get significant cluster IDs for both pos and neg contrasts
        sig_pos_ids = slm.P['clus'][0].loc[slm.P['clus'][0]['P'] < threshold, 'clusid'].values if not slm.P['clus'][0].empty else []
        sig_neg_ids = slm.P['clus'][1].loc[slm.P['clus'][1]['P'] < threshold, 'clusid'].values if not slm.P['clus'][1].empty else []
        # default to zeros if no negative or positive cluster at all
        pos_map = slm.P['clusid'][0].squeeze().copy() if slm.P['clusid'][0] is not None else np.zeros(slm.t.squeeze().shape)
        neg_map = slm.P['clusid'][1].squeeze().copy() if slm.P['clusid'][1] is not None else np.zeros(slm.t.squeeze().shape)
        
        #zero out non-significant clusters
        pos_map[~np.isin(pos_map, sig_pos_ids)] = 0
        neg_map[~np.isin(neg_map, sig_neg_ids)] = 0
    
        #combine IDs
        scalars = pos_map.copy()
        scalars[neg_map != 0] = -neg_map[neg_map != 0]
        scalars[scalars == 0] = np.nan
        
        cmap = cmap
        clim = None
        title = 'Significant clusters'
    else:
        raise ValueError(f"'{stat}' is not applicable. Choose from: 't', 'p', 'p_fdr', 'p_rft', 'clusters'")
    
    #threshold
    if threshold is not None:
        if stat == 't':
            mask = np.abs(scalars) < threshold
        else:  # p-values: mask above threshold
            mask = scalars > threshold
        scalars = scalars.copy().astype(float)
        scalars[mask] = np.nan
    
    #PyVista plotter
    mesh[title] = scalars
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars=title,
        cmap=cmap,
        clim=clim,
        nan_color='lightgrey', 
        show_scalar_bar=True,
        smooth_shading=True,
    )
    plotter.add_text(title, font_size=12)
    plotter.show()