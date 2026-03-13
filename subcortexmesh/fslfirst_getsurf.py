#####################################################################################
######Collects meshes from a FSL-first output to make them SCM-compatible############

import os
import re
import shutil
from typing import Union
from pathlib import Path

def fslfirst_getsurf(
    inputdir: Union[str, Path],
    outputdir: Union[str, Path],
    overwrite=True,
    silent=False
    ):
    """Extracting surface meshes from a FSL FIRST output directory
    
    Functions such as mesh_metrics(), merge_all() require access to sub_surfaces/ directory,
    which contains all of a subject cohort's subcortical mesh files. This function creates 
    it automatically for a FSL first segmentation directory (meshes created with run_first_all). 
    /!\ This function follows the sub-*** convention for participant IDs and will parse
    folder and file names accordingly. It will disregard any other type of labelling and
    fail to identify subjects that do not have a "sub-ID" explicitly named somewhere in their
    folder name.
    
    Parameters
    ----------
    inputdir : str, Path
        The output path where the FSL FIRST has outputted subject .vtk meshes after using 
        the "run_first_all" command. Typically, the inputdir should be what was indicated
        with the "-o" flag when running FSL.
    outputdir : str, Path
        The path where subcortical meshes will be saved (creates the "sub_surfaces" 
        directory inside the path).
    overwrite : bool
        Whether files are to be overwritten or skipped if already in outputdir. 
        Default is True.
    silent : bool
        Whether messages about the process are to be silenced. Default is False.
    """
    
    #other paths checks
    if not os.path.exists(f"{inputdir}"):
        raise FileNotFoundError("Subjects directory not found. Please verify the path provided as inputdir.")
    
    if not silent: 
        print(f"Writing the {outputdir}/sub_surfaces directory...")
    
    os.makedirs(os.path.join(f"{outputdir}/sub_surfaces"), exist_ok=True)
    
    #list subs (only counted as sub folders with the right pattern)
    pattern = r'(sub-[^_]+)'
    sub_list = [f for f in next(os.walk(inputdir))[1] if re.search(pattern, f)]
    _ = sub_list.sort()
    
    subindex=0
    for subid in sub_list: 
        
        #subject id reformatted and limited to sub-xxx for SCM
        newsubid=re.search(pattern, subid).group(1)      
        os.makedirs(os.path.join(f"{outputdir}/sub_surfaces/{newsubid}"), exist_ok=True)
                        
        subindex=subindex+1
        
        if not silent: 
            print(f"Copying {subid} meshes... [{subindex}/{len(sub_list)}]")
          
        mesh_list=[f for f in os.listdir(f"{inputdir}/{subid}") if f.endswith(".vtk")]
        
        if len(mesh_list) > 0:
            for meshfile in mesh_list:
                
                #################################################################################
                ###########Renames files to standardize names and structure in SCM###############
                
                #rename labels
                region=re.search(r'-(?:(L|R)_)?([^-_]+)_first', meshfile).group(2)
                if region == 'Accu': 
                    region='accumbens-area'
                if region == 'Amyg': 
                    region='amygdala'
                if region == 'Caud': 
                    region='caudate'
                if region == 'Hipp': 
                    region='hippocampus'
                if region == 'Pall': 
                    region='pallidum'
                if region == 'Puta': 
                    region='putamen'
                if region == 'Thal': 
                    region='thalamus'
                if region == 'BrStem': 
                    region='brain-stem'
                #get hemisphere 
                hemi=re.search(r'-(L|R)?_?([^_]+)_first', meshfile).group(1)
                if hemi=='L':
                    hemi='left-'
                if hemi=='R':
                    hemi='right-'
                if hemi is None:
                    hemi=''
                    
                #Copy
                if not os.path.exists(f"{outputdir}/sub_surfaces/{newsubid}/{hemi}{region}.vtk") or overwrite:
                    shutil.copy2(f"{inputdir}/{subid}/{meshfile}", f"{outputdir}/sub_surfaces/{newsubid}/{hemi}{region}.vtk")
        else:
            if not silent: 
                print(f"{subid} has no .vtk file in its directory.")