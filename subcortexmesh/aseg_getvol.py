#####################################################################################
##################getting subcortical volumes from aseg##############################

import os
import ants
from typing import Optional, Union
from pathlib import Path
from subcortexmesh import template_data_fetch

def aseg_getvol(
    inputdir: Union[str, Path],
    outputdir: Union[str, Path],
    toolboxdata: Optional[Union[str, Path]] = None,
    overwrite=True,
    silent=False,
    ):
    """Extracting and coregistering subcortical volumes from a cohort's aseg volumes
    
    This function reads through subjects of a FreeSurfer output preprocessing directory
    and identifies aseg.mgz volumes. For each subject, it coregisters the aseg volume to 
    the fsaverage space (from FREESURFER_HOME/subjects/fsaverage/mri/T1.mgz), and then 
    extracts each subcortical region-of-interest separately, in subject space.
    
    Parameters
    ----------
    inputdir : str, Path
        The path to the subjects directory outputted via the FreeSurfer recon-all preprocessing pipeline (SUBJECTS_DIR).
    outputdir : str, Path
        The path where subcortical volumes will be saved (creates the "sub_volumes" 
        directory).
    toolboxdata : str, Path, optional
        The path of the "subcortexmesh_data" package data directory. The  default path 
        is assumed to be the user's home directory (pathlib's Path.home()). Users will 
        be prompted to download it if not found.
    overwrite : bool
        Whether files are to be overwritten or skipped if already in outputdir. 
        Default is True.
    silent : bool
        Whether messages about the process are to be silenced. Default is False.
    """
    
    #template data is needed
    toolboxdata=template_data_fetch(datapath=toolboxdata, template = 'fsaverage')
    
    #other paths checks
    if not os.path.exists(f"{inputdir}"):
        raise FileNotFoundError("Subjects directory not found. Please verify the path provided as inputdir.")
    
    #FreeSurfer loaded check
    if os.system('mri_segstats > /dev/null') != 256:
        raise OSError("FreeSurfer seems to be unavailable. Please make sure it is loaded in the environment and the FREESURFER_HOME variable set.")
    
    if not silent: 
        print(f"Writing the {outputdir}/sub_volumes directory...")
    os.makedirs(os.path.join(f"{outputdir}/sub_volumes"), exist_ok=True)
    
    if not os.path.exists(f"{outputdir}/sub_volumes/aseg_labels.txt" or overwrite):
        if not silent: 
            print("Extracting the aseg atlas ROI labels...")
        #aseg.mgz was 7.4.1's $FREESURFER_HOME/subjects/fsaverage/mri/aseg.mgz
        _ = os.system(f"mri_segstats --seg {toolboxdata}/template_data/fsaverage/aseg.mgz --sum {outputdir}/sub_volumes/aseg_labels.txt  >/dev/null")
    
    #get aseg atlas's ROI indices from fsaverage's aseg.mgz into an array
    #indices are in the second column (lines with no #) printed
    aseg_indices = sorted({int(line.split()[1]) for line in open(f"{outputdir}/sub_volumes/aseg_labels.txt") if not line.startswith("#")})
    #list subs
    sub_list=[f for f in next(os.walk(f"{inputdir}"))[1] if f.startswith("sub-")]
    _ = sub_list.sort()
    
    subindex=0
    for subid in sub_list: 
        subindex=subindex+1
        if not silent: 
            print(f"Processing {subid} ... [{subindex}/{len(sub_list)}]")
        
        #####################################################################################
        ##################Coregistration of subject's aseg to fsaverage######################
        
        if not os.path.exists(f"{inputdir}/{subid}/mri/aseg.mgz"):
            if not silent: 
                print(f"{subid} has no aseg.mgz file in its mri/ directory.")
        
        #will make a inputdirectory called "ants" in the freesurfer output for the warped volumes for tidiness
        antsdir=f"{outputdir}/sub_volumes/{subid}/ants_coreg"
        os.makedirs(os.path.join(f"{antsdir}"), exist_ok=True)
        #check if coregistration done already
        if not os.path.exists(f"{antsdir}/aseg_fsaverage_rigid_coreg.nii.gz") or overwrite:
            #resample subject's own aseg to fsaverage space for alignment
            if not silent: 
                print(f"Coregistering {subid}'s aseg to fsaverage before volumes are extracted...")
            #mgz are converted to nii.gz as ANTS uses those
            _ = os.system(f"mri_convert {inputdir}/{subid}/mri/aseg.mgz {antsdir}/aseg.nii.gz >/dev/null")
            _ = os.system(f"mri_convert {inputdir}/{subid}/mri/T1.mgz {antsdir}/T1.nii.gz >/dev/null")
            templatevol=f"{toolboxdata}/template_data/fsaverage/T1.mgz" #imported. It is 7.4.1's $FREESURFER_HOME/subjects/fsaverage/mri/T1.mgz
            #one can also use MNI, as fsaverage is based on it
            #templatevol=f"/usr/local/fsl/data/standard/MNI152_T1_1mm.nii.gz"
            
            #register subject's T1 to faverage, and use that warp to register aseg
            #apply rigid (-t r) transformation to preserve actual size of ROI
            fixedT1=ants.image_read(filename=templatevol, dimension=3)
            movingT1=ants.image_read(filename=f"{antsdir}/T1.nii.gz", dimension=3)
            mytx=ants.registration(fixed=fixedT1, moving=movingT1, type_of_transform="Rigid", outprefix=f"{antsdir}/T1_to_template_rigid")
            movingaseg=ants.image_read(filename=f"{antsdir}/aseg.nii.gz", dimension=3)
            warpedimg=ants.apply_transforms(fixed=fixedT1, moving=movingaseg, transformlist=mytx['fwdtransforms'], interpolator="multiLabel")
            _ = ants.image_write(warpedimg, f"{antsdir}/aseg_fsaverage_rigid_coreg.nii.gz")
            
            #to visualise:
            #os.system(f"freeview -v {templatevol} {antsdir}/aseg_fsaverage_rigid_coreg.nii.gz:colormap=LUT")
             
            #error if registered volumes did not output successfully
            if os.path.exists(f"{antsdir}/aseg_fsaverage_rigid_coreg.nii.gz"):
                if not silent: 
                    print(f"=> Done: coregistered volume stored to {antsdir}")
                asegvol=f"{antsdir}/aseg_fsaverage_rigid_coreg.nii.gz"
            else:
                if not silent: 
                    print("Coregistration of aseg for $subid failed. Subcortical volumes will not be outputted.")
        else:
            asegvol=f"{antsdir}/aseg_fsaverage_rigid_coreg.nii.gz"
        
        #####################################################################################
        ##################Saving each ROI as separate volumes################################
        
        #if coregistration worked, separating each ROI from the coregistered aseg
        if os.path.exists(asegvol):
            
            if not silent: 
                print(f"Splitting subcortical volumes to {outputdir}/sub_volumes/{subid}/...")
            
            #get volumes of each aseg subcortical region separately
            #read through FreeSurferColorLUT.txt, which contains aseg ROI labels, and extract their volume masks independently
            lookuptable=f"{toolboxdata}/template_data/fsaverage/FreeSurferColorLUT.txt" #7.4.1's "$FREESURFER_HOME/FreeSurferColorLUT.txt"
            #indices for aseg ROIs we don't care about (manually checked in the LUT what they referred to) 
            skip_indices={0, 2, 3, 4, 5, 6, 7, 14, 15, 24, 30, 31, 41, 42, 43, 44, 45, 46, 62, 63, 77, 85, 251, 252, 253, 254, 255}
            
            with open(lookuptable, "r", encoding="utf-8") as f:
              for line in f:
                  line = line.rstrip("\n")
                  # Skip empty lines or comments
                  if not line or line.startswith("#"): 
                    continue
                  #get index (first column)
                  parts = line.split()
                  index = int(parts[0]) 
                  #stop once index reaches 251 (subsequent ROI labels ignored)
                  if index >= 251: 
                    break
                  #if index not among indices found in aseg.mgz,skip 
                  if index not in aseg_indices: 
                    continue
                  if index in skip_indices: 
                    continue
                  #get ROI name 
                  label = parts[1] if len(parts) > 1 else None
                    
                  #Check if volume was extracted already
                  if not os.path.exists(f"{outputdir}/sub_volumes/{subid}/{label}.nii.gz") or overwrite:
                      if not silent: 
                        print(f"=> Extracting {label} ...")
                      
                      #Get ROI's mask and multiply with original labels to preserve labelling
                      _ = os.system(f"mri_binarize --i {asegvol} --match {index} --o {outputdir}/sub_volumes/{subid}/{label}.mgz > /dev/null")
                      _ = os.system(f"mri_mask {asegvol} {outputdir}/sub_volumes/{subid}/{label}.mgz {outputdir}/sub_volumes/{subid}/{label}.mgz > /dev/null")
                      #convert to .nii for later use
                      _ = os.system(f"mri_convert {outputdir}/sub_volumes/{subid}/{label}.mgz {outputdir}/sub_volumes/{subid}/{label}.nii.gz > /dev/null")
                      os.remove(f"{outputdir}/sub_volumes/{subid}/{label}.mgz") #no longer need the mgz
                  else:
                    if not silent: 
                        print(f"=> {label} already extracted")
