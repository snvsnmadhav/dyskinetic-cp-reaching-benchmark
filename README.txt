README

This dataset belongs to the project AVI-DYS, https://researchportal.be/en/project/automated-video-based-assessment-dyskinesia-cerebral-palsy-using-markerless-pose-estimation

Data is analysed retrospectively from the project IDCA, https://researchportal.be/en/project/instrumented-dystonia-and-choreoathetosis-assessment-protocol-idca-upper-limb-movements

The dataset contains following folders and files, described in detail below: 
1. data
2. data_summarized
3. code
  

1. Data
'Data' contains the x,y-coordinates of all keypoints in pixels (Forehead, Chin, Ankle, Hip, Knee, Shoulder, Elbow, Wrist, MCP3, Object=Target/PTA); and the joint angles (R-Shoulder, L-Shoulder, R-Elbow, L-Elbow, R-Wrist, L-Wrist) of all participants.
The first data row are the coordinates of the start point the last row are the coordinates at the point of task achievement (PTA)

Filename= Nr_performing arm_group_repetition
* Case number
Nr=number
* Performing arm (side)
R=Right
L=Left
* Group 
TD=typically developing
DCP=dyskinetic cerebral palsy
* Repetitions reaching task (rep1-rep8)
rep=repetition

2. Data_summarized
'Data_summarized' contains dataset_3DMA and dataset_MMC
3DMA=three-dimensional motion analysis; MMC=Markerless Motion Capture

Both datasets (dataset_3DMA and dataset_MMC) present the summarized joint angle of eight repetitions at the point of task achievement of all participants and all joints, either measured by 3DMA or MMC.

3.Scripts
'Calculation_features.ipynb' contains a script to calculate the described movement features (i.e. convex hull perimeter, convex hull area, overshoot and task excection time) for each participant.
'Plots_validation.ipynb' contains a script to plot the graphs (violon and regression plots) used for validation between 3DMA and MMC and between DCP and TD group. 

The scripts can be opened and runned with Jupyter Notebook using the base (root) Anaconda3 (Anaconda3 Version 2021.11) environment. 
- Install Anaconda3-2021.11 via https://repo.anaconda.com/archive/
- Open Jupyter Notebook 
- Open the scripts in Jupyter Notebook
- Change path number (and case number, side and group if this is of interest)
- Consecutively run all parts of the script


