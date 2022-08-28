# Core-Shell Clustering Algorithm

* Core-Shell Clustering algorithm is an unsupervised clustering algorithm for the analysis of spatiotemporal variations of SST imagery data;
* This algorithm uses .mat files as the SST grids;
* The final results produced by this algorithm are called core-shell clusters;
* In this repository there is the fully proposed pipeline, beginning from the original SST data, the preprocessing pipeline, the S-STSEC algorithm implementation, the Core-Shell clustering algorithm implementation and several auxiliary functions used to plot and get SST images.

# Software setup

* If not installed, please install julia in your machine, it is used for the first stage of the preprocessing phase: https://julialang.org/downloads/;
* If not installed, please install the julia wrapper for the GMT tools: https://github.com/GenericMappingTools/GMT.jl.

# Run the software

* Download the latest release and unzip the downloaded folder;
* Use an IDE to run the software. This is due to the need to use several paths as inputs to the program;
* To run the main algorithm, open the algorithmScript.py file.

# Files explanation

* algorithmScript.py: Main script of the Core-Shell clustering algorithm;
* anomalousPattern.py: Implementation of the IAP algorithm;
* coastlineFinder.py: Gets coastline pixels and near the coast pixels for the STSEC algorithm;
* experimentUtils.py: Auxiliary functions;
* fileUtils: Auxiliary functions regarding file operations;
* matrixUtils: Functions regarding matrices;
* moovingAvgs: Functions regarding the moving averages filter;
* plotters: Functions to plot images;
* processingTools: More auxiliary functions;
* regionGrowing: Functions used for the SRG used in the STSEC base algorithm;
* sstsec: Implementation of the SSTSEC algorithm;
* stsec: Implementation of the basic STSEC algorithm;
* temperatureProcessingV3: Current version of the preprocessor pipeline of the Portuguese coast;
* upwellingClumpsBuilder: Functions used to use the IAP algorithm for the building of the Upwelling Ranges.

# MAT files

* A sample of the full dataset is present in the folder "PortugalImagesSample";
* Each year has its information in its own folder;
* Only the original images are in the format .png;
* The explanation for each folder will be given with the assistance of the image "full_pipeline.png" that shows the developed pipeline, in order to give context to each folders' content;
* Inside a "mat" folder are the resulting computation results from each phase of the preprocessing pipeline:
  * preprocessing_phase_1: SST grids that suffered the removal of the North-South gradient using the GMT tools (Point 1 in the pipeline);
  * preprocessing_phase_2: SST grids that suffered the previous transformation and where the moving average filter was applied (Point 2 in the pipeline);
  * sst_instants: SST grids completely preprocessed which are called SST instants (Point 3 in the pipeline);
  * sst_instants_no_preprocessing: SST grids that suffered all the preprocessing stages but the first one, in order to be able to assess the SST instants with the original temperature ranges.
  * sst_instants/segmentations: SSTSEC segmentation results (Point 4 in the pipeline).

# Experimental Study

* A sample of the experimental study results can be assessed in the folder "experimentsSample/Portugal";
* Inside the root folder there are 2 other folders regarding the experimental studies for the years 2007 and 2012;
* For each year, there are several subfolders:
  * averages_original: 
  * comparisons: SST images referent to the .mat files in the folders "sst_instants_no_preprocessing";
  * core_shell_algorithm: results of the core-shell clustering algorithm (core-shell clusters and clustering parameter G evolutions, point 5 in the pipeline);
  * instants_segmentations: SST images referent to the .mat files in the folders "sst_instants/segmentations";
  * preprocessing_phase1: SST images referent to the .mat files in the folders "preprocessing_phase_1";
  * preprocessing_phase2: SST images referent to the .mat files in the folders "preprocessing_phase_2";
  * sst_instants: SST images referent to the .mat files in the folders "sst_instants".
