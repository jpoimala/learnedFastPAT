# learnedFastPAT
A Fourier domain reconstruction approach for planar photoacoustic tomography (PAT) measurements augmented with learned components.

Two data-trained networks are embedded in the standard Fourier reconstruction: one in image k-space (model correction), the second in the image space (post processing) itself, which are trained jointly. 

A model-correction term is utilised to compensate for wrong speed of sound and used together with a post-processing network to improve image quality and compensate for limited-view artefacts.

Creation of computational grids are described in 'kgrids_pytorch_kspace_3D'. Data loading, reconstruction approaches and network architectures etc. are described in 'nets_pytorch_kspace_3D'.There are two template files 'FFT_PP' and 'FFT_MC_PP' to show how to compute a Fourier reconstruction with post processing and a Fourier reconstruction with model correction and post processing, respectively. 
