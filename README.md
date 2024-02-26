# learnedFastPAT
A Fourier domain reconstruction approach for planar photoacoustic tomography (PAT) measurements augmented with learned components.

Two data-trained networks are embedded in the standard Fourier reconstruction: one in image k-space (model correction), the second in the image space (post processing) itself, which are trained jointly. 

A model-correction term is utilised to compensate for wrong speed of sound and used together with a post-processing network to improve image quality and compensate for limited-view artefacts.
