# FSIM
feature similarity index implementation on PyTorch



This measure of image quality is originally implemented in MATLAB on CPU. The drawback of CPU computing appears when measuring the similarity of high-resolution images e.g. 4K images.

The project should be found [here](<https://www4.comp.polyu.edu.hk/~cslzhang/IQA/FSIM/FSIM.htm>)



Please cite their paper if you like:

Lin Zhang, Lei Zhang, X. Mou and D. Zhang, FSIM: A Feature Similarity Index for Image Quality Assessment, _IEEE Trans. Image Processing_, vol. 20, no. 8, pp. 2378-2386, 2011.



This repo aims to re-implement the original matlab program into PyTorch, which can cope with GPU. 



This work is not completed yet because `phasecong2` is not yet understood and I would deal with that part as long as I have spare time.



With external calling MATLAB, computing FSIM of 2K videos can run _20x times faster_ with  1080ti