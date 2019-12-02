# segmentation_rladies_20191202

### source code
1. segmentation
*   https://github.com/tensorflow/models/tree/master/research/deeplab
2. matting
*   https://github.com/huochaitiantang/pytorch-deep-image-matting
3. trimap
*   https://github.com/lnugraha/trimap_generator
*   https://github.com/foamliu/Deep-Image-Matting/files/2844890/RobustMatting_1.45.zip

### step in linux
1. download code
*   git clone https://github.com/jack155861/segmentation_rladies_20191202
2. change directory
*   cd segmentation_share
3. download deeplabv3+ model
*   wget http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz
4. download matting model
*   wget https://github.com/huochaitiantang/pytorch-deep-image-matting/releases/download/v1.4/stage1_sad_54.4.pth

### run with python
1. file name
*   demo.ipynb
2. tensorflow version
*   python3 -m pip install tensorflow==1.14
3. deeplabv3plus(photo_input, website)
*   photo_input (from local or website)
4. result_trimap = trimap(image, size, erosion)
*   image (mask)
*   size & erosion (unknown area in trimap)
5. result_matting = matting_result(pic_input, tri_input)
*   pic_input (original photo)
*   tri_input (trimap from result_trimap function)

### run with R (through reticulate)
1. file name
*   code_20191106.R
2. set python route
3. set working directory
4. import py file
5. segmentation & matting

### run with R (to do list)
1. file name
*   test.R
