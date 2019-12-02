# library packages
library(reticulate)

# set python route
python_route <- "C:/Users/pc053/Anaconda3" 
use_python(python_route,required = TRUE)

# set working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# import py file
matting_result <- import("matting", convert = FALSE)$matting_result
deeplabv3plus <- import("deeplabv3plus", convert = FALSE)$deeplabv3plus
trimap <- import("trimap", convert = FALSE)$trimap
np <- import("numpy", convert = FALSE)

# 去背
segmentation <- deeplabv3plus('06.jpg')
trimap_result <- trimap(image = np$array(segmentation[1]),
                        size = 30,
                        erosion = 10)
matting <- matting_result(pic_input = segmentation[2],
                          tri_input = trimap_result)

img <- EBImage::imageData(py_to_r(np$array(matting))/255)
img <- EBImage::Image(img, colormode = 'Color')
img <- EBImage::flip(img)
img <- EBImage::rotate(img, 90)
plot(img)
