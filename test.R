library(keras)
library(tensorflow)
library(EBImage)
library(dplyr)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

##### segmentation #####
# segmentation function 
segmentation_model <- function(){
  model_file = 'deeplabv3_pascal_trainval/frozen_inference_graph.pb'
  graph = tf$Graph()
  with(tf$io$gfile$GFile(model_file, "rb") %as% fid, {graph_def = tf$compat$v1$GraphDef$FromString(fid$read())}) #tf$GraphDef$FromString
  with(graph$as_default(), {tf$import_graph_def(graph_def, name='')})
  sess = tf$compat$v1$Session(graph = graph) #tf$Session
  return(sess)
  }
image_deal <- function(image_input, width, height){
  img <- image_input/255
  img <- Image(img, colormode = 'Color') %>%
    flip() %>%
    rotate(90) %>% 
    resize(width, height)
  return(img)}
segmentation_fn <- function(image_input, sess_input){
  # read image and resize
  img_original <- keras::image_load(image_input)
  width <- unlist(img_original$size[1])
  height <- unlist(img_original$size[2])
  resize_ratio <- 1.0 * 513 / max(width, height)
  target_width <- floor(resize_ratio * width)
  target_height <- floor(resize_ratio * height)
  img <- keras::image_load(image_input, target_size = c(target_height, target_width)) %>%
    image_to_array()
  img_reshape <- array_reshape(img, c(1, dim(img)))
  
  # segmentation
  batch_seg_map <- sess_input$run(fetches = 'SemanticPredictions:0', 
                                  feed_dict = dict('ImageTensor:0' = img_reshape))
  
  # result_seqmentation
  img_seq_array <- batch_seg_map[1,,]
  img_seq <- img
  img_seq[img_seq_array==0] <- 255
  img_seq <- image_deal(img_seq, width, height)

  # result_mask
  img_mask <- img
  img_mask[img_seq_array==0] <- 0
  img_mask[img_seq_array!=0] <- 255
  img_mask <- image_deal(img_mask, width, height)
  
  # original
  img_original <- img_original %>% image_to_array()
    
  return(list(mask = img_mask, segmentation = img_seq, original = img_original))}
# segmentation result
model_input <- segmentation_model()
segmentation_result <- segmentation_fn(image_input = '06.jpg', 
                                       sess_input = model_input)
plot(segmentation_result[[1]])
plot(segmentation_result[[2]])

##### trimap #####
library(reticulate)
python_route <- "C:/Users/pc053/Anaconda3" 
use_python(python_route,required = TRUE)
cv2 <- import("cv2", convert = FALSE)
# trimap function 
trimap_fn <- function(mask_input, size, erosion){
  # parameter
  if(length(dim(mask_input))>2){img <- mask_input@.Data[,,1]*255}
  row <- dim(img)[2]
  col <- dim(img)[1]
  pixels <- 2*size + 1       
  kernel <- array(1:1, c(pixels, pixels))
  erosion_kernel <- array(1:1, c(3, 3))
  # erode
  img <- cv2$erode(img, erosion_kernel, erosion) %>%
    py_to_r()
  img[img>0] <- 255
  img[img<=0] <- 0
  # dilation
  dilation <- cv2$dilate(img, kernel, 1) %>%
    py_to_r()
  dilation[dilation==255] <- 128
  remake <- dilation
  remake[remake!=128]<- 0
  remake[img>128] <- 200
  remake[img<=128] <- dilation[img<=128]
  remake[remake<128] <- 0
  remake[remake>200] <- 0
  remake[remake==200] <- 255
  # final
  img <- imageData(remake/255) %>%
    Image()
  return(img)}
# trimap result 
trimap_result <- trimap_fn(mask_input = segmentation_result[[1]], 
                           size = 20, 
                           erosion = 10)
plot(trimap_result)



library(rTorch)
matting_model <- function(){
  resume <- "stage1_sad_54.4.pth"
  ckpt <- torch$load(resume, map_location=torch$device('cpu'))
  model <- torch$nn$Module()
  model$conv1_1 <- torch$nn$Conv2d(as.integer(4), as.integer(64), kernel_size = as.integer(3), stride = 1, padding = 1, bias = TRUE)
  model$conv1_2 <- torch$nn$Conv2d(as.integer(64), as.integer(64), kernel_size = as.integer(3), stride = 1, padding = 1, bias = TRUE)
  model$conv2_1 <- torch$nn$Conv2d(as.integer(64), as.integer(128), kernel_size = as.integer(3), padding = 1, bias = TRUE)
  model$conv2_2 <- torch$nn$Conv2d(as.integer(128), as.integer(128), kernel_size = as.integer(3), padding = 1, bias = TRUE)
  model$conv3_1 <- torch$nn$Conv2d(as.integer(128), as.integer(256), kernel_size=  as.integer(3), padding = 1, bias = TRUE)
  model$conv3_2 <- torch$nn$Conv2d(as.integer(256), as.integer(256), kernel_size = as.integer(3), padding = 1, bias = TRUE)
  model$conv3_3 <- torch$nn$Conv2d(as.integer(256), as.integer(256), kernel_size = as.integer(3), padding = 1, bias = TRUE)
  model$conv4_1 <- torch$nn$Conv2d(as.integer(256), as.integer(512), kernel_size = as.integer(3), padding = 1, bias = TRUE)
  model$conv4_2 <- torch$nn$Conv2d(as.integer(512), as.integer(512), kernel_size = as.integer(3), padding = 1, bias = TRUE)
  model$conv4_3 <- torch$nn$Conv2d(as.integer(512), as.integer(512), kernel_size = as.integer(3), padding = 1, bias = TRUE)
  model$conv5_1 <- torch$nn$Conv2d(as.integer(512), as.integer(512), kernel_size = as.integer(3), padding = 1, bias = TRUE)
  model$conv5_2 <- torch$nn$Conv2d(as.integer(512), as.integer(512), kernel_size = as.integer(3), padding = 1, bias = TRUE)
  model$conv5_3 <- torch$nn$Conv2d(as.integer(512), as.integer(512), kernel_size = as.integer(3), padding = 1, bias = TRUE)
  model$conv6_1 <- torch$nn$Conv2d(as.integer(512), as.integer(512), kernel_size = as.integer(3), padding = 1, bias = TRUE)
  model$deconv6_1 <- torch$nn$Conv2d(as.integer(512), as.integer(512), kernel_size = as.integer(1), bias = TRUE)
  model$deconv5_1 <- torch$nn$Conv2d(as.integer(512), as.integer(512), kernel_size = as.integer(5), padding = 2, bias = TRUE)
  model$deconv4_1 <- torch$nn$Conv2d(as.integer(512), as.integer(256), kernel_size = as.integer(5), padding = 2, bias = TRUE)
  model$deconv3_1 <- torch$nn$Conv2d(as.integer(256), as.integer(128), kernel_size = as.integer(5), padding = 2, bias = TRUE)
  model$deconv2_1 <- torch$nn$Conv2d(as.integer(128), as.integer(64), kernel_size = as.integer(5), padding = 2, bias = TRUE)
  model$deconv1_1 <- torch$nn$Conv2d(as.integer(64), as.integer(64), kernel_size = as.integer(5), padding = 2, bias = TRUE)
  model$deconv1 <- torch$nn$Conv2d(as.integer(64), as.integer(1), kernel_size = as.integer(5), padding = 2, bias = TRUE)
  model$load_state_dict <- model$load_state_dict(ckpt$state_dict, strict=TRUE)
  return(model)}
model <- matting_model()

img_original <- image_load('06.jpg') %>%
  image_to_array()
plot(trimap_result)


  rotate(-90) %>% 
  image_to_array()

original_im <- segmentation_result[[3]]
trimap_im <- trimap_result %>%
  rotate(-90)
if(length(dim(trimap_result))>2){trimap_im <- trimap_im[,,1]}

h <- dim(original_im)[1]
w <- dim(original_im)[2]
c <- dim(original_im)[3]
max_size <- 1600
new_h <- min(max_size, h - (h%%32))
new_w <- min(max_size, w - (w%%32))
scale_img <- cv2$resize(original_im, tuple(as.integer(new_w), as.integer(new_h)), interpolation = cv2$INTER_LINEAR)
scale_img <- scale_img$round()$astype(np$uint8)
scale_trimap <- cv2$resize(trimap_im * 255, tuple(as.integer(new_w), as.integer(new_h)), interpolation = cv2$INTER_LINEAR)
scale_trimap <- scale_trimap$round()$astype(np$uint8)

trimap_result
normalize <- torchvision$transforms$Compose(c(torchvision$transforms$ToTensor(),
                                              torchvision$transforms$Normalize(mean = c(0.485, 0.456, 0.406), 
                                                                               std = c(0.229, 0.224, 0.225))))
scale_img_rgb <- cv2$cvtColor(scale_img, cv2$COLOR_BGR2RGB)
tensor_img <- normalize(scale_img_rgb)
tensor_img <- tensor_img$unsqueeze(as.integer(0))

x = cv2$Sobel(scale_img, cv2$CV_16S, as.integer(1), as.integer(0))
y = cv2$Sobel(scale_img, cv2$CV_16S, as.integer(0), as.integer(1))
absX = cv2$convertScaleAbs(x)
absY = cv2$convertScaleAbs(y)
grad = cv2$addWeighted(absX, 0.5, absY, 0.5, 0)
scale_grad = cv2$cvtColor(grad, cv2$COLOR_BGR2GRAY)
scale_trimap = scale_trimap$astype('float32')

tensor_trimap = torch$from_numpy(scale_trimap[np.newaxis, np.newaxis, :, :])
tensor_grad = torch$from_numpy(scale_grad[np.newaxis, np.newaxis, :, :])




