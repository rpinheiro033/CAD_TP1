#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define THREADS_PER_BLOCK 512



__global__ void vectorAdd(float* inputImage, unsigned char* ucharImage, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        ucharImage[index] = (unsigned char) (255 * inputImage[index]);

}

__global__ void greyScaleTransf(unsigned char* ucharImage, unsigned char* greyImage, int n) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int index = col + row * n;
  unsigned char r;
  unsigned char g;
  unsigned char b;

  if (col < n && row < n) {
    r = ucharImage[3*index];
    g = ucharImage[3*index + 1];
    b = ucharImage[3*index + 2];
    greyImage[index] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}

int main(int argc, char **argv) {

  /* parse the input arguments */
  wbImage_t inputImage = wbImport(argv[1]);
  
  int imageWidth    = wbImage_getWidth(inputImage);
  int imageHeight   = wbImage_getHeight(inputImage);
  int imageChannels = wbImage_getChannels(inputImage);
  
  //step 1
  float *imageData = wbImage_getData(inputImage);
  float *d_imageData;
  
  unsigned char *ucharImage;
  unsigned char *d_ucharImage;
  
  int heightPerWidth = imageWidth * imageHeight;
  int max = heightPerWidth * imageChannels;
  
  int ucharImageSize = max * sizeof(unsigned char);
  int imageFloatSize = max * sizeof(float);
  
  cudaMalloc((void**)&d_imageData, imageFloatSize);
  cudaMalloc((void**)&d_ucharImage, ucharImageSize);
  
  ucharImage = (unsigned char *)malloc(ucharImageSize);
  
  cudaMemcpy(d_imageData, imageData, imageFloatSize, cudaMemcpyHostToDevice);
  
  vectorAdd<<<max/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_imageData, d_ucharImage, max);
  
  cudaMemcpy(d_ucharImage, ucharImage, ucharImageSize, cudaMemcpyHostToHost);
  
  free(imageData);
  cudaFree(d_imageData);
  
  //step 2
  unsigned char *grayImage;
  unsigned char *d_grayImage;
  
  int grayImageSize = heightPerWidth * sizeof(unsigned char);
  
  cudaMalloc((void**)&d_grayImage, grayImageSize);
  
  grayImage = (unsigned char *)malloc(grayImageSize);
  
  greyScaleTransf<<<heightPerWidth/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_ucharImage, d_grayImage, heightPerWidth)
  
  cudaMemcpy(d_grayImage, grayImage, grayImageSize, cudaMemcpyHostToHost);
  
  
  free(ucharImage);
  free(grayImage);
  
  cudaFree(d_ucharImage);
  cudaFree(d_grayImage);
  
  
  
  
  return 0;
}