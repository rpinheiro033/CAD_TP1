#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define THREADS_PER_BLOCK 512



__global__ void vectorAdd(float* inputImage, unsigned char* ucharImage, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        ucharImage[index] = (unsigned char) (255 * inputImage[index]);

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
  printf("meu, ok?");
  free(imageData);
  free(ucharImage);

  cudaFree(d_imageData);
  cudaFree(d_ucharImage);





  return 0;
}
