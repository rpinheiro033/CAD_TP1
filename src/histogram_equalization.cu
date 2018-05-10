// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define THREADS_NUMBER 512


    void histogram_equalization(wbImage_t& inputImage, wbImage_t& outputImage) {

    //TODO

}


__global__ void vectorAdd(float* inputImage, unsigned char* ucharImage, int n) {
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

__global__ void histogram_comput(int* histogramAux, unsigned char* greyImage, int n) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index < n) {
    int c = (int) greyImage[index];
    atomicAdd(&histogramAux[c], 1);
  }
}

int main(int argc, char **argv) {
  /* parse the input arguments */
  wbImage_t inputImage = wbImport(argv[1]);

  //wbImage_t *inputImage_;

  //cudaMalloc((void**)&inputImage_, size);

  int imageWidth    = wbImage_getWidth(inputImage);
  int imageHeight   = wbImage_getHeight(inputImage);
  int imageChannels = wbImage_getChannels(inputImage);

  int valueHistogram = imageWidth * imageHeight;

  float * imageData = wbImage_getData(inputImage);
  float * imageData_;
  int size = sizeof(float*);
  cudaMalloc((void**)&imageData_, size);

  wbImage_t outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  // histogram_equalization(inputImage, outputImage);

  /* Cast the image from float to unsigned char */
  unsigned char *ucharImage_;
  unsigned char ucharImage;
  int size_char_image = sizeof(unsigned char&);

  unsigned char *ucharImageFinal_;
  unsigned char ucharImageFinal;

  cudaMalloc((void**)&ucharImage_, size_char_image);
  cudaMalloc((void**)&ucharImageFinal_, size_char_image);

  cudaMemcpy(imageData_, &imageData, size, cudaMemcpyHostToDevice);
  cudaMemcpy(ucharImage_, &ucharImage, size_char_image, cudaMemcpyHostToDevice);

  int max_ = imageWidth * imageHeight * imageChannels;
  vectorAdd<<<max_/THREADS_NUMBER,THREADS_NUMBER>>>(imageData_, &ucharImage, max_);
  cudaDeviceSynchronize();
  cudaMemcpy(&ucharImageFinal, ucharImageFinal_, size_char_image, cudaMemcpyDeviceToHost);

 /* Step 2 */
 //wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
 unsigned char *grayImage_;
 unsigned char grayImage;

 int size_grayImage = sizeof(unsigned char&);

 unsigned char *grayImageFinal_;
 unsigned char grayImageFinal;

 cudaMalloc((void**)&grayImage_, size_grayImage);
 cudaMalloc((void**)&grayImageFinal_, size_grayImage);

  cudaMemcpy(grayImage_, &grayImage, size_grayImage, cudaMemcpyHostToDevice);

  dim3 dimBlock(THREADS_NUMBER, THREADS_NUMBER);
  dim3 dimGrid((int)ceil(valueHistogram/dimBlock.x), (int)ceil(valueHistogram/dimBlock.y));
  greyScaleTransf<<<dimGrid, dimBlock>>>(&ucharImage, &grayImage, valueHistogram);
  cudaDeviceSynchronize();
  cudaMemcpy(grayImageFinal_, &grayImageFinal, size_grayImage, cudaMemcpyDeviceToHost);
 //wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  int * histoLength;
  int length = sizeof(int&) * valueHistogram;

  cudaMalloc((void**) &histoLength, length);
  cudaMemset((void**) &histoLength, 0, length);

  for(int i = 0; i < sizeof(histoLength); i++) {
    printf("Positions of histogram: %d\n", histoLength[i]);
  }


  int * histoLengthToPrint = new int[valueHistogram];
  histogram_comput<<<valueHistogram/THREADS_NUMBER, THREADS_NUMBER>>>(histoLength, grayImageFinal_, valueHistogram);
  cudaDeviceSynchronize();
  cudaMemcpy(histoLength, &valueHistogram, length, cudaMemcpyDeviceToHost);

  cudaFree(grayImageFinal_);
  cudaFree(grayImage_);
  cudaFree(ucharImageFinal_);
  cudaFree(ucharImage_);
  cudaFree(imageData_);
  cudaFree(histoLength);

  printf("imageWidth = %d\n", imageWidth);
  printf("MAX = %d\n", max_);

  return 0;
}
