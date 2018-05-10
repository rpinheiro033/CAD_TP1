#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define THREADS_PER_BLOCK 512

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

int conflict_free_offset(int n)
{
	return ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS));
}

 __global__ void prescan(float* cdf, int* histogram, int n)
 {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
		cdf[index] = histogram[index] / n;
 }
 
 __global__ void scan(float *g_odata, float *g_idata, int n)
{
	extern __shared__ float temp[];  // allocated on invocation
	int thid = threadIdx.x;
	int offset = 1;
	
	int ai = thid;
	int bi = thid + (n/2);
	int bankOffsetA = conflict_free_offset(ai);
	int bankOffsetB = conflict_free_offset(bi);
	temp[ai + bankOffsetA] = g_idata[ai];
	temp[bi + bankOffsetB] = g_idata[bi];
	
	for (int d = n>>1; d > 0; d >>= 1)                    // build sum in place up the tree
	{ 
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			ai += conflict_free_offset(ai);
			bi += conflict_free_offset(bi);
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (thid==0)
		temp[n - 1 + conflict_free_offset(n - 1)] = 0;
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)                     
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			ai += conflict_free_offset(ai);
			bi += conflict_free_offset(bi);
			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t; 
		}
	}
	__syncthreads();
	g_odata[ai] = temp[ai + bankOffsetA];
	g_odata[bi] = temp[bi + bankOffsetB];
}

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

__global__ void histogram_comput(int* histogram, unsigned char* greyImage, int n) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index < n) {
    int c = (int) greyImage[index];
    atomicAdd(&histogram[c], 1);
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
  
  cudaMemcpy(d_ucharImage, ucharImage, ucharImageSize, cudaMemcpyDeviceToHost);
  
  free(imageData);
  cudaFree(d_imageData);
  
  //step 2
  unsigned char *grayImage;
  unsigned char *d_grayImage;
  
  int grayImageSize = heightPerWidth * sizeof(unsigned char);
  
  cudaMalloc((void**)&d_grayImage, grayImageSize);
  
  grayImage = (unsigned char *)malloc(grayImageSize);
  
  dim3 dimBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
  dim3 dimGrid((int)ceil(heightPerWidth/dimBlock.x), (int)ceil(heightPerWidth/dimBlock.y));
  
  greyScaleTransf<<<dimGrid, dimBlock>>>(d_ucharImage, d_grayImage, heightPerWidth);
  
  cudaMemcpy(d_grayImage, grayImage, grayImageSize, cudaMemcpyDeviceToHost);
  
  //step 3
  int *histogram;
  int *d_histogram;
  
  int histogramSize = HISTOGRAM_LENGTH * sizeof(int);
  
  cudaMalloc((void**)&d_histogram, histogramSize);
  
  histogram = (int *)malloc(histogramSize);
  
  cudaMemSet((void**)&d_histogram, 0, histogramSize);
  
  histogram_comput<<<heightPerWidth/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_histogram, d_grayImage, heightPerWidth);
  
  cudaMemcpy(d_histogram, histogram, histogramSize, cudaMemcpyDeviceToHost);
  
  //step 4
  float *cdf;
  float *d_cdf;
  
  int cdfSize = HISTOGRAM_LENGTH * sizeof(float);
  
  cudaMalloc((void**)&d_cdf, cdfSize);
  
  cdf = (float *)malloc(cdfSize);
  
  prescan<<<HISTOGRAM_LENGTH,1>>>(d_cdf, d_histogram, HISTOGRAM_LENGTH);
  
  float *finalCdf;
  float *d_finalCdf;
  
  int finalCdfSize = HISTOGRAM_LENGTH * sizeof(float);
  
  cudaMalloc((void**)&d_finalCdf, finalCdfSize);
  
  finalCdf = (float *)malloc(finalCdfSize);
  
  scan<<<HISTOGRAM_LENGTH,1>>>(finalCdf, d_cdf, HISTOGRAM_LENGTH);
  
  cudaMemcpy(d_finalCdf, finalCdf, finalCdfSize, cudaMemcpyDeviceToHost);
  
  free(ucharImage);
  free(grayImage);
  free(histogram);
  free(cdf);
  free(finalCdf);
  
  cudaFree(d_ucharImage);
  cudaFree(d_grayImage);
  cudaFree(d_histogram);
  cudaFree(d_cdf);
  cudaFree(d_finalCdf);
  
 
  
  
  return 0;
}