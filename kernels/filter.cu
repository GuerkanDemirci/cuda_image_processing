extern "C" __global__ void grayscale(unsigned char* d_sourceImagePtr, unsigned char* d_outImagePtr,int width, int height)
{
	const int i = blockIdx.y * blockDim.y + threadIdx.y;
	const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= height || j >= width ) {
        return;
    }

    const int index = j*3 + i * width*3;
    const int gray_color = (d_sourceImagePtr[index + 0] + d_sourceImagePtr[index + 1] + d_sourceImagePtr[index + 2]) / 3;

    d_outImagePtr[index + 0] = gray_color;
    d_outImagePtr[index + 1] = gray_color;
    d_outImagePtr[index + 2] = gray_color;
}
