
//Includes for IntelliSense 

#define _SIZE_T_DEFINED 
#ifndef __CUDACC__ 
#define __CUDACC__ 
#endif 
#ifndef __cplusplus 
#define __cplusplus 
#endif

#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>


extern "C"  {
#define VT  4
	__device__ double distance(double dx, double dy)
	{
		if (dx != 0.0 && dy != 0.0)
			return (dx) * (dx) + (dy) * (dy);
		return 1.0;
	}
#ifdef CUDA_3_5
	// dynamic parallelism function call
	__device__  Idw4Item(const double * data_x, const double * data_y, const double * data_z, double * output, const double x, const double y, int N)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x * VT;
		if (i < N)
		{
			double d = distance((data_x[i] - x), (data_y[i] - y));
			output[i] = data_z[i] / d * d;
		}
	}

	__device__ double * data_output;
#endif
	__device__ double Idw4(const double * data_x, const double * data_y, const double * data_z, const double x, const double y, double z, int N)
	{
		//cudaMalloc()
		double d = 0.0;
		#pragma unroll
		for (int i = 0; i < N; i++)
		{			
			d = distance((data_x[i] - x), (data_y[i] - y)); // square of euclidian distance
			if (d < 90.0 && d > 1.0)			
				z += data_z[i] / d * d;        // data value weighted by distance to -4 power			
		}
		return z;
	}

	__device__ bool pointInPolygon(const double x, const double y, const double * xCoords, const double * yCoords, int nPoints)
	{
		//__shared__ 
		return false;
	}

	// Device code
	__global__ void RasterInterpolate(const double* Ax, const double * Ay, const double * Az, const double* Cx, const double * Cy, double * Cz, int N, int nData)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x * VT;
		if (i < N)
		{
			Cz[i] = Idw4(Ax, Ay, Az, Cx[i], Cy[i], Cy[i], nData);
		}
		
		/*if (i + VT < N)
		{
            #pragma unroll
			for (int j = i; j <= i + VT; j++)
				Cz[j] = Idw4(Ax, Ay, Az, Cx[j], Cy[j], Cy[j], nData);
		}
		else if (i < N)
		{            
			for (int j = i; j < N; j++)
				Cz[j] = Idw4(Ax, Ay, Az, Cx[j], Cy[j], Cy[j], nData);
		}*/
		__syncthreads();
	}

	__global__ void ClipRaster(const double* Ax, const double * Ay, const double * coordXs, const double * coordYs, uchar1 * mask, int N, int nPoints)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (pointInPolygon(Ax[i], Ay[i], coordXs, coordYs, nPoints))
		{
			uchar1* v = new uchar1();
			v->x = '\x01';
			mask[i] = *v;
		}
		else
		{
			uchar1* v = new uchar1();
			v->x = '\x00';
			mask[i] = *v;
		}
	}
}