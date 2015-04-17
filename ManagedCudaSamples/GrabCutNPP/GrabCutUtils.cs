using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Reflection;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.NPP;

namespace GrabCutNPP
{
	class GrabCutUtils
	{
		CudaContext ctx;
		CudaKernel TrimapFromRectKernel;
		CudaKernel ApplyMatteKernelMode0;
		CudaKernel ApplyMatteKernelMode1;
		CudaKernel ApplyMatteKernelMode2;
		CudaKernel convertRGBToRGBAKernel;

		public GrabCutUtils()
		{
			ctx = new CudaContext(CudaContext.GetMaxGflopsDeviceId(), false);


			//Load Kernel image from resources
			string resName;
			if (IntPtr.Size == 8)
				resName = "GrabCutUtils_x64.ptx";
			else
				resName = "GrabCutUtils.ptx";

			string resNamespace = "GrabCutNPP";
			string resource = resNamespace + "." + resName;
			Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(resource);
			if (stream == null) throw new ArgumentException("Kernel not found in resources.");
			byte[] kernel = new byte[stream.Length];

			int bytesToRead = (int)stream.Length;
			while (bytesToRead > 0)
			{
				bytesToRead -= stream.Read(kernel, (int)stream.Position, bytesToRead);
			}

            TrimapFromRectKernel = ctx.LoadKernelPTX(kernel, "_Z20TrimapFromRectKernelPhi8NppiRectii");
            ApplyMatteKernelMode0 = ctx.LoadKernelPTX(kernel, "_Z16ApplyMatteKernelILi0EEvP6uchar4iPKS0_iPKhiii");
            ApplyMatteKernelMode1 = ctx.LoadKernelPTX(kernel, "_Z16ApplyMatteKernelILi1EEvP6uchar4iPKS0_iPKhiii");
            ApplyMatteKernelMode2 = ctx.LoadKernelPTX(kernel, "_Z16ApplyMatteKernelILi2EEvP6uchar4iPKS0_iPKhiii");
            convertRGBToRGBAKernel = ctx.LoadKernelPTX(kernel, "_Z22convertRGBToRGBAKernelP6uchar4iP6uchar3iii");

		}

		public void TrimapFromRect(CudaPitchedDeviceVariable<byte> alpha, NppiRect rect, int width, int height )
		{
			dim3 block = new dim3(32,8,1);
			dim3 grid = new dim3((int)( (width+(block.x*4)-1) / (block.x*4)), (height+31) / 32,1);

			//rect.y = height - 1 - (rect.y + rect.height - 1) ; // Flip horizontal (FreeImage inverts y axis)
			
			TrimapFromRectKernel.BlockDimensions = block;
			TrimapFromRectKernel.GridDimensions = grid;
			TrimapFromRectKernel.Run(alpha.DevicePointer, (int)alpha.Pitch, rect, width, height);
			//TrimapFromRectKernel<<<grid, block>>>(alpha, alpha_pitch, rect, width, height );

		}

		public void ApplyMatte(int mode, CudaPitchedDeviceVariable<uchar4> result, CudaPitchedDeviceVariable<uchar4> image, CudaPitchedDeviceVariable<byte> matte, int width, int height)
		{
			dim3 block = new dim3(32, 8, 1);
			dim3 grid = new dim3((width + 31) / 32, (height + 31) / 32, 1);

			switch (mode)
			{
				case 0:
					ApplyMatteKernelMode0.BlockDimensions = block;
					ApplyMatteKernelMode0.GridDimensions = grid;
					ApplyMatteKernelMode0.Run(result.DevicePointer, (int)result.Pitch / 4, image.DevicePointer, (int)image.Pitch / 4, matte.DevicePointer, (int)matte.Pitch, width, height);

					//ApplyMatteKernel<0><<<grid, block>>>(result, result_pitch/4, image, image_pitch/4, matte, matte_pitch, width, height);
					break;

				case 1:
					ApplyMatteKernelMode1.BlockDimensions = block;
					ApplyMatteKernelMode1.GridDimensions = grid;
					ApplyMatteKernelMode1.Run(result.DevicePointer, (int)result.Pitch / 4, image.DevicePointer, (int)image.Pitch / 4, matte.DevicePointer, (int)matte.Pitch, width, height);

					//ApplyMatteKernel<1><<<grid, block>>>(result, result_pitch/4, image, image_pitch/4, matte, matte_pitch, width, height);
					break;

				case 2:
					ApplyMatteKernelMode2.BlockDimensions = block;
					ApplyMatteKernelMode2.GridDimensions = grid;
					ApplyMatteKernelMode2.Run(result.DevicePointer, (int)result.Pitch / 4, image.DevicePointer, (int)image.Pitch / 4, matte.DevicePointer, (int)matte.Pitch, width, height);

					//ApplyMatteKernel<2><<<grid, block>>>(result, result_pitch/4, image, image_pitch/4, matte, matte_pitch, width, height);
					break;

			}
		}

		public void convertRGBToRGBA(CudaPitchedDeviceVariable<uchar4> i4, CudaPitchedDeviceVariable<uchar3> i3, int width, int height)
		{
			dim3 block = new dim3(32, 8, 1);
			dim3 grid = new dim3((width + 31) / 32, (height + 31) / 32, 1);
			
			convertRGBToRGBAKernel.BlockDimensions = block;
			convertRGBToRGBAKernel.GridDimensions = grid;
			
			convertRGBToRGBAKernel.Run(i4.DevicePointer, (int)i4.Pitch, i3.DevicePointer, (int)i3.Pitch, width, height);
			//convertRGBToRGBAKernel<<<grid, block>>>(i4, i4_pitch, i3, i3_pitch, width, height);
		}

	}
}
