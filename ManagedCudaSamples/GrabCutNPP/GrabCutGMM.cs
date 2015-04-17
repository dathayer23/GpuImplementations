using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Reflection;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;


namespace GrabCutNPP
{
	class GrabCutGMM
	{		
		static int[] det_indices = new int[]{ (9 << (4*4)) + (4 << (3*4)) + (6 << (2*4)) + (5 << (1*4)) + (4 << (0*4)),
		(5 << (4*4)) + (8 << (3*4)) + (6 << (2*4)) + (6 << (1*4)) + (7 << (0*4)),
		(5 << (4*4)) + (8 << (3*4)) + (7 << (2*4)) + (8 << (1*4)) + (9 << (0*4))};


		static int[] inv_indices = new int[]{ (4 << (5*4)) + (5 << (4*4)) + (4 << (3*4)) + (5 << (2*4)) + (6 << (1*4)) + (7 << (0*4)),
		(7 << (5*4)) + (6 << (4*4)) + (9 << (3*4)) + (8 << (2*4)) + (8 << (1*4)) + (9 << (0*4)),
		(5 << (5*4)) + (4 << (4*4)) + (6 << (3*4)) + (6 << (2*4)) + (5 << (1*4)) + (8 << (0*4)),
		(5 << (5*4)) + (8 << (4*4)) + (6 << (3*4)) + (7 << (2*4)) + (9 << (1*4)) + (8 << (0*4))};


		CudaContext ctx;
		CudaKernel GMMReductionKernelCreateGmmFlags;
		CudaKernel GMMReductionKernelNoCreateGmmFlags;
		CudaKernel GMMFinalizeKernelInvertSigma;
		CudaKernel GMMFinalizeKernelNoInvertSigma;
		CudaKernel GMMcommonTerm;
		CudaKernel DataTermKernel;
		CudaKernel GMMAssignKernel;
		CudaKernel GMMFindSplit;
		CudaKernel GMMDoSplit;
		CudaKernel MeanEdgeStrengthReductionKernel;
		CudaKernel MeanEdgeStrengthFinalKernel;
		CudaKernel EdgeCuesKernel;
		CudaKernel SegmentationChangedKernel;
		CudaKernel downscaleKernel1;
		CudaKernel downscaleKernel2;
		CudaKernel upsampleAlphaKernel;
		CudaTextureLinearPitched2D<uchar4> texref;
		CudaTextureLinearPitched2D<uchar4> texref2;

		public GrabCutGMM()
		{
			ctx = new CudaContext(CudaContext.GetMaxGflopsDeviceId(), false);


			//Load Kernel image from resources
			string resName;
			if (IntPtr.Size == 8)
				resName = "GrabCutGMM_x64.ptx";
			else
				resName = "GrabCutGMM.ptx";

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
			 
			CUmodule module = ctx.LoadModulePTX(kernel);
            GMMReductionKernelCreateGmmFlags = new CudaKernel("_Z18GMMReductionKernelILi4ELb1EEviPfiPK6uchar4iPhiiiPj", module, ctx);
			GMMReductionKernelNoCreateGmmFlags = new CudaKernel("_Z18GMMReductionKernelILi4ELb0EEviPfiPK6uchar4iPhiiiPj", module, ctx);
			GMMFinalizeKernelInvertSigma = new CudaKernel("_Z17GMMFinalizeKernelILi4ELb1EEvPfS0_ii", module, ctx);
			GMMFinalizeKernelNoInvertSigma = new CudaKernel("_Z17GMMFinalizeKernelILi4ELb0EEvPfS0_ii", module, ctx);
			GMMcommonTerm = new CudaKernel("_Z13GMMcommonTermiPfi", module, ctx);
			DataTermKernel = new CudaKernel("_Z14DataTermKernelPiiiPKfiPK6uchar4iPKhiii", module, ctx);
			GMMAssignKernel = new CudaKernel("_Z15GMMAssignKerneliPKfiPK6uchar4iPhiii", module, ctx);
			GMMFindSplit = new CudaKernel("_Z12GMMFindSplitP10GMMSplit_tiPfi", module, ctx);
			GMMDoSplit = new CudaKernel("_Z10GMMDoSplitPK10GMMSplit_tiPfiPK6uchar4iPhiii", module, ctx);
			MeanEdgeStrengthReductionKernel = new CudaKernel("_Z31MeanEdgeStrengthReductionKerneliiPf", module, ctx);
			MeanEdgeStrengthFinalKernel = new CudaKernel("_Z27MeanEdgeStrengthFinalKernelPfi", module, ctx);
			EdgeCuesKernel = new CudaKernel("_Z14EdgeCuesKernelfPKfPiS1_S1_S1_S1_S1_S1_S1_iiii", module, ctx);
			SegmentationChangedKernel = new CudaKernel("_Z25SegmentationChangedKernelPiPhS0_iii", module, ctx);
			downscaleKernel1 = new CudaKernel("_Z18downscaleKernelBoxI6uchar4EvPT_iiiPKS1_iii", module, ctx);
			downscaleKernel2 = new CudaKernel("_Z18downscaleKernelMaxIhEvPT_iiiPKS0_iii", module, ctx);
			upsampleAlphaKernel = new CudaKernel("_Z19upsampleAlphaKernelPhS_iiii", module, ctx);

			GMMFinalizeKernelInvertSigma.SetConstantVariable("det_indices", det_indices);
			GMMFinalizeKernelInvertSigma.SetConstantVariable("inv_indices", inv_indices);
			GMMFinalizeKernelNoInvertSigma.SetConstantVariable("det_indices", det_indices);
			GMMFinalizeKernelNoInvertSigma.SetConstantVariable("inv_indices", inv_indices);
		}

		public void GMMUpdate(int gmm_N, CudaDeviceVariable<float> gmm, CudaDeviceVariable<byte> scratch_mem, int gmm_pitch, CudaPitchedDeviceVariable<uchar4> image, CudaPitchedDeviceVariable<byte> alpha, int width, int height)
		{
			dim3 grid = new dim3((width + 31) / 32, (height + 31) / 32, 1);
			dim3 block = new dim3(32, 4, 1);

			GMMReductionKernelCreateGmmFlags.BlockDimensions = block;
			GMMReductionKernelCreateGmmFlags.GridDimensions = grid;
			GMMReductionKernelNoCreateGmmFlags.BlockDimensions = block;
			GMMReductionKernelNoCreateGmmFlags.GridDimensions = grid;

			GMMReductionKernelCreateGmmFlags.Run((int)0, scratch_mem.DevicePointer + (grid.x * grid.y * 4), (int)gmm_pitch / 4, image.DevicePointer, (int)image.Pitch / 4, alpha.DevicePointer, (int)alpha.Pitch, width, height, scratch_mem.DevicePointer);

			//GMMReductionKernel<4, true><<<grid, block>>>(0, &scratch_mem[grid.x * grid.y], gmm_pitch/4, image, image_pitch/4, alpha, alpha_pitch, width, height, (unsigned int*) scratch_mem);
			for (int i = 1; i < gmm_N; ++i)
			{
				GMMReductionKernelNoCreateGmmFlags.Run(i, scratch_mem.DevicePointer + (grid.x * grid.y * 4), (int)gmm_pitch / 4, image.DevicePointer, (int)image.Pitch / 4, alpha.DevicePointer, (int)alpha.Pitch, width, height, scratch_mem.DevicePointer);

				//GMMReductionKernel<4, false><<<grid, block>>>(i, &scratch_mem[grid.x * grid.y], gmm_pitch/4, image, image_pitch/4, alpha, alpha_pitch, width, height, (unsigned int*) scratch_mem);
			}
			GMMFinalizeKernelInvertSigma.BlockDimensions = new dim3(32 * 4, 1, 1);
			GMMFinalizeKernelInvertSigma.GridDimensions = new dim3(gmm_N, 1, 1);
			GMMFinalizeKernelInvertSigma.Run(gmm.DevicePointer, scratch_mem.DevicePointer + (grid.x * grid.y * 4), (int)gmm_pitch / 4, grid.x * grid.y);
			//GMMFinalizeKernel<4, true><<<gmm_N, 32*4>>>(gmm, &scratch_mem[grid.x * grid.y], gmm_pitch/4, grid.x * grid.y);

			block.x = 32; block.y = 2;
			GMMcommonTerm.BlockDimensions = block;
			GMMcommonTerm.GridDimensions = new dim3(1, 1, 1);
			GMMcommonTerm.Run(gmm_N / 2, gmm.DevicePointer, (int)gmm_pitch / 4);
			//GMMcommonTerm<<<1, block>>>(gmm_N / 2, gmm, gmm_pitch/4);
		}

		public void DataTerm(CudaPitchedDeviceVariable<int> terminals, int gmmN, CudaDeviceVariable<float> gmm, int gmm_pitch, CudaPitchedDeviceVariable<uchar4> image, CudaPitchedDeviceVariable<byte> trimap, int width, int height)
		{
			dim3 block = new dim3(32, 8, 1);
			dim3 grid = new dim3((int)((width + block.x - 1) / block.x), (int)((height + block.y - 1) / block.y), 1);

			DataTermKernel.BlockDimensions = block;
			DataTermKernel.GridDimensions = grid;
			DataTermKernel.Run(terminals.DevicePointer, (int)terminals.Pitch / 4, gmmN, gmm.DevicePointer, (int)gmm_pitch / 4, image.DevicePointer, (int)image.Pitch / 4, trimap.DevicePointer, (int)trimap.Pitch, width, height);
			//DataTermKernel<<<grid, block>>>(terminals, terminal_pitch/4, gmmN, gmm, gmm_pitch/4, image, image_pitch/4, trimap, trimap_pitch, width, height);

		}

		public void GMMAssign(int gmmN, CudaDeviceVariable<float> gmm, int gmm_pitch, CudaPitchedDeviceVariable<uchar4> image, CudaPitchedDeviceVariable<byte> alpha, int width, int height) 
		{
			dim3 block = new dim3(32, 16, 1);
			dim3 grid = new dim3((int)((width+block.x-1) / block.x), (int)((height+block.y-1) / block.y), 1);
			
			GMMAssignKernel.BlockDimensions = block;
			GMMAssignKernel.GridDimensions = grid;
			GMMAssignKernel.Run(gmmN, gmm.DevicePointer, (int)gmm_pitch/4, image.DevicePointer, (int)image.Pitch/4, alpha.DevicePointer, (int)alpha.Pitch, width, height);
			//GMMAssignKernel<<<grid, block>>>(gmmN, gmm, gmm_pitch/4, image, image_pitch/4, alpha, alpha_pitch, width, height);
		}

		public void GMMInitialize(int gmm_N, CudaDeviceVariable<float> gmm, CudaDeviceVariable<byte> scratch_mem, int gmm_pitch, CudaPitchedDeviceVariable<uchar4> image, CudaPitchedDeviceVariable<byte> alpha, int width, int height)
		{
			dim3 grid = new dim3((width + 31) / 32, (height + 31) / 32, 1);
			dim3 block = new dim3(32, 4, 1);
			dim3 smallblock = new dim3(32, 2, 1);

			GMMReductionKernelCreateGmmFlags.BlockDimensions = block;
			GMMReductionKernelCreateGmmFlags.GridDimensions = grid;
			GMMReductionKernelNoCreateGmmFlags.BlockDimensions = block;
			GMMReductionKernelNoCreateGmmFlags.GridDimensions = grid;
			GMMFinalizeKernelNoInvertSigma.BlockDimensions = new dim3(32 * 4, 1, 1);

			GMMFindSplit.BlockDimensions = smallblock;
			GMMFindSplit.GridDimensions = new dim3(1, 1, 1);
			GMMDoSplit.BlockDimensions = block;
			GMMDoSplit.GridDimensions = grid;

			for (int k = 2; k < gmm_N; k += 2)
			{
				GMMReductionKernelCreateGmmFlags.Run(0, scratch_mem.DevicePointer + (grid.x * grid.y * 4), (int)gmm_pitch / 4, image.DevicePointer, (int)image.Pitch / 4, alpha.DevicePointer, (int)alpha.Pitch, width, height, scratch_mem.DevicePointer);
				//GMMReductionKernel<4, true><<<grid, block>>>(0, &scratch_mem[grid.x * grid.y], gmm_pitch/4, image, image_pitch/4, alpha, alpha_pitch, width, height, (unsigned int*) scratch_mem);

				for (int i = 1; i < k; ++i)
				{
					GMMReductionKernelNoCreateGmmFlags.Run(i, scratch_mem.DevicePointer + (grid.x * grid.y * 4), (int)gmm_pitch / 4, image.DevicePointer, (int)image.Pitch / 4, alpha.DevicePointer, (int)alpha.Pitch, width, height, scratch_mem.DevicePointer);
					//GMMReductionKernel<4, false><<<grid, block>>>(i, &scratch_mem[grid.x * grid.y], gmm_pitch/4, image, image_pitch/4, alpha, alpha_pitch, width, height, (unsigned int*) scratch_mem);
				}
				GMMFinalizeKernelNoInvertSigma.GridDimensions = new dim3(k, 1, 1);
				GMMFinalizeKernelNoInvertSigma.Run(gmm.DevicePointer, scratch_mem.DevicePointer + (grid.x * grid.y * 4), gmm_pitch / 4, grid.x * grid.y);
				//GMMFinalizeKernel<4, false><<<k, 32*4>>>(gmm, &scratch_mem[grid.x * grid.y], gmm_pitch/4, grid.x * grid.y);

				GMMFindSplit.Run(scratch_mem.DevicePointer, k / 2, gmm.DevicePointer, gmm_pitch / 4);
				//GMMFindSplit<<<1, smallblock>>>((GMMSplit_t*) scratch_mem, k / 2, gmm, gmm_pitch/4);
				GMMDoSplit.Run(scratch_mem.DevicePointer, (k / 2) << 1, gmm.DevicePointer, gmm_pitch / 4, image.DevicePointer, (int)image.Pitch / 4, alpha.DevicePointer, (int)alpha.Pitch, width, height);
				//GMMDoSplit<<<grid, block>>>((GMMSplit_t*) scratch_mem, (k/2) << 1, gmm, gmm_pitch/4, image, image_pitch / 4, alpha, alpha_pitch, width, height);
			}

		}

		public void EdgeCues( float alpha, CudaPitchedDeviceVariable<uchar4> image, CudaPitchedDeviceVariable<int> left_transposed, CudaPitchedDeviceVariable<int> right_transposed, 
			CudaPitchedDeviceVariable<int> top, CudaPitchedDeviceVariable<int> bottom, CudaPitchedDeviceVariable<int> topleft, CudaPitchedDeviceVariable<int> topright, 
			CudaPitchedDeviceVariable<int> bottomleft, CudaPitchedDeviceVariable<int> bottomright, int width, int height, CudaDeviceVariable<byte> scratch_mem  )
		{
			if (texref == null)
				texref = new CudaTextureLinearPitched2D<uchar4>(MeanEdgeStrengthReductionKernel, "imageTex", CUAddressMode.Clamp, CUFilterMode.Point, CUTexRefSetFlags.ReadAsInteger, CUArrayFormat.UnsignedInt8, image);
			else
				texref.Reset(image);

			if (texref2 == null)
				texref2 = new CudaTextureLinearPitched2D<uchar4>(EdgeCuesKernel, "imageTex", CUAddressMode.Clamp, CUFilterMode.Point, CUTexRefSetFlags.ReadAsInteger, CUArrayFormat.UnsignedInt8, image);
			else
				texref2.Reset(image);

			
			dim3 grid = new dim3( (width+31) / 32, (height+31) / 32, 1);
			dim3 block = new dim3(32, 4, 1);
			dim3 large_block = new dim3(32,8,1);
			
			MeanEdgeStrengthReductionKernel.BlockDimensions = large_block;
			MeanEdgeStrengthReductionKernel.GridDimensions = grid;
			MeanEdgeStrengthFinalKernel.BlockDimensions = block;
			MeanEdgeStrengthFinalKernel.GridDimensions = new dim3(1,1,1);
			EdgeCuesKernel.BlockDimensions = block;
			EdgeCuesKernel.GridDimensions = grid;

			MeanEdgeStrengthReductionKernel.Run(width, height, scratch_mem.DevicePointer);

			//MeanEdgeStrengthReductionKernel<<<grid, large_block>>>( width, height, scratch_mem);
			MeanEdgeStrengthFinalKernel.Run(scratch_mem.DevicePointer, grid.x * grid.y);
			//MeanEdgeStrengthFinalKernel<<<1,block>>>( scratch_mem, grid.x * grid.y);

			EdgeCuesKernel.Run(alpha, scratch_mem.DevicePointer, left_transposed.DevicePointer, right_transposed.DevicePointer, top.DevicePointer, bottom.DevicePointer, topleft.DevicePointer, topright.DevicePointer,
				bottomleft.DevicePointer, bottomright.DevicePointer, (int)top.Pitch/4, (int)right_transposed.Pitch/4, width, height);
			//EdgeCuesKernel<<<grid, block>>>( alpha , scratch_mem, left_transposed, right_transposed, top, bottom, topleft, topright, bottomleft, bottomright, pitch / 4, transposed_pitch/ 4, width, height );

		}

		public bool SegmentationChanged(CudaDeviceVariable<byte> d_changed, CudaPitchedDeviceVariable<byte> alpha_old, CudaPitchedDeviceVariable<byte> alpha_new, int width, int height) 
		{
			dim3 grid = new dim3( (width+31) / 32, (height+31) / 32, 1);
			dim3 block = new dim3(32, 8, 1);

			CudaDeviceVariable<int> d_changedInt = new CudaDeviceVariable<int>(d_changed.DevicePointer, false);
			d_changedInt[0] = 0;
			
			SegmentationChangedKernel.BlockDimensions = block;
			SegmentationChangedKernel.GridDimensions = grid;
			SegmentationChangedKernel.Run(d_changedInt.DevicePointer, alpha_old.DevicePointer, alpha_new.DevicePointer, (int)alpha_old.Pitch, width, height);
			//SegmentationChangedKernel<<<grid, block>>>(d_changed, alpha_old, alpha_new, alpha_pitch, width, height);

			int h_changed = d_changedInt[0];
			//error = cudaMemcpy(&h_changed, d_changed, 4, cudaMemcpyDeviceToHost);

			return (h_changed != 0);
		}

		public void Downscale(CudaPitchedDeviceVariable<uchar4> small_image, int small_width, int small_height, CudaPitchedDeviceVariable<uchar4> image, int width, int height)
		{ 
			dim3 grid = new dim3((width + 63)/64, (height+63)/64, 1);
			dim3 block = new dim3(32,8,1);

			downscaleKernel1.BlockDimensions = block;
			downscaleKernel1.GridDimensions = grid;
			downscaleKernel1.Run(small_image.DevicePointer, (int)small_image.Pitch / 4, small_width, small_height, image.DevicePointer, (int)image.Pitch / 4, width, height);

			//downscaleKernel<<<grid, block>>>(small_image, small_pitch/4, small_width, small_height, image, pitch/4, width, height, boxfilter_functor());

		}

		public void DownscaleTrimap(CudaPitchedDeviceVariable<byte> small_image, int small_width, int small_height, CudaPitchedDeviceVariable<byte> image, int width, int height) 
		{
	
			dim3 grid = new dim3((width + 63)/64, (height+63)/64, 1);
			dim3 block = new dim3(32,8,1);
			
			downscaleKernel2.BlockDimensions = block;
			downscaleKernel2.GridDimensions = grid;
			downscaleKernel2.Run(small_image.DevicePointer, (int)small_image.Pitch, small_width, small_height, image.DevicePointer, (int)image.Pitch, width, height);
			//downscaleKernel<<<grid, block>>>(small_image, small_pitch, small_width, small_height, image, pitch, width, height, maxfilter_functor());
	
		}

		public void UpsampleAlpha(CudaPitchedDeviceVariable<byte> alpha, CudaPitchedDeviceVariable<byte> small_alpha, int width, int height, int small_width, int small_height)
		{
			dim3 grid = new dim3((width + 127) / 128, (height + 31) / 32, 1);
			dim3 block = new dim3(32,8, 1);

			int factor = width / small_width;
			int shift = 0;

			while(factor > (1<<shift)) shift++;
			
			upsampleAlphaKernel.BlockDimensions = block;
			upsampleAlphaKernel.GridDimensions = grid;
			upsampleAlphaKernel.Run(alpha.DevicePointer, small_alpha.DevicePointer, (int)alpha.Pitch, width, height, shift);
			//upsampleAlphaKernel<<<grid, block>>>(alpha, small_alpha, alpha_pitch, width, height, shift);

		}
	}
}
