using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.NPP;

namespace GrabCutNPP
{
	class GrabCut
	{	
		const int MAX_ITERATIONS = 10;
		const int COLOR_CLUSTER = 4;
		const float EDGE_STRENGTH = 50.0f;
		
		CudaPitchedDeviceVariable<uchar4> d_image;
		float edge_strength;

		public CudaPitchedDeviceVariable<uchar4> d_small_image;
		NppiSize small_size;

		CudaPitchedDeviceVariable<byte> d_trimap;

		CudaPitchedDeviceVariable<byte>[] d_small_trimap;
		int small_trimap_idx;

		NppiSize size;

		CudaPitchedDeviceVariable<int> d_terminals;
		CudaPitchedDeviceVariable<int> d_left_transposed, d_right_transposed;
		CudaPitchedDeviceVariable<int> d_top, d_bottom, d_topleft, d_topright, d_bottomleft, d_bottomright;
		

		CudaPitchedDeviceVariable<byte>[] d_alpha;
		int current_alpha;

		CudaDeviceVariable<byte> d_scratch_mem;
		GraphCut8 graphcut8;
		GraphCut8 graphcut8Small;
		//NppiGraphcutState* pState;

		CudaDeviceVariable<float> d_gmm;
		int gmm_pitch;

		int gmms;
		int blocks;

		int iteration;
		float runtime;

		public GrabCutGMM grabCutGMM;
		public GrabCutUtils grabCutUtils;

		public GrabCut(CudaPitchedDeviceVariable<uchar4> image, CudaPitchedDeviceVariable<byte> trimap, int width, int height)
		{
			d_trimap = trimap;
			//The first one will also init the CUDA context!
			grabCutUtils = new GrabCutUtils();
			grabCutGMM = new GrabCutGMM();
		
			size.width = width;
			size.height = height;
			graphcut8 = new GraphCut8(size);

			gmms = 2 * COLOR_CLUSTER;
			edge_strength = EDGE_STRENGTH;

			blocks = ((width+31)/32) * ((height+31)/32);
			gmm_pitch = 11 * sizeof(float);

			//d_image =  new CudaPitchedDeviceVariable<uchar4>(size.width, size.height);
			//d_image.CopyToDevice(image);
			d_image = image;

			// Doublebuffered alpha
			d_alpha = new CudaPitchedDeviceVariable<byte>[2];
			d_alpha[0] = new CudaPitchedDeviceVariable<byte>(size.width, size.height, 4);
			d_alpha[1] = new CudaPitchedDeviceVariable<byte>(size.width, size.height, 4);

			// Graph 
			d_terminals = new CudaPitchedDeviceVariable<int>(size.width, size.height);
			d_top = new CudaPitchedDeviceVariable<int>(size.width, size.height);
			d_topleft = new CudaPitchedDeviceVariable<int>(size.width, size.height);
			d_topright = new CudaPitchedDeviceVariable<int>(size.width, size.height);
			d_bottom = new CudaPitchedDeviceVariable<int>(size.width, size.height);
			d_bottomleft = new CudaPitchedDeviceVariable<int>(size.width, size.height);
			d_bottomright = new CudaPitchedDeviceVariable<int>(size.width, size.height);
			d_left_transposed = new CudaPitchedDeviceVariable<int>(size.height, size.width);
			d_right_transposed = new CudaPitchedDeviceVariable<int>(size.height, size.width);

	
			//int scratch_gc_size = 0;
			//nppiGraphcut8GetSize(size, &scratch_gc_size);

			int scratch_gmm_size = (int)(blocks * gmm_pitch * gmms + blocks * 4);
			d_scratch_mem = new CudaDeviceVariable<byte>(scratch_gmm_size);
			//CUDA_SAFE_CALL( cudaMalloc(&d_scratch_mem, MAX(scratch_gmm_size, scratch_gc_size)) );

			//NPP_CHECK_NPP(nppiGraphcutInitAlloc(size, &pState, d_scratch_mem) );
			d_gmm = new CudaDeviceVariable<float>(gmm_pitch * gmms);
			//CUDA_SAFE_CALL( cudaMalloc(&d_gmm, gmm_pitch * gmms) );

			// Estimate color models on lower res input image first
			createSmallImage(Math.Max(width/4, height/4));
			
		}

		private void createSmallImage(int max_dim)
		{
			int[] temp_width = new int[2];
			int[] temp_height = new int[2];

			CudaPitchedDeviceVariable<uchar4>[] d_temp = new CudaPitchedDeviceVariable<uchar4>[2];

			temp_width[0] = (int)Math.Ceiling(size.width * 0.5f);
			temp_height[0] = (int)Math.Ceiling(size.height * 0.5f);

			temp_width[1] = (int)Math.Ceiling(temp_width[0] * 0.5f);
			temp_height[1] = (int)Math.Ceiling(temp_height[0] * 0.5f);

			d_temp[0] = new CudaPitchedDeviceVariable<uchar4>(temp_width[0], temp_height[0]);
			d_temp[1] = new CudaPitchedDeviceVariable<uchar4>(temp_width[1], temp_height[1]);

			// Alloc also the small trimaps
			d_small_trimap = new CudaPitchedDeviceVariable<byte>[2];
			d_small_trimap[0] = new CudaPitchedDeviceVariable<byte>(temp_width[0], temp_height[0], 4);
			d_small_trimap[1] = new CudaPitchedDeviceVariable<byte>(temp_width[1], temp_height[1], 4);

			grabCutGMM.Downscale(d_temp[0], temp_width[0], temp_height[0], d_image, size.width, size.height);
			int current = 0;

			while (temp_width[current] > max_dim || temp_height[current] > max_dim)
			{
				grabCutGMM.Downscale(d_temp[1 - current], temp_width[1 - current], temp_height[1 - current], d_temp[current], temp_width[current], temp_height[current]);
				current ^= 1;
				temp_width[1 - current] = (int)Math.Ceiling(temp_width[current] * 0.5f);
				temp_height[1 - current] = (int)Math.Ceiling(temp_height[current] * 0.5f);
			}

			d_small_image = d_temp[current];
			small_size.width = temp_width[current];
			small_size.height = temp_height[current];

			graphcut8Small = new GraphCut8(small_size);
			d_temp[1 - current].Dispose();
		}

		private void createSmallTrimap()
		{
			int[] temp_width = new int[2];
			int[] temp_height = new int[2];

			temp_width[0] = (int)Math.Ceiling(size.width * 0.5f);
			temp_height[0] = (int)Math.Ceiling(size.height * 0.5f);

			temp_width[1] = (int)Math.Ceiling(temp_width[0] * 0.5f);
			temp_height[1] = (int)Math.Ceiling(temp_height[0] * 0.5f);

			grabCutGMM.DownscaleTrimap(d_small_trimap[0], temp_width[0], temp_height[0], d_trimap, size.width, size.height);

			small_trimap_idx = 0;

			while (temp_width[small_trimap_idx] != small_size.width)
			{
				grabCutGMM.DownscaleTrimap(d_small_trimap[1 - small_trimap_idx], temp_width[1 - small_trimap_idx], temp_height[1 - small_trimap_idx], d_small_trimap[small_trimap_idx], temp_width[small_trimap_idx], temp_height[small_trimap_idx]);
				small_trimap_idx ^= 1;
				temp_width[1 - small_trimap_idx] = (int)Math.Ceiling(temp_width[small_trimap_idx] * 0.5f);
				temp_height[1 - small_trimap_idx] = (int)Math.Ceiling(temp_height[small_trimap_idx] * 0.5f);
			}
		}

		public void computeSegmentationFromTrimap()
		{
			CudaStopWatch stopwatch = new CudaStopWatch();
			stopwatch.Start();

			iteration=0;
			current_alpha = 0;
	
			// Solve Grabcut on lower resolution first. Reduces total computation time.
			createSmallTrimap();

			d_alpha[0].AsyncCopyToDevice(d_small_trimap[small_trimap_idx], new CUstream());
			
			for (int i = 0; i < 2; ++i)
			{
				grabCutGMM.GMMInitialize(gmms, d_gmm, d_scratch_mem, gmm_pitch, d_small_image, d_alpha[current_alpha], small_size.width, small_size.height);
				
				grabCutGMM.GMMUpdate(gmms, d_gmm, d_scratch_mem, gmm_pitch, d_small_image, d_alpha[current_alpha], small_size.width, small_size.height);
				
				grabCutGMM.EdgeCues(edge_strength, d_small_image, d_left_transposed, d_right_transposed, d_top, d_bottom, d_topleft, d_topright, d_bottomleft, d_bottomright, small_size.width, small_size.height, d_scratch_mem);
				
				grabCutGMM.DataTerm(d_terminals, gmms, d_gmm, gmm_pitch, d_small_image, d_small_trimap[small_trimap_idx], small_size.width, small_size.height);

				graphcut8Small.GraphCut(d_terminals, d_left_transposed, d_right_transposed, d_top, d_topleft, d_topright, d_bottom, d_bottomleft, d_bottomright, d_alpha[1 - current_alpha]);

				current_alpha = 1 - current_alpha;
			}

			grabCutGMM.UpsampleAlpha(d_alpha[1-current_alpha], d_alpha[current_alpha], size.width, size.height, small_size.width, small_size.height);
			current_alpha = 1-current_alpha;
			
			grabCutGMM.GMMInitialize(gmms, d_gmm, d_scratch_mem, gmm_pitch, d_image, d_alpha[current_alpha], size.width, size.height);
			grabCutGMM.GMMUpdate(gmms, d_gmm, d_scratch_mem, gmm_pitch, d_image, d_alpha[current_alpha], size.width, size.height);

			while (true)
			{
				grabCutGMM.EdgeCues(edge_strength, d_image, d_left_transposed, d_right_transposed, d_top, d_bottom, d_topleft, d_topright, d_bottomleft, d_bottomright, size.width, size.height, d_scratch_mem);
				grabCutGMM.DataTerm(d_terminals, gmms, d_gmm, gmm_pitch, d_image, d_trimap, size.width, size.height);
				
				current_alpha = 1 ^ current_alpha;

				graphcut8.GraphCut(d_terminals, d_left_transposed, d_right_transposed, d_top, d_topleft, d_topright, d_bottom, d_bottomleft, d_bottomright, d_alpha[current_alpha]);
				
				if( iteration > 0 ) 
				{
					bool changed = grabCutGMM.SegmentationChanged(d_scratch_mem, d_alpha[1 - current_alpha], d_alpha[current_alpha], size.width, size.height);			
					
					// Solution has converged
					if( !changed ) break;
				}

				if( iteration > MAX_ITERATIONS ) 
				{
					// Does not converge, fallback to rect selection
					System.Windows.Forms.MessageBox.Show("Warning: Color models did not converge after " + MAX_ITERATIONS + " iterations.");
					break;
				}

				grabCutGMM.GMMInitialize(gmms, d_gmm, d_scratch_mem, gmm_pitch, d_image, d_alpha[current_alpha], size.width, size.height);
				grabCutGMM.GMMUpdate(gmms, d_gmm, d_scratch_mem, gmm_pitch, d_image, d_alpha[current_alpha], size.width, size.height);
				
				iteration++;
			}

			stopwatch.Stop();
			stopwatch.StopEvent.Synchronize();

			runtime = stopwatch.GetElapsedTime();
		}

		public void updateSegmentation()
		{
			grabCutGMM.EdgeCues(edge_strength, d_image, d_left_transposed, d_right_transposed, d_top, d_bottom, d_topleft, d_topright, d_bottomleft, d_bottomright, size.width, size.height, d_scratch_mem);
			grabCutGMM.DataTerm(d_terminals, gmms, d_gmm, gmm_pitch, d_image, d_trimap, size.width, size.height);
			graphcut8.GraphCut(d_terminals, d_left_transposed, d_right_transposed, d_top, d_topleft, d_topright, d_bottom, d_bottomleft, d_bottomright, d_alpha[current_alpha]);
			
		}

		public void updateImage(CudaPitchedDeviceVariable<uchar4> image)
		{
			d_image.CopyToDevice(image);
		}

		public CudaPitchedDeviceVariable<byte> AlphaMap 
		{
			get
			{
				return d_alpha[current_alpha];
			}
		}

		public float Runtime
		{
			get { return runtime; }
		}

		public int Iterations
		{
			get { return iteration; }
		}
	}
}
