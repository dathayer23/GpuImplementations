using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using GeoAPI.Geometries;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using WilburEllis.RisingTide.Base;
using WilburEllis.RisingTide.Common.DBLayer;
using WilburEllis.RisingTide.MachineIO;
using WilburEllis.RisingTide.Spatial;
using WilburEllis.RisingTide.Spatial.Raster;
using WilburEllis.RisingTide.Services.ControllerWindowsService;
using WilburEllis.RisingTide.Services.Common.Services;
using WilburEllis.RisingTide.Spatial.Imaging;
using WilburEllis.RisingTide.Spatial.Interpolation.Algorithm;
using WilburEllis.RisingTide.Spatial.Interpolation;

namespace GpuInterpolation
{

    public class CalculateHeatmap
    {
        CudaContext ctx;
        public CudaDeviceProperties dev;
        CudaKernel InterpolateKernel;
        //static float3[] h_A;
        //static float3[] h_C;
        //static CudaDeviceVariable<float3> d_A;
        //static CudaDeviceVariable<float3> d_C;
        public CalculateHeatmap()
        {
            ctx = new CudaContext(CudaContext.GetMaxGflopsDeviceId());
            dev = ctx.GetDeviceInfo();
            Console.WriteLine("Using CUDA Device {0} compute level {1} timeout {2}", dev.DeviceName, dev.ComputeCapability, dev.KernelExecTimeoutEnabled ? "enabled" : "disabled");
            string resName;
            resName = @"C:\WEDEV\GpuImplementations\GpuInterpolation\RasterInterpolation_x64.ptx";            
            Console.WriteLine("Loading Interpolation Kernel");
            InterpolateKernel = ctx.LoadKernelPTX(resName, "RasterInterpolate");
        }

        public RtRaster<double> RunCpuInterpolation(YieldReportData dataPoints, IGeometry dataBoundary)
        {
            var bounds = new RTBounds(dataBoundary).ToMeters();
            var xMax = bounds.MaxX;
            var xMin = bounds.MinX;
            var yMax = bounds.MaxY;
            var yMin = bounds.MinY;

            CollectionStats yld = dataPoints.YieldData.Yield;
            Stopwatch timer = new Stopwatch();
            timer.Start();
            var ip = new Interpolator();
            var raster =  ip.OldInterpolate(yld.data.Select(v => new RtPointZ(v.X, v.Y, (float)v.Z)).ToList(), xMin, xMax, yMin, yMax, 0.0f, 4, 1, 1.0f, 1.0f);

            //var raster = CommonServices.CreateHeatMap(yld.data.Select(v => new RtPointZ(v.X,v.Y,(float)v.Z)).ToList(), xMin,yMin,xMax,yMax);
            timer.Stop();
            var elapsed = timer.ElapsedTicks;
            Console.WriteLine("Cpu Interpolation took {0} ms", elapsed / (Stopwatch.Frequency / 1000));

            RtRaster<double> rasterDbl = new RtRaster<double>(raster.NumCols, raster.NumRows, raster.Xll, raster.Yll, 1.0, 1);
            rasterDbl.SetRasterBand(0, raster.Bands[0].TransformDataBand(v => (double)v));
            return rasterDbl;
        }

        public RtRaster<double> RunGpuInterpolation(YieldReportData dataPoints, IGeometry dataBoundary)
        {
            try
            {
                int cellSize = 1;
                var bounds = new RTBounds(dataBoundary).ToMeters();
                var xMax = bounds.MaxX;
                var xMin = bounds.MinX;
                var yMax = bounds.MaxY;
                var yMin = bounds.MinY;
                var xDist = (int)(xMax - xMin + 1);
                var yDist = (int)(yMax - yMin + 1);
                //var cost = (xDist * yDist * dataPoints.Count) / (Environment.ProcessorCount * cellSize * cellSize);
                var nCols = xDist / cellSize;
                var nRows = yDist / cellSize;
                
                CollectionStats yld = dataPoints.YieldData.Yield;
                //create host side arrays for input data and boundary and output data
                double[] h_datax = yld.DataPoints.Select(pt => RtPoint<double>.LatLonToMeters(pt)).Select(ptz => ptz.X).ToArray();
                double[] h_datay = yld.DataPoints.Select(pt => RtPoint<double>.LatLonToMeters(pt)).Select(ptz => ptz.Y).ToArray();
                double[] h_dataz = yld.DataPoints.Select(pt => RtPoint<double>.LatLonToMeters(pt)).Select(ptz => ptz.Z).ToArray();

                var dataSize = sizeof(double) * h_datax.Length;
                Console.WriteLine("Performing interpolation on {0} data points of {1} bytes", h_datax.Length, dataSize);
                var cols = Enumerable.Range(0, nCols - 1).ToArray();
                var coords = Enumerable.Range(0, nRows).SelectMany(row => cols.Select(col => Tuple.Create(row, col))).ToArray();
                double[] h_outputx = coords.Select(coord => xMin + (coord.Item2 * cellSize)).ToArray();
                double[] h_outputy = coords.Select(coord => yMin + (coord.Item1 * cellSize)).ToArray();
                double[] h_outputz = coords.Select(coord => 0.0).ToArray();

                var outputSize = sizeof(double) * h_outputx.Length;
                Console.WriteLine("Output Raster has {0} raster points of {1} bytes", h_outputx.Length, outputSize);
                double2[] h_boundCoords = dataBoundary.Coordinates.Select(coord => new double2(coord.X, coord.Y)).ToArray();
                using (CudaDeviceVariable<double> d_datax = h_datax)
                using (CudaDeviceVariable<double> d_datay = h_datay)
                using (CudaDeviceVariable<double> d_dataz = h_dataz)
                using (CudaDeviceVariable<double> d_outputx = h_outputx)
                using (CudaDeviceVariable<double> d_outputy = h_outputy)
                using (CudaDeviceVariable<double> d_outputz = h_outputz)
                {
                    const int ThreadsPerBlock = 512;
                    InterpolateKernel.BlockDimensions = ThreadsPerBlock;
                    InterpolateKernel.GridDimensions = (h_outputx.Length + ThreadsPerBlock - 1) / ThreadsPerBlock;
                    Console.WriteLine("Invoke Kernel with Grid Dimensions = {0}", InterpolateKernel.GridDimensions);

                    Stopwatch timer = new Stopwatch();
                    timer.Start();
                    InterpolateKernel.Run(d_datax.DevicePointer, d_datay.DevicePointer, d_dataz.DevicePointer, 
                                          d_outputx.DevicePointer, d_outputy.DevicePointer, d_outputz.DevicePointer, 
                                          h_outputx.Length, h_datax.Length);
                    timer.Stop();
                    var elapsed = timer.ElapsedTicks;
                    Console.WriteLine("Gpu Interpolation took {0} ms", elapsed/(Stopwatch.Frequency/1000));
                    h_outputz = d_outputz;
                }

                if (h_outputz.All(pt => pt != 0.0))
                {
                    Console.WriteLine("Kernel succeeded");
                }
                else
                {
                    Console.WriteLine("Kernel failed");
                }

                var raster = new RtRaster<double>(nCols, nRows, xMin, yMin, cellSize * 2);
                var noDataValue = 0.0f;
                var rasterBand = RtRasterBand<double>.CreateRasterBand(typeof(double), nCols, nRows, noDataValue, h_outputz);
                raster.SetRasterBand(0, rasterBand);
                
                return raster;
            }

            
            catch(Exception ex)
            {
                ctx.Dispose();

                Console.WriteLine(ex.Message + "\nStackTrace: " + ex.StackTrace);
                //+ 
                //    ex.InnerException == null ? "" : 
                //    String.Format("\n\t{0}\n\tStackTrace: {1}", ex.InnerException.Message, 
                //    ex.InnerException.StackTrace)
                
                return null;
            }

        }

        

        public void SaveResults(RtRaster<double> raster, IControllerDataService dataService, string name)
        {
            var res = CalculateHeatmap.GeneratePaletteAndHistogramForData(dataService, raster);
            SaveRasterToFile(raster, res.Item1, name);
        }

        private void SaveRasterToFile(RtRaster<double> raster, RtColorPalette palette, string name)
        {
            //raster.CellSize = 2;
            var bmpAndPalette = GetRasterBitmap(raster, palette);
            Bitmap bmp = bmpAndPalette.Item1;

            bmp.Save(name, ImageFormat.Png);
        }

        private Tuple<Bitmap, RtColorPalette> GetRasterBitmap(RtRaster<double> raster, RtColorPalette palette)
        {            
            if (palette != null)
            {
                var image = raster.ToImage(palette, Color.AntiqueWhite);
                return Tuple.Create(image, palette);
            }

            return Tuple.Create(null as Bitmap, null as RtColorPalette);
        }

        public static Tuple<RtColorPalette, Quantile[], string> GeneratePaletteAndHistogramForData(IDataService _dataService, RtRaster<double> raster, bool ignoreZero = true, Action<int, string> trace = null)
        {
            var rasterBand = raster.Bands[0] as IRasterBand<double>;
            var distinct = rasterBand.RasterData.Where(v => v != rasterBand.NoDataValue && v != double.MinValue)
                .Distinct().OrderBy(x => x)
                .ToArray();

            var cp = new RtColorPalette();
            int index = 0;
            double hMin = double.MinValue;
            if (distinct.Length > 0)
            {
                hMin = distinct[index];
                while (hMin == double.MinValue)
                {
                    index++;
                    if (index < distinct.Length)
                        hMin = distinct[index++];
                    else
                        break;
                }
                if (hMin == double.MinValue)
                {
                    //Defer exception creation until later
                    return Tuple.Create(cp, new Quantile[] { }, "No values in Heatmap from which to generate a histogram");
                }
            }
            else
            {
                //Defer exception creation until later
                return Tuple.Create(cp, new Quantile[] { }, "No values in Heatmap from which to generate a histogram");
            }

            float hMax = (float)distinct[distinct.Length - 1];
            if (trace != null) trace(5, String.Format("Requesting Histogram generation with Min = {0} and Max = {1} and number of distinct values = {2}", hMin, hMax, distinct.Count()));

            var allColors = _dataService.GetColors(int.MaxValue);
            float[] valuesToIgnore = ignoreZero ? new float[] { 0, (float)rasterBand.NoDataValue } : new float[] { (float)rasterBand.NoDataValue };
            var quantiles = Histogram.GetHistogram((float)hMin, (float)hMax, rasterBand.RasterData.Select(v => (float)v).ToArray(), allColors.Length, valuesToIgnore, true, trace);
            //var colors = new[] { 0xF92525, 0xF96C25, 0xF98F25, 0xF99625, 0xF9A425, 0xF9C825, 0xF9D625, 0xF9F225, 0xEBF925, 0xE4F925, 0xCFF925, 0xB2F925, 0x88F925, 0x48FD1B, 0x1BFD22, 0x04FF58 };
            if (quantiles.Length > 0)
            {
                var colors = allColors.SelectValues(quantiles.Length).ToArray();
                var j = 0;
                for (int i = 0; i < quantiles.Length; i++)
                {
                    var q = quantiles[i];
                    Color colorToAssign = Color.FromArgb(230, Color.FromArgb(colors[j]));
                    quantiles[i].Color = Color.FromArgb(230, Color.FromArgb(colors[j]));
                    cp.AddColorBand(q.Min, q.Max, j, colorToAssign, q);
                    j++;
                }
            }
            cp.PaletteType = RtColorPalette.ColorPaletteType.Range;
            cp.MissingBandColor = Color.Transparent;

            return Tuple.Create(cp, quantiles, "Success");
        }

    }
}
