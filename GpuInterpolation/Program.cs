/*
 * This code is based on code from the NVIDIA CUDA SDK. (Ported from C++ to C# using managedCUDA)
 * This software contains source code provided by NVIDIA Corporation.
 *
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;
using System.IO;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using WilburEllis.RisingTide.Base;
using WilburEllis.RisingTide.Common.DBLayer;
using WilburEllis.RisingTide.Services.ControllerWindowsService;
using WilburEllis.RisingTide.Common.Models;
using WilburEllis.RisingTide.Spatial;
using WilburEllis.RisingTide.MachineIO;

namespace GpuInterpolation
{
	class Program
	{
        //1862, 649, 1852, 648, 1859, 1854, 1853, 1861, 1855, 1858, 1857, 1856, 1863, 1860
        const string dbConnectionString = @"server=127.0.0.1;port=5432;database=RisingTide;user id=postgres;password=Aditi01*;enlist=true;pooling=false;minpoolsize=1;maxpoolsize=100;timeout=150;";// databaseName="RisingTide" isDefault="true"
        static IControllerDataService _dbService; 
		static void Main(string[] args)
		{
            _dbService = new ControllerDataService(dbConnectionString);
            var interpolater = new CalculateHeatmap();
            Console.WriteLine("Retrieve Yield Report Data id = {0} and Boundary data from Database", 1862);
            YieldReportData yieldData = _dbService.GetYieldReport(649, false, false);
            var boundary = _dbService.GetFieldBoundary(yieldData.FieldId);
            //var data = yieldData.YieldData.Yield.data;
            Console.WriteLine("Call GPU interpolation function");
            var raster = interpolater.RunGpuInterpolation(yieldData, boundary);
            interpolater.SaveResults(raster, _dbService, @"C:\AgVerdict\Gpu\GpuRaster.png");
            Console.WriteLine("Saving output to image file");

            Console.WriteLine("Call CPU interpolation function");
            raster = interpolater.RunCpuInterpolation(yieldData, boundary);
            interpolater.SaveResults(raster, _dbService, @"C:\AgVerdict\Gpu\CpuRaster.png");
            Console.WriteLine("Saving output to image file");
            Console.WriteLine("press any key to exit ...");
            Console.ReadKey();
		}	
		
	}
}
