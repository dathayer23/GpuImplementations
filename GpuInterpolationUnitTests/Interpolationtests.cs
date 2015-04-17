using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using GpuInterpolation;
using WilburEllis.RisingTide.Base;
using WilburEllis.RisingTide.Common.DBLayer;
using WilburEllis.RisingTide.Services.ControllerWindowsService;
using WilburEllis.RisingTide.Common.Models;
using WilburEllis.RisingTide.Spatial;
using WilburEllis.RisingTide.MachineIO;

namespace GpuInterpolationUnitTests
{
    //valid report ids 1862, 649, 1852, 648, 1859, 1854, 1853, 1861, 1855, 1858, 1857, 1856, 1863, 1860
    [TestClass]
    public class Interpolationtests
    {
        const string dbConnectionString = @"server=127.0.0.1;port=5432;database=RisingTide;user id=postgres;password=Aditi01*;enlist=true;pooling=false;minpoolsize=1;maxpoolsize=100;timeout=150;";// databaseName="RisingTide" isDefault="true"
        IControllerDataService _dbService; 
        [TestMethod]
        public void TestMethod1()
        {
            _dbService = new ControllerDataService(dbConnectionString);
            var interpolater = new CalculateHeatmap();
            YieldReportData yieldData = _dbService.GetYieldReport(1862, false, false);
            var boundary = _dbService.GetFieldBoundary(yieldData.FieldId);
            //var data = yieldData.YieldData.Yield.data;
            interpolater.RunGpuInterpolation(yieldData, boundary);
        }
    }
}
