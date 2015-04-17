using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda.VectorTypes;

namespace GpuInterpolation
{
    /// <summary>
    /// double3
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct double3 : ICudaVectorType
    {
        /// <summary>
        /// X
        /// </summary>
        public double x;
        /// <summary>
        /// Y
        /// </summary>
        public double y;

        public double z;

        #region Operator Methods
        /// <summary>
        /// per element Add
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 Add(double3 src, double3 value)
        {
            double3 ret = new double3(src.x + value.x, src.y + value.y, src.z + value.z);
            return ret;
        }

        /// <summary>
        /// per element Add
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 Add(double3 src, double value)
        {
            double3 ret = new double3(src.x + value, src.y + value, src.z + value);
            return ret;
        }

        /// <summary>
        /// per element Add
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 Add(double src, double3 value)
        {
            double3 ret = new double3(src + value.x, src + value.y, src + value.z);
            return ret;
        }

        /// <summary>
        /// per element Substract
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 Subtract(double3 src, double3 value)
        {
            double3 ret = new double3(src.x - value.x, src.y - value.y, src.z - value.z);
            return ret;
        }

        /// <summary>
        /// per element Substract
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 Subtract(double3 src, double value)
        {
            double3 ret = new double3(src.x - value, src.y - value, src.z - value);
            return ret;
        }

        /// <summary>
        /// per element Substract
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 Subtract(double src, double3 value)
        {
            double3 ret = new double3(src - value.x, src - value.y, src - value.z);
            return ret;
        }

        /// <summary>
        /// per element Multiply
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 Multiply(double3 src, double3 value)
        {
            double3 ret = new double3(src.x * value.x, src.y * value.y, src.z * value.z);
            return ret;
        }

        /// <summary>
        /// per element Multiply
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 Multiply(double3 src, double value)
        {
            double3 ret = new double3(src.x * value, src.y * value, src.z * value);
            return ret;
        }

        /// <summary>
        /// per element Multiply
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 Multiply(double src, double3 value)
        {
            double3 ret = new double3(src * value.x, src * value.y, src * value.z);
            return ret;
        }

        /// <summary>
        /// per element Divide
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 Divide(double3 src, double3 value)
        {
            double3 ret = new double3(src.x / value.x, src.y / value.y, src.z / value.z);
            return ret;
        }

        /// <summary>
        /// per element Divide
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 Divide(double3 src, double value)
        {
            double3 ret = new double3(src.x / value, src.y / value, src.z / value);
            return ret;
        }

        /// <summary>
        /// per element Divide
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 Divide(double src, double3 value)
        {
            double3 ret = new double3(src / value.x, src / value.y, src/value.z);
            return ret;
        }
        #endregion

        #region operators
        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 operator +(double3 src, double3 value)
        {
            return Add(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 operator +(double3 src, double value)
        {
            return Add(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 operator +(double src, double3 value)
        {
            return Add(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 operator -(double3 src, double3 value)
        {
            return Subtract(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 operator -(double3 src, double value)
        {
            return Subtract(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 operator -(double src, double3 value)
        {
            return Subtract(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 operator *(double3 src, double3 value)
        {
            return Multiply(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 operator *(double3 src, double value)
        {
            return Multiply(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 operator *(double src, double3 value)
        {
            return Multiply(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 operator /(double3 src, double3 value)
        {
            return Divide(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 operator /(double3 src, double value)
        {
            return Divide(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 operator /(double src, double3 value)
        {
            return Divide(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static bool operator ==(double3 src, double3 value)
        {
            if (object.ReferenceEquals(src, value)) return true;
            return src.Equals(value);
        }
        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static bool operator !=(double3 src, double3 value)
        {
            return !(src == value);
        }
        #endregion

        #region Override Methods
        /// <summary>
        /// 
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            if (obj == null) return false;
            if (!(obj is double3)) return false;

            double3 value = (double3)obj;

            bool ret = true;
            ret &= this.x == value.x;
            ret &= this.y == value.y;
            ret &= this.z == value.z;
            return ret;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public bool Equals(double3 value)
        {
            bool ret = true;
            ret &= this.x == value.x;
            ret &= this.y == value.y;
            ret &= this.z == value.z;
            return ret;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return string.Format(CultureInfo.CurrentCulture, "({0}, {1}, {2})", this.x, this.y, this.z);
        }
        #endregion

        #region constructors
        /// <summary>
        /// 
        /// </summary>
        /// <param name="xValue"></param>
        /// <param name="yValue"></param>
        public double3(double xValue, double yValue, double zValue)
        {
            this.x = xValue;
            this.y = yValue;
            this.z = zValue;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="val"></param>
        public double3(double val)
        {
            this.x = val;
            this.y = val;
            this.z = val;
        }
        #endregion

        #region Methods
        /// <summary>
        /// Component wise minimum as the CUDA function fminf
        /// </summary>
        /// <param name="aValue"></param>
        /// <param name="bValue"></param>
        /// <returns></returns>
        public static double3 Min(double3 aValue, double3 bValue)
        {
            return new double3(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y), Math.Min(aValue.z, bValue.z));
        }

        /// <summary>
        /// Component wise maximum as the CUDA function fmaxf
        /// </summary>
        /// <param name="aValue"></param>
        /// <param name="bValue"></param>
        /// <returns></returns>
        public static double3 Max(double3 aValue, double3 bValue)
        {
            return new double3(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y), Math.Max(aValue.y, bValue.y));
        }
        #endregion

        #region SizeOf
        /// <summary>
        /// Gives the size of this type in bytes. <para/>
        /// Is equal to <c>Marshal.SizeOf(double3);</c>
        /// </summary>
        public static uint SizeOf
        {
            get
            {
                return (uint)Marshal.SizeOf(typeof(double3));
            }
        }

        /// <summary>
        /// Gives the size of this type in bytes. <para/>
        /// Is equal to <c>Marshal.SizeOf(this);</c>
        /// </summary>
        public uint Size
        {
            get
            {
                return (uint)Marshal.SizeOf(this);
            }
        }
        #endregion
    }
}
