using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.NPP;

namespace GrabCutNPP
{
	public partial class Form1 : Form
	{
		CudaContext ctx;
		GrabCut grabcut;
		Bitmap bmp_src;
		Bitmap bmp_mask;
		Bitmap bmp_res;
		int width, height;
		NppiRect selection = new NppiRect();
		NppiRect org_selection;
		NPPImage_8uC4 npp_bmp_src;
		NPPImage_8uC4 npp_bmp_res;
		NPPImage_8uC1 npp_bmp_mask;
		CudaPitchedDeviceVariable<uchar4> d_bmp_src;
		CudaPitchedDeviceVariable<uchar4> d_bmp_res;
		CudaPitchedDeviceVariable<byte> d_bmp_mask;

		MouseButtons pressed = MouseButtons.None;
		int[] marker;
		int clickedCorner;
		Point pos;


		public Form1()
		{
			InitializeComponent();

			ctx = new CudaContext(CudaContext.GetMaxGflopsDeviceId());
		}

		private void SetPalette(Bitmap bmp)
		{
			ColorPalette pal = bmp.Palette;
			//for (int i = 0; i < 256; i++)
			//{
			//    pal.Entries[i] = Color.FromArgb(255, i, i, i);
			//}
			pal.Entries[0] = Color.Black;
			pal.Entries[1] = Color.White;
			bmp.Palette = pal;
		}

		private void btn_openImg_Click(object sender, EventArgs e)
		{
			OpenFileDialog ofd = new OpenFileDialog();
			ofd.Filter = "Images|*.bmp;*.jpg;*.jpeg;*.tiff;*.tif;*.png;*.gif";
			if (ofd.ShowDialog() != System.Windows.Forms.DialogResult.OK) return;

			bmp_src = new Bitmap(ofd.FileName);
			
			if (bmp_src.PixelFormat != PixelFormat.Format24bppRgb)
			{
				MessageBox.Show("Only 24-bit RGB images are supported!");
				bmp_src = null;
				bmp_mask = null;
				bmp_res = null;
				if (npp_bmp_src != null) npp_bmp_src.Dispose();
				if (npp_bmp_res != null) npp_bmp_res.Dispose();
				if (npp_bmp_mask != null) npp_bmp_mask.Dispose();
				if (d_bmp_src != null) d_bmp_src.Dispose();
				if (d_bmp_res != null) d_bmp_res.Dispose();
				if (d_bmp_mask != null) d_bmp_mask.Dispose();
				return;
			}
			width = bmp_src.Width;
			height = bmp_src.Height;
			marker = new int[width * height];
			bmp_res = new Bitmap(width, height, PixelFormat.Format32bppArgb);
			bmp_mask = new Bitmap(width, height, PixelFormat.Format8bppIndexed);
			SetPalette(bmp_mask);
			pictureBox_src.Image = bmp_src;

			selection.x = (int)Math.Ceiling(width * 0.1);
			selection.y = (int)Math.Ceiling(height * 0.1);
			selection.width = width - 2 * selection.x;
			selection.height = height - 2 * selection.y;

			if (npp_bmp_src != null) npp_bmp_src.Dispose();
			if (npp_bmp_res != null) npp_bmp_res.Dispose();
			if (npp_bmp_mask != null) npp_bmp_mask.Dispose();
			if (d_bmp_src != null) d_bmp_src.Dispose();
			if (d_bmp_res != null) d_bmp_res.Dispose();
			if (d_bmp_mask != null) d_bmp_mask.Dispose();

			NPPImage_8uC3 npp_temp = new NPPImage_8uC3(width, height);
			CudaPitchedDeviceVariable<uchar3> d_bmp_temp = new CudaPitchedDeviceVariable<uchar3>(npp_temp.DevicePointer, width, height, npp_temp.Pitch);
			npp_temp.CopyToDevice(bmp_src);

			npp_bmp_src = new NPPImage_8uC4(width, height);
			npp_bmp_res = new NPPImage_8uC4(width, height);
			npp_bmp_mask = new NPPImage_8uC1(width, height);
			d_bmp_src = new CudaPitchedDeviceVariable<uchar4>(npp_bmp_src.DevicePointer, width, height, npp_bmp_src.Pitch);
			d_bmp_res = new CudaPitchedDeviceVariable<uchar4>(npp_bmp_res.DevicePointer, width, height, npp_bmp_res.Pitch);
			d_bmp_mask = new CudaPitchedDeviceVariable<byte>(npp_bmp_mask.DevicePointer, width, height, npp_bmp_mask.Pitch);

			grabcut = new GrabCut(d_bmp_src, d_bmp_mask, width, height);
			grabcut.grabCutUtils.convertRGBToRGBA(d_bmp_src, d_bmp_temp, width, height);
			d_bmp_temp.Dispose();
			npp_temp.Dispose();

		}

		private void pictureBox_src_MouseEnter(object sender, EventArgs e)
		{
			pressed = MouseButtons.None;
		}

		private void pictureBox_src_MouseLeave(object sender, EventArgs e)
		{
			pressed = MouseButtons.None;
		}

		private void pictureBox_src_MouseDown(object sender, MouseEventArgs e)
		{
			pressed = e.Button;
			clickedCorner = GetCorner(e.X, e.Y);
			if (clickedCorner > 0)
			{
				org_selection = selection;
				pos = new Point(e.X, e.Y);
			}

		}

		private void pictureBox_src_MouseUp(object sender, MouseEventArgs e)
		{
			pressed = MouseButtons.None;
			clickedCorner = 0;
		}

		private enum Direction
		{
			Positive,
			Negative,
			None
		}

		private double GetDistance(int x1, int y1, int x2, int y2)
		{
			return Math.Sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
		}

		private Direction CheckDirection(int x, int y)
		{
			if (GetDistance(x, y, selection.x, selection.y) < 5)
				return Direction.Negative;
			if (GetDistance(x, y, selection.x + selection.width, selection.y + selection.height) < 5)
				return Direction.Negative;

			if (GetDistance(x, y, selection.x, selection.y + selection.height) < 5)
				return Direction.Positive;
			if (GetDistance(x, y, selection.x + selection.width, selection.y) < 5)
				return Direction.Positive;

			return Direction.None;
		}

		private int GetCorner(int x, int y)
		{
			if (GetDistance(x, y, selection.x, selection.y) < 5)
				return 1;
			if (GetDistance(x, y, selection.x + selection.width, selection.y + selection.height) < 5)
				return 4;

			if (GetDistance(x, y, selection.x, selection.y + selection.height) < 5)
				return 3;
			if (GetDistance(x, y, selection.x + selection.width, selection.y) < 5)
				return 2;
			return 0;
		}

		private void pictureBox_src_MouseMove(object sender, MouseEventArgs e)
		{
			Direction dir = CheckDirection(e.X, e.Y);
			if (dir == Direction.Negative) Cursor.Current = Cursors.SizeNWSE;
			else
				if (dir == Direction.Positive) Cursor.Current = Cursors.SizeNESW;
				else Cursor.Current = Cursors.Default;


			if (pressed == System.Windows.Forms.MouseButtons.Left)
			{
				NppiRect newSelection = new NppiRect(selection.x, selection.y, selection.width, selection.height);
				switch (clickedCorner)
				{
					case 1:
						newSelection.x = org_selection.x - (pos.X - e.X);
						newSelection.y = org_selection.y - (pos.Y - e.Y);
						newSelection.width = org_selection.width + (pos.X - e.X);
						newSelection.height = org_selection.height + (pos.Y - e.Y);
						break;
					case 2:
						newSelection.y = org_selection.y - (pos.Y - e.Y);
						newSelection.width = org_selection.width - (pos.X - e.X);
						newSelection.height = org_selection.height + (pos.Y - e.Y);
						break;
					case 3:
						newSelection.x = org_selection.x - (pos.X - e.X);
						newSelection.width = org_selection.width + (pos.X - e.X);
						newSelection.height = org_selection.height - (pos.Y - e.Y);
						break;
					case 4:
						newSelection.width = org_selection.width - (pos.X - e.X);
						newSelection.height = org_selection.height - (pos.Y - e.Y);
						break;
				}
				if (clickedCorner > 0)
				{
					if (newSelection.width > 20 && newSelection.height > 20)
					{ 
						if (newSelection.x >= 0 && newSelection.y >= 0)
							if (newSelection.x + newSelection.width <= bmp_src.Width && newSelection.y + newSelection.height <= bmp_src.Height)
							{
								selection = newSelection;
								pictureBox_src.Invalidate();
							}
					}
				}
			}
		}


		private void button3_Click(object sender, EventArgs e)
		{
			if (grabcut == null) return; 

			grabcut.grabCutUtils.TrimapFromRect(d_bmp_mask, selection, width, height);

			grabcut.computeSegmentationFromTrimap();
			
			NPPImage_8uC1 alphamap = new NPPImage_8uC1(grabcut.AlphaMap.DevicePointer, grabcut.AlphaMap.Width, grabcut.AlphaMap.Height, grabcut.AlphaMap.Pitch);
			alphamap.CopyToHost(bmp_mask);


			pictureBox_Mask.Image = bmp_mask;

			int mode = 0;
			if (rb_Masked.Checked) mode = 1;
			if (rb_CutOut.Checked) mode = 2;

			grabcut.grabCutUtils.ApplyMatte(mode, d_bmp_res, d_bmp_src, grabcut.AlphaMap, width, height);
			npp_bmp_res.CopyToHost(bmp_res);

			pictureBox_Result.Image = bmp_res;

			lbl_Iterations.Text = grabcut.Iterations.ToString();
			lbl_runtime.Text = grabcut.Runtime.ToString() + " [ms]";
		}

		private void pictureBox_src_Paint(object sender, PaintEventArgs e)
		{
			if (pictureBox_src.Image == null || selection.width == 0 || selection.height == 0) return;

			Pen red = new Pen(Color.Red, 2);
			Brush blue = new SolidBrush(Color.LightBlue);
			Rectangle rect = new Rectangle(selection.x, selection.y, selection.width, selection.height);
			e.Graphics.DrawRectangle(red, rect);

			Rectangle[] corners = new Rectangle[4];
			corners[0] = new Rectangle(selection.x - 3, selection.y - 2, 6, 6);
			corners[1] = new Rectangle(selection.x - 3 + selection.width, selection.y - 3, 6, 6);
			corners[2] = new Rectangle(selection.x - 3 + selection.width, selection.y - 3 + selection.height, 6, 6);
			corners[3] = new Rectangle(selection.x - 3, selection.y - 3 + selection.height, 6, 6);

			e.Graphics.FillRectangles(blue, corners);
		}

		private void button1_Click(object sender, EventArgs e)
		{
			if (grabcut == null) return;

			int mode = 0;
			if (rb_Masked.Checked) mode = 1;
			if (rb_CutOut.Checked) mode = 2;

			grabcut.grabCutUtils.ApplyMatte(mode, d_bmp_res, d_bmp_src, grabcut.AlphaMap, width, height);
			npp_bmp_res.CopyToHost(bmp_res);

			pictureBox_Result.Image = bmp_res;
		}
	}
}
