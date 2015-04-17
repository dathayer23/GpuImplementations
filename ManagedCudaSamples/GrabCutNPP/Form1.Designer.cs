namespace GrabCutNPP
{
	partial class Form1
	{
		/// <summary>
		/// Erforderliche Designervariable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Verwendete Ressourcen bereinigen.
		/// </summary>
		/// <param name="disposing">True, wenn verwaltete Ressourcen gelöscht werden sollen; andernfalls False.</param>
		protected override void Dispose(bool disposing)
		{
			if (disposing && (components != null))
			{
				components.Dispose();
			}
			base.Dispose(disposing);
		}

		#region Vom Windows Form-Designer generierter Code

		/// <summary>
		/// Erforderliche Methode für die Designerunterstützung.
		/// Der Inhalt der Methode darf nicht mit dem Code-Editor geändert werden.
		/// </summary>
		private void InitializeComponent()
		{
			this.btn_openImg = new System.Windows.Forms.Button();
			this.pictureBox_src = new System.Windows.Forms.PictureBox();
			this.pictureBox_Mask = new System.Windows.Forms.PictureBox();
			this.pictureBox_Result = new System.Windows.Forms.PictureBox();
			this.btn_Calc = new System.Windows.Forms.Button();
			this.ViewMode = new System.Windows.Forms.GroupBox();
			this.rb_CutOut = new System.Windows.Forms.RadioButton();
			this.rb_Masked = new System.Windows.Forms.RadioButton();
			this.rb_Original = new System.Windows.Forms.RadioButton();
			this.btn_Show = new System.Windows.Forms.Button();
			this.label1 = new System.Windows.Forms.Label();
			this.label2 = new System.Windows.Forms.Label();
			this.lbl_Iterations = new System.Windows.Forms.Label();
			this.lbl_runtime = new System.Windows.Forms.Label();
			((System.ComponentModel.ISupportInitialize)(this.pictureBox_src)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.pictureBox_Mask)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.pictureBox_Result)).BeginInit();
			this.ViewMode.SuspendLayout();
			this.SuspendLayout();
			// 
			// btn_openImg
			// 
			this.btn_openImg.Location = new System.Drawing.Point(12, 468);
			this.btn_openImg.Name = "btn_openImg";
			this.btn_openImg.Size = new System.Drawing.Size(75, 23);
			this.btn_openImg.TabIndex = 0;
			this.btn_openImg.Text = "Open Image";
			this.btn_openImg.UseVisualStyleBackColor = true;
			this.btn_openImg.Click += new System.EventHandler(this.btn_openImg_Click);
			// 
			// pictureBox_src
			// 
			this.pictureBox_src.Location = new System.Drawing.Point(12, 12);
			this.pictureBox_src.Name = "pictureBox_src";
			this.pictureBox_src.Size = new System.Drawing.Size(600, 450);
			this.pictureBox_src.TabIndex = 1;
			this.pictureBox_src.TabStop = false;
			this.pictureBox_src.Paint += new System.Windows.Forms.PaintEventHandler(this.pictureBox_src_Paint);
			this.pictureBox_src.MouseDown += new System.Windows.Forms.MouseEventHandler(this.pictureBox_src_MouseDown);
			this.pictureBox_src.MouseEnter += new System.EventHandler(this.pictureBox_src_MouseEnter);
			this.pictureBox_src.MouseLeave += new System.EventHandler(this.pictureBox_src_MouseLeave);
			this.pictureBox_src.MouseMove += new System.Windows.Forms.MouseEventHandler(this.pictureBox_src_MouseMove);
			this.pictureBox_src.MouseUp += new System.Windows.Forms.MouseEventHandler(this.pictureBox_src_MouseUp);
			// 
			// pictureBox_Mask
			// 
			this.pictureBox_Mask.Location = new System.Drawing.Point(618, 468);
			this.pictureBox_Mask.Name = "pictureBox_Mask";
			this.pictureBox_Mask.Size = new System.Drawing.Size(600, 450);
			this.pictureBox_Mask.TabIndex = 2;
			this.pictureBox_Mask.TabStop = false;
			// 
			// pictureBox_Result
			// 
			this.pictureBox_Result.Location = new System.Drawing.Point(618, 12);
			this.pictureBox_Result.Name = "pictureBox_Result";
			this.pictureBox_Result.Size = new System.Drawing.Size(600, 450);
			this.pictureBox_Result.TabIndex = 2;
			this.pictureBox_Result.TabStop = false;
			// 
			// btn_Calc
			// 
			this.btn_Calc.Location = new System.Drawing.Point(93, 468);
			this.btn_Calc.Name = "btn_Calc";
			this.btn_Calc.Size = new System.Drawing.Size(118, 23);
			this.btn_Calc.TabIndex = 5;
			this.btn_Calc.Text = "Calculate mask";
			this.btn_Calc.UseVisualStyleBackColor = true;
			this.btn_Calc.Click += new System.EventHandler(this.button3_Click);
			// 
			// ViewMode
			// 
			this.ViewMode.Controls.Add(this.rb_CutOut);
			this.ViewMode.Controls.Add(this.rb_Masked);
			this.ViewMode.Controls.Add(this.rb_Original);
			this.ViewMode.Controls.Add(this.btn_Show);
			this.ViewMode.Location = new System.Drawing.Point(12, 497);
			this.ViewMode.Name = "ViewMode";
			this.ViewMode.Size = new System.Drawing.Size(92, 124);
			this.ViewMode.TabIndex = 7;
			this.ViewMode.TabStop = false;
			this.ViewMode.Text = "View mode";
			// 
			// rb_CutOut
			// 
			this.rb_CutOut.AutoSize = true;
			this.rb_CutOut.Location = new System.Drawing.Point(12, 65);
			this.rb_CutOut.Name = "rb_CutOut";
			this.rb_CutOut.Size = new System.Drawing.Size(59, 17);
			this.rb_CutOut.TabIndex = 2;
			this.rb_CutOut.Text = "Cut out";
			this.rb_CutOut.UseVisualStyleBackColor = true;
			// 
			// rb_Masked
			// 
			this.rb_Masked.AutoSize = true;
			this.rb_Masked.Checked = true;
			this.rb_Masked.Location = new System.Drawing.Point(12, 42);
			this.rb_Masked.Name = "rb_Masked";
			this.rb_Masked.Size = new System.Drawing.Size(63, 17);
			this.rb_Masked.TabIndex = 1;
			this.rb_Masked.TabStop = true;
			this.rb_Masked.Text = "Masked";
			this.rb_Masked.UseVisualStyleBackColor = true;
			// 
			// rb_Original
			// 
			this.rb_Original.AutoSize = true;
			this.rb_Original.Location = new System.Drawing.Point(12, 19);
			this.rb_Original.Name = "rb_Original";
			this.rb_Original.Size = new System.Drawing.Size(60, 17);
			this.rb_Original.TabIndex = 0;
			this.rb_Original.Text = "Original";
			this.rb_Original.UseVisualStyleBackColor = true;
			// 
			// btn_Show
			// 
			this.btn_Show.Location = new System.Drawing.Point(6, 88);
			this.btn_Show.Name = "btn_Show";
			this.btn_Show.Size = new System.Drawing.Size(75, 23);
			this.btn_Show.TabIndex = 8;
			this.btn_Show.Text = "Show";
			this.btn_Show.UseVisualStyleBackColor = true;
			this.btn_Show.Click += new System.EventHandler(this.button1_Click);
			// 
			// label1
			// 
			this.label1.AutoSize = true;
			this.label1.Location = new System.Drawing.Point(120, 516);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(56, 13);
			this.label1.TabIndex = 9;
			this.label1.Text = "Iterations: ";
			this.label1.TextAlign = System.Drawing.ContentAlignment.TopRight;
			// 
			// label2
			// 
			this.label2.AutoSize = true;
			this.label2.Location = new System.Drawing.Point(120, 536);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(49, 13);
			this.label2.TabIndex = 10;
			this.label2.Text = "Runtime:";
			this.label2.TextAlign = System.Drawing.ContentAlignment.TopRight;
			// 
			// lbl_Iterations
			// 
			this.lbl_Iterations.AutoSize = true;
			this.lbl_Iterations.Location = new System.Drawing.Point(182, 516);
			this.lbl_Iterations.Name = "lbl_Iterations";
			this.lbl_Iterations.Size = new System.Drawing.Size(13, 13);
			this.lbl_Iterations.TabIndex = 11;
			this.lbl_Iterations.Text = "0";
			// 
			// lbl_runtime
			// 
			this.lbl_runtime.AutoSize = true;
			this.lbl_runtime.Location = new System.Drawing.Point(182, 536);
			this.lbl_runtime.Name = "lbl_runtime";
			this.lbl_runtime.Size = new System.Drawing.Size(13, 13);
			this.lbl_runtime.TabIndex = 12;
			this.lbl_runtime.Text = "0";
			// 
			// Form1
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(1226, 927);
			this.Controls.Add(this.lbl_runtime);
			this.Controls.Add(this.lbl_Iterations);
			this.Controls.Add(this.label2);
			this.Controls.Add(this.label1);
			this.Controls.Add(this.ViewMode);
			this.Controls.Add(this.btn_Calc);
			this.Controls.Add(this.pictureBox_Result);
			this.Controls.Add(this.pictureBox_Mask);
			this.Controls.Add(this.pictureBox_src);
			this.Controls.Add(this.btn_openImg);
			this.Name = "Form1";
			this.Text = "NPP GrabCut";
			((System.ComponentModel.ISupportInitialize)(this.pictureBox_src)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.pictureBox_Mask)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.pictureBox_Result)).EndInit();
			this.ViewMode.ResumeLayout(false);
			this.ViewMode.PerformLayout();
			this.ResumeLayout(false);
			this.PerformLayout();

		}

		#endregion

		private System.Windows.Forms.Button btn_openImg;
		private System.Windows.Forms.PictureBox pictureBox_src;
		private System.Windows.Forms.PictureBox pictureBox_Mask;
		private System.Windows.Forms.PictureBox pictureBox_Result;
		private System.Windows.Forms.Button btn_Calc;
		private System.Windows.Forms.GroupBox ViewMode;
		private System.Windows.Forms.RadioButton rb_CutOut;
		private System.Windows.Forms.RadioButton rb_Masked;
		private System.Windows.Forms.RadioButton rb_Original;
		private System.Windows.Forms.Button btn_Show;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.Label lbl_Iterations;
		private System.Windows.Forms.Label lbl_runtime;
	}
}

