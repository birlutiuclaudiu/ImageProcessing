// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
using namespace cv;
using namespace std;
void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		cv::imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = MAX_PATH - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // no dword alignment is done !!!
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar* lpSrc = src.data;
		uchar* lpDst = dst.data;
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i * w + j];
				lpDst[i * w + j] = 255 - val;
				/* sau puteti scrie:
				uchar val = lpSrc[i*width + j];
				lpDst[i*width + j] = 255 - val;
				//	w = width pt. imagini cu 8 biti / pixel
				//	w = 3*width pt. imagini cu 24 biti / pixel
				*/
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // latimea in octeti a unei linii de imagine

		Mat dstH = Mat(height, width, CV_8UC1);
		Mat dstS = Mat(height, width, CV_8UC1);
		Mat dstV = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* dstDataPtrH = dstH.data;
		uchar* dstDataPtrS = dstS.data;
		uchar* dstDataPtrV = dstV.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);
		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				// sau int hi = i*w + j * 3;	//w = 3*width pt. imagini 24 biti/pixel
				int gi = i * width + j;

				dstDataPtrH[gi] = hsvDataPtr[hi] * 510 / 360;		// H = 0 .. 255
				dstDataPtrS[gi] = hsvDataPtr[hi + 1];			// S = 0 .. 255
				dstDataPtrV[gi] = hsvDataPtr[hi + 2];			// V = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", dstH);
		imshow("S", dstS);
		imshow("V", dstV);
		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int k = 0.4;
		int pH = 50;
		int pL = k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey();
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey();  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src1 = (Mat*)param;
	Mat src = (*src1);
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",x, y,(int)(src).at<Vec3b>(y, x)[2],(int)(src).at<Vec3b>(y, x)[1],(int)(src).at<Vec3b>(y, x)[0]);
		Vec3b color = src.at<Vec3b>(y, x);
		Mat dst = src.clone();
		Mat proj = Mat::zeros(src.size(), CV_8UC1);
		int height = src.rows;
		int width = src.cols;
		int* h = (int*)calloc(sizeof(int), height);
		int *V = (int*)calloc(sizeof(int), width);
		if (h == NULL || V == NULL) {
			exit(1);
		}

		//calcul parametri 
		int A = 0; int xc = 0; int yc = 0; int P = 0; int i, j;
		int positionXc = 0; int positionYc = 0;
		//parcugere imagine
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				Vec3b crt_color = src.at<Vec3b>(i, j);
				if (crt_color == color) {
					A++; xc +=j; yc += i; positionXc ++; positionYc++;
					if ((j - 1) >= 0 && src.at<Vec3b>(i, j - 1) != color ||
						(j + 1) < width && src.at<Vec3b>(i, j + 1) != color ||
						(i - 1) >= 0 && src.at<Vec3b>(i - 1, j) != color ||
						(i + 1) < height && src.at<Vec3b>(i + 1, j) != color ||
						(i - 1) >= 0 && (j - 1) >= 0 && src.at<Vec3b>(i - 1, j - 1) != color ||
						(i - 1) >= 0 && (j + 1) < width && src.at<Vec3b>(i - 1, j + 1) != color ||
						(i + 1) < height && (j - 1) >= 0 && src.at<Vec3b>(i + 1, j - 1) != color ||
						(i + 1) < height && (j + 1) < width && src.at<Vec3b>(i + 1, j + 1) != color) {

						P++;
						dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
					}
						
					h[i]++;
					V[j] ++;
				}
			}
		}

		printf("Aria: %d ", A);
		int  xcnorm = xc / A; 
		int  ycnorm = yc / A;
		printf("Xc=%d, XcNorm=%d, yc=%d, ycNorm=%d, Perimeter: %d\n", xc, xcnorm, yc, ycnorm,P);
		//Pentru axa de alungire
		int nr = 0;
		int num1 = 0; int num2 = 0;
		
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				Vec3b crt_color = src.at<Vec3b>(i, j);
				if (crt_color == color) {
					nr = nr + (i - ycnorm)*(j - xcnorm);
					num1 = num1 + (j - xcnorm)*(j - xcnorm);
					num2 = num2 + (i - ycnorm) * (i - ycnorm);
				}
			}
		}
		//Calcul teta
		double teta2 = atan2(2 * nr, (double)(num1 - num2));
		double teta = teta2 / 2;
		int grade = teta * 180 / PI;
		printf(" Grade:%d", grade);
		drawMarker(dst, Point(xcnorm, ycnorm), Scalar(0, 0, 0), MARKER_CROSS,10, 1);
		int delta = 30; 
		Point P1 = Point(xcnorm - delta, ycnorm - (int)(delta * tan(teta)));
		Point P2 = Point(xcnorm + delta, ycnorm + (int)(delta * tan(teta)));
		line(dst, P1, P2,Scalar(0,0,0), 1, 8);
	

		imshow("with permiter", dst);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < h[i]; j++) {
				proj.at<uchar>(i, j) = 255;
			}
		}
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < V[i]; j++) {
				proj.at<uchar>(j, i) = 255;
			}
		}
		imshow("projection", proj);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void brightnessAditive(int const additive)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				int newVal = val + additive;
				int neg = min(newVal, 255);
				uchar neg2 = max(neg, 0);
				dst.at<uchar>(i, j) = neg2;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);

		waitKey();
	}
}

void brightnessMultiplicative(int const multiplicative)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				int newVal = val * multiplicative;
				int neg = min(newVal, 255);
				uchar neg2 = max(neg, 0);
				dst.at<uchar>(i, j) = neg2;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void createMatrix_256() {
	Mat matrix = Mat(256, 256, CV_8UC3);
	Vec3b white = Vec3b(255, 255, 255);
	Vec3b red = Vec3b(255, 0, 0);
	Vec3b blue = Vec3b(0, 0, 255);
	Vec3b green = Vec3b(0, 255, 0);

	for (int i = 0; i < 255; i++)
	{
		for (int j = 0; j < 255; j++)
		{
			if (i < 128 && j < 128) {
				matrix.at<Vec3b>(i, j) = white;
			}
			else if (i < 128 && j>127) {
				matrix.at<Vec3b>(i, j) = red;
			}
			else if (i > 127 && j < 128) {
				matrix.at<Vec3b>(i, j) = blue;
			}
			else {
				matrix.at<Vec3b>(i, j) = green;
			}
		}
	}
	imshow("result image", matrix);
	waitKey();
}

void createMatrixFloat() {
	float vals[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 12 };
	Mat matrix = Mat(3, 3, CV_32FC1, vals);
	Mat matrixDst = Mat(3, 3, CV_32FC1);    //nu e nevoie de declarare matrice
	matrixDst = matrix.inv();


	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			printf("%.3f ", matrixDst.at<float>(i, j));
		}
		printf("\n");
	}
	waitKey();
	waitKey(4000);
}

// -------------------------------------------LAB 2--------------------------------------------
void extractRGB() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname,CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;
		//matrice destinatie pentru cele 3 culori
		/*Mat dstB = Mat(height, width, CV_8UC1);
		Mat dstG = Mat(height, width, CV_8UC1);
		Mat dstR = Mat(height, width, CV_8UC1);
		Vec3b aux;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				aux = src.at<Vec3b>(i, j);
				dstB.at<uchar>(i, j) = aux[0];
				dstG.at<uchar>(i, j) = aux[1];
				dstR.at<uchar>(i, j) = aux[2];
			}
		}

		imshow("input image", src);
		imshow("B", dstB);
		imshow("G", dstG);
		imshow("R", dstR);
		waitKey();*/

		Mat channels[3];
		split(src, channels);
		imshow("input image", src);
		imshow("B", channels[0]);
		imshow("G", channels[1]);
		imshow("R", channels[2]);
		waitKey();

	}
}
/*functie pentru conversia din color in grayscale*/
void colorToGrayscale() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;
		

		Mat dst = Mat(height, width, CV_8UC1);
		Vec3b aux;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				aux = src.at<Vec3b>(i, j);
				dst.at<uchar>(i,j) = (aux[0] + aux[1] + aux[2] )/ 3;
			}
		}

		imshow("input image", src);
		imshow("grayscale image", dst);
		waitKey();
	}
}
/*functie pentru binarizarea unei imagini*/
void binarizeImage() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		
		int threshold;
		printf("Write threshold:");
		scanf("%d", &threshold);

		Mat dst = Mat(height, width, CV_8UC1);
		Vec3b aux;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				dst.at<uchar>(i, j) = src.at<uchar>(i, j) < threshold ? 0 : 255;

			}
		}

		imshow("input image", src);
		imshow("white-black image", dst);
		waitKey();
	}
}
/*Metoda pentru calculul lui H, S,V*/
void extractHSV() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;
		//matrice destinatie pentru cele 3 culori
		Mat dstH = Mat(height, width, CV_8UC1);
		Mat dstS = Mat(height, width, CV_8UC1);
		Mat dstV = Mat(height, width, CV_8UC1);
		Vec3b aux;
		float r, g, b, M, m, C, H, S, V;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				aux = src.at<Vec3b>(i, j);
				//normalizare componente R, G, B
				r = (float)aux[2] / 255;
				g = (float)aux[1] / 255;
				b = (float)aux[0] / 255;
				//pentru definitia corecta a macro-ului
				M = max(r, max(g, b));
				m = min(r, min(g, b));
				C = M - m;
				//value
				V = M;
				//saturation
				if (V != 0) {
					S = C / V;
				}
				else {
					S = 0;
				}
				//hue
				if (C != 0) {
					if (M == r) H = 60 * (g - b) / C;
					if (M == g)	H = 120 + 60 * (b - r) / C;
					if (M == b)   H = 240 + 60 * (r - g) / C;
				}
				else {
					H = 0; //grayscale
				}

				if (H < 0)
					H = H + 360;
				//aducerea valorilor in intervelul 0..255
				dstH.at<uchar>(i, j) = (uchar)(H * 255 / 360);
				dstS.at<uchar>(i, j) = (uchar)(S * 255);
				dstV.at<uchar>(i, j) = (uchar)(V * 255);
			}
		}

		imshow("RGB image", src);
		imshow("H", dstH);
		imshow("S", dstS);
		imshow("V", dstV);
		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);
		imshow("HSV image", hsvImg);
		waitKey();
	}
}
bool isInside(Mat img, int i, int j) {
	if (i < 0 || j < 0) return false;
	if (i < img.rows && j < img.cols) return true;
	return false;
}

void testIsInside() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int i, j; 
		printf("i=");
		scanf("%d", &i);
		printf("j=");
		scanf("%d", &j);
		printf("Height=%d Width=%d =>\n", src.rows, src.cols);
		isInside(src, i, j) ? printf("	Is inside (i,j)\n") : printf("	Is not inside (i,j)\n");
		waitKey(5);
	}
}
//////////////////////////////////////////////////LAB3////////////////////
//2 metode overload pentru calcul vectorului a- specific histogramei
void compute_histogram(int* a,int n) {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	int height = src.rows;
	int width = src.cols;
	//intializare vector
	for (int i = 0; i < n; i++) {
		a[i] = 0;
	}
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			a[src.at<uchar>(i, j)] +=1;
		}
	}
	for (int i = 0; i < n; i++)
		printf("%d ", a[i]);
	waitKey(1000);
}
void compute_histogram(int* a,int n, Mat src) {
	
	int height = src.rows;
	int width = src.cols;
	for (int i = 0; i < n; i++) {
		a[i] = 0;
	}
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			a[src.at<uchar>(i, j)] += 1;
		}
	}
}
//2 metode overload pentru calcul vectorului fdp; in care se citeste imaginea in interiorul functiei sau se primeste ca parametru
void compute_FDP(float* p, int n) {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	int a[256];
	compute_histogram(a, 256, src);
	int M = src.rows * src.cols;
	for (int i = 0; i < n; i++) {
		p[i] = (float)a[i] / M;
		printf("%f ", p[i]);
	}
	waitKey(3000);
}
void compute_FDP(float* p, int n, Mat src) {
	int a[256];
	compute_histogram(a, 256, src);
	int M = src.rows * src.cols;
	for (int i = 0; i < n; i++) {
		
		p[i] =(float) ((float)a[i]) / M;
		
	}
}
//functie pentru afisarea histogramei
void showHistogram(const string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 102, 0)); // histogram bins colored in orange
	}

	imshow(name, imgHist);
	waitKey(0);
}

//exercitiu 4; acumulatoare mai mici sau egale cu 256
void lower_histogram_acc() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{	
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height	= src.rows;
		int width	= src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		printf("Write acumulator: ");
		int m;
		scanf("%d", &m);
		if (m < 1 || m>256) {
			printf("Invalid accumulator");
			continue;
		}
		float factor = (float) m / 256;
		int *a = (int*)calloc(m, sizeof(int)); //pentru initializare cu 0 a vectorului de frecventa
		if (!a) {
			printf("Could not allocate memory for frequency array");
		}
		int value;
		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++){
				value = src.at<uchar>(i, j) * factor;
				dst.at<uchar>(i, j) = value;
				
				a[value] += 1;
			}
		}
	
		imshow("imagine", src);
		imshow("imagine_nivele_reduse_gri", dst*16);
		showHistogram("Histograma", a, m, m);
		waitKey();
		free(a);
		a = NULL;

	}	
}

//exercitiu 5
void multiple_thresholds() {
	float fdp[256];
	char fname[MAX_PATH];
	int WH = 5;
	float TH = 0.0003f;   //prag
	float v; //media valorilor
	
	while (openFileDlg(fname)) {
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);
		GaussianBlur(src, src, Size(5, 5), 0.8, 0.8);
		compute_FDP(fdp, 256, src);
		vector< uchar > maxime;   //vector de maxime locale
		maxime.push_back(0);

		for (int k = 0 + WH; k <= 255 - WH; k++) {
			float sum = 0.0f;
			float maxim = 0.0f;
			for (int i = k - WH; i <= k + WH; i++) {
				sum += fdp[i];
				if (maxim < fdp[i]) {
					maxim = fdp[i];
				}
			}
			v = sum / (2 * WH + 1); //fereastra de latime (2 * WH + 1)
			if (fdp[k] > v + TH && fdp[k] >= maxim) {
				maxime.push_back(k);
			}
		}
		maxime.push_back(255);
		
		//determinarea praguri
		vector< uchar > mijloace;   //vector de maxime locale
		mijloace.push_back(0);
		for (int i = 0; i < maxime.size() - 1; i++) {
			mijloace.push_back((maxime[i] + maxime[i + 1]) / 2);
		}
		mijloace.push_back(255);
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				for (int k = 0; k < mijloace.size()-1; k++) {
					if (src.at<uchar>(i, j) >= mijloace[k] && src.at<uchar>(i, j) <= mijloace[k + 1]) {
						if (mijloace[k + 1] - src.at<uchar>(i, j) < src.at<uchar>(i, j) - mijloace[k]) 
							dst.at<uchar>(i, j) = mijloace[k + 1];   //cel mai apropiat mijloc
						else
							dst.at<uchar>(i, j) = mijloace[k];
						break;
					}
				}
			}
		}
		imshow("Imagine sursa", src);
		imshow("Imagine rezultat", dst);
		waitKey(0);
	}
}

void floydSteinberg_alg() {
	float fdp[256];
	char fname[MAX_PATH];
	int WH = 5;
	float TH = 0.0003f;   //prag
	float v; //media valorilor

	while (openFileDlg(fname)) {
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = src.clone();
				//Mat srcAux = src.clone();    //creare clona pentru src pentru a nu fi modificata imaginea initiala
		GaussianBlur(src, src, Size(5, 5), 0.8, 0.8);
		compute_FDP(fdp, 256, src);
		vector< uchar > maxime;   //vector de maxime locale
		maxime.push_back(0);

		for (int k = 0 + WH; k <= 255 - WH; k++) {
			float sum = 0.0f;
			float maxim = 0.0f;
			for (int i = k - WH; i <= k + WH; i++) {
				sum += fdp[i];
				if (maxim < fdp[i]) {
					maxim = fdp[i];
				}
			}
			v = sum / (2 * WH + 1); //fereastra de latime (2 * WH + 1)
			if (fdp[k] > v + TH && fdp[k] >= maxim) {
				maxime.push_back(k);
			}
		}
		maxime.push_back(255);

		//determinarea praguri
		vector< uchar > mijloace;   //vector de maxime locale
		mijloace.push_back(0);
		for (int i = 0; i < maxime.size() - 1; i++) {
			mijloace.push_back((maxime[i] + maxime[i + 1]) / 2);
		}
		mijloace.push_back(255);
		imshow("Imagine sursa", src);
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				for (int k = 0; k < mijloace.size() - 1; k++) {
					uchar val = src.at<uchar>(i, j);
					if (val >= mijloace[k] && val <= mijloace[k + 1]) {
						if (mijloace[k + 1] - val < val - mijloace[k]) {
							src.at<uchar>(i, j) = mijloace[k + 1];   //cel mai apropiat mijloc
							dst.at<uchar>(i, j) = mijloace[k + 1];
						}
						else {
							src.at<uchar>(i, j) = mijloace[k];
							dst.at<uchar>(i, j) = mijloace[k];
						}
						break;
					}
					int error = val - src.at<uchar>(i, j);
					if (isInside(src, i, j + 1))
						src.at<uchar>(i, j+1) = src.at<uchar>(i, j+1) + 7 * error / 16;
					
					if (isInside(src, i+1, j - 1))
						src.at<uchar>(i+1, j - 1) = src.at<uchar>(i+1, j - 1) + 3 * error / 16;
					
					if (isInside(src, i+1, j))
						src.at<uchar>(i+1, j) = src.at<uchar>(i+1, j) + 5 * error / 16;
					
					if (isInside(src, i+1, j + 1))
						src.at<uchar>(i+1, j + 1) = src.at<uchar>(i+1, j + 1) +  error / 16;
				}
			}
		}
		imshow("Cuantizare fara Floyd", dst);
		imshow("Imagine after Floyd", src);
		//imshow("Imagine rezultat", dst);
		waitKey(0);
	}
}
//-----------------------------------------------------------------------------------LABORATOR 4--------------------------------------------------------------------
void lab4() {
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}

}
//////////////////////////////////////////////////////     LAB5      ////////////////////////////////////////////////////////////////////////////////////////////
#include<queue>  //pentru BFS
void labelingAlgorithm(Mat src) {
	double t=(double)getTickCount();
	Scalar colorLUT[1000] = { 0 };
	Scalar color; 
	for (int i = 0; i < 1000; i++) {
		
		colorLUT[i] = Scalar (rand() % 255, rand() % 255, rand() % 255);
	}
	colorLUT[0] = Scalar(0, 0, 0);  //pentru ca fundalul sa fie negru


	Mat labels = Mat::zeros(src.size(), CV_16UC1); //labels matrix
	Mat dst = Mat::zeros(src.size(), CV_8UC3); //destination matrix
	int height = src.rows;
	int width = src.cols;
	
	//for neighbors
	int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 }; // row 
	int di[8] = { 0, -1, -1, -1,  0,  1, 1, 1 }; // col 

	int label = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) == 0 && labels.at<ushort>(i, j) == 0) {
				queue<Point> queue;
				queue.push(Point(j, i));
				labels.at<ushort>(i, j) = label;
				label++;
				while (!queue.empty()) {
					Point oldest = queue.front();
					int jj = oldest.x;
					int ii = oldest.y;
					queue.pop();
					for (int k = 0; k < 8; k++) {
						if (isInside(src, ii + di[k], jj + dj[k])) {
							if (src.at<uchar>(ii + di[k], jj + dj[k])==0 && labels.at<ushort>(ii + di[k], jj + dj[k])==0) {
								queue.push(Point(jj + dj[k],ii + di[k]));
								labels.at<ushort>(ii + di[k], jj + dj[k]) = label;
							}
						}
					}
				}
			}
		}
	}
	
	printf("%d", label);
	for (int i = 0; i < height ; i++) {
		for (int j = 0; j < width ; j++) {
			Scalar color = colorLUT[labels.at<ushort>(i, j)];
			dst.at<Vec3b>(i, j)[0] = color[0];
			dst.at<Vec3b>(i, j)[1] = color[1];
			dst.at<Vec3b>(i, j)[2] = color[2];
		}
	}
	imshow("Sursa", src);
	imshow("Destinatie", dst);

	t = ((double)getTickCount() - t) / getTickFrequency();
	printf("Time = % .2f[ms]\n", t * 1000);
}

void lab5() {
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		//Create a window
		labelingAlgorithm(src);
		//show the image
	

		// Wait until user press some key
		waitKey(0);
	}

}


/// //////////////////////////////////////////////////LAB 6
typedef struct {
	int x, y; 
	byte c; 
	char cd; 
}my_point;

uchar OBIECT = 0;
uchar FUNDAL = 255;
void conturTracing(Mat src) {

	//vecotri pentru imagine 
	int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 };
	int di[8] = { 0, - 1, -1, -1, 0,  1, 1, 1 };
	//vector pentru contur
	vector<my_point> contur;
	int x_start; int y_start;
	bool found_start_pixel = false;
	byte dir; //variabila ce indica di
	//determinarea pixel de inceput contur
	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == OBIECT) {
				found_start_pixel = true;
				x_start = j;
				y_start = i;
				contur.push_back(my_point{ j, i, 7, 7 % 8 });
				dir = 7;
				break;
			}
		}
		if (found_start_pixel)
			break;
	}
	if (!found_start_pixel) {
		printf("Contur not found");
		return;
	}
	//algoritmul de detectie contur
	bool finished = false; 
	int j = x_start; 
	int i = y_start;
	int n = 0; //indexul pixelului de contur curent din vector
	imshow("Imagine sursa", src);
	while (!finished) {
		char prev_dir = dir;
		if (dir % 2 == 0) 
			dir = (dir + 7) % 8;
		else
			dir = (dir + 6) % 8;
		//parcurgere in sens trigonometric
		for (byte k = 0; k < 8; k++) {
			byte d = (dir + k) % 8;
			int x = j + dj[d];
			int y = i + di[d];
			//printf("%d, %d\n", x, y);
			if (isInside(src, y, x) && src.at<uchar>(y, x) == OBIECT) {
				
				dir = d; 
				char dif = (dir - prev_dir)%8;
				contur.push_back(my_point{ x,y, dir, dif });
				j = x; i = y;
				n++;
				break;
			}
		}
		if (n > 1 && contur[0].x==contur[n - 1].x && contur[0].y == contur[n - 1].y 
							&& contur[1].x == contur[n].x && contur[1].y == contur[n].y) {
			finished = true;
		}

	}

	
	//scriere vector in fisier
	FILE* fp = fopen("D:\\Facultate\\PI\\Laborator\\OpenCVApplication-VS2019_OCV3411_basic\\OpenCVApplication-VS2019_OCV3411_basic\\debug.txt", "w");
	if (fp == NULL)
		printf("Error opening the text file !\n");
	fprintf(fp, "n (x, y) c, cd\n");
	for (int i = 0; i <=n; i++) {
		
		fprintf(fp, " %d (%d %d) %d %d\n",i, contur[i].x, contur[i].y, contur[i].c, contur[i].cd);
	}
	fclose(fp);
}



void lab6() {
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		//Create a window
		//show the image
		//in caz de binarizare
		/*int height = src.rows;
		int width = src.cols;

		int threshold=128;

		Mat dst = Mat(height, width, CV_8UC1);
		Vec3b aux;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				dst.at<uchar>(i, j) = src.at<uchar>(i, j) < threshold ? 0 : 255;

			}
		}   
		*/
		conturTracing(src);
		// Wait until user press some key
		waitKey(0);
	}
}

void reconstructFunction(Mat src) {
	FILE* fp = fopen("reconstruct.txt", "rt");
	if (fp == NULL)
		printf("Error opening the text file !\n");
	int x_start;
	int y_start;
	int n; 
	fscanf(fp, "%d", &x_start);
	fscanf(fp, "%d", &y_start);
	fscanf(fp, "%d", &n);
	src.at<uchar>(y_start, x_start) = 0;
	int x_new; int y_new;
	int x_prev = x_start; 
	int y_prev = y_start;

	boolean finished = false;
	int dir;
	int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 };
	int di[8] = { 0, -1, -1, -1, 0,  1, 1, 1 };
	printf("%d %d %d\n", x_start, y_start, n);
	for(int i=0; i<n; i++){
		if (fscanf(fp, "%d", &dir) == NULL) {
			// error 
			if (feof(fp)) //EOF reached 
				cout << "End of log file reached" << endl;
			if (ferror(fp))
				cout << "Log file read error: data from the current line is skiped" << endl;
			break;
		}
		else {
			x_new = x_prev + dj[dir];
			y_new = y_prev + di[dir];
			src.at<uchar>(y_new, x_new) = 0;
			x_prev = x_new;
			y_prev = y_new;
		}
	}
	imshow("Image", src);
	fclose(fp);


}

void lab6_reconstruct() {
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		reconstructFunction(src);
		// Wait until user press some key
		waitKey(0);
	}
}

//////////////////////////////////////// LAB 7
#define OBJECT 0
#define FUNDAL 255

void dilatare(Mat src, Mat *result) {
	*result = src.clone();
	int height = src.rows;
	int width = src.cols;
	int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 };
	int di[8] = { 0, -1, -1, -1,  0,  1, 1, 1 };

	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			if (src.at<uchar>(i, j) == OBJECT) {
				for (int v = 0; v < 8; v++) {
					int x = j + dj[v];
					int y = i + di[v];
					(*result).at<uchar>(y, x) = OBJECT;
				}
			}
		}
	}
}
void eroziune(Mat src, Mat* result) {
	*result = src.clone();
	int height = src.rows;
	int width = src.cols;
	int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 };
	int di[8] = { 0, -1, -1, -1, 0,  1, 1, 1 };

	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			if (src.at<uchar>(i, j) == OBJECT) {
				for (int v = 0; v < 8; v++) {
					int x = j + dj[v];
					int y = i + di[v];
					if (src.at<uchar>(y, x) == FUNDAL) {
						(*result).at<uchar>(i, j) = FUNDAL;
						break;
					}
				}
			}
		}
	}
}
void inchidere(Mat src, Mat *result) {
	Mat temp;
	dilatare(src, &temp);
	eroziune(temp, result);
}
void deschidere(Mat src, Mat* result) {
	Mat temp;
	eroziune(src, &temp);
	dilatare(temp, result);
}
void nDilate(Mat src,int n, Mat* result) {
	Mat temp=src.clone();
	*result = temp.clone();
	for (int i = 0; i < n; i++) {
		dilatare(temp, result);
		temp = (*result).clone();
	}
}
void nEroziune(Mat src, int n, Mat* result) {
	Mat temp = src.clone();
	*result = temp.clone();
	for (int i = 0; i < n; i++) {
		eroziune(temp, result);
		temp = (*result).clone();
	}
}
void lab7() {
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		//pentru dilatare
		Mat dstDilatare; 
		dilatare(src, &dstDilatare);
		imshow("Imagine sursa", src);
		imshow("Imagine ditatata", dstDilatare);
		//pentru eroziune
		Mat dstEroziune;
		eroziune(src, &dstEroziune);
		imshow("Imagine eroziune", dstEroziune);
		//inchidere
		Mat dstInchidere;
		inchidere(src, &dstInchidere);
		imshow("Inchidere", dstInchidere);
		//deschidere
		Mat dstDeschidere;
		deschidere(src, &dstDeschidere);
		imshow("Deschidere", dstDeschidere);
		//dilatare de n ori
		int n = 0;
		printf("Write the nb of dilation times: ");
		scanf("%d", &n);
		Mat nDilation;
		nDilate(src, n, &nDilation);
		imshow("N dilation: ", nDilation);
		//eroziune de n ori
		printf("Write the nb of erodate times: ");
		scanf("%d", &n);
		Mat nErodate;
		nEroziune(src, n, &nErodate);
		imshow("N erodation: ", nErodate);

		waitKey(0);
	}

}

//***********************TEMA LAB 7
//implementare operatie scadere
void difference(Mat src,  Mat* result) {
	Mat temp;
	*result = Mat(src.size(), CV_8UC1);
	eroziune(src, &temp);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) != temp.at<uchar>(i, j)) {
				(*result).at<uchar>(i, j) = 0;
			}
			else {
				(*result).at<uchar>(i, j) = 255;
			}
		}
	}
}

void fill_regions(Mat src, Mat* result) {
	
	Mat comp = src.clone();
	comp = 255 - src;
	Mat x1, x2, temp;
	x1 = Mat(src.rows, src.cols, CV_8UC1, FUNDAL);
	x2 = Mat(src.rows, src.cols, CV_8UC1, FUNDAL);
	temp = Mat(src.rows, src.cols, CV_8UC1, FUNDAL);
	int i_start = 80;
	int j_start = 80;
	x1.at<uchar>(i_start, j_start) = OBJECT;
	bool finished = false;
	int k = 1;
	
	do {
		finished = true;
		dilatare(x1, &temp);
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (temp.at<uchar>(i, j) == comp.at<uchar>(i, j)) {
					x2.at<uchar>(i, j) = temp.at<uchar>(i, j);
				}
				else {
					x2.at<uchar>(i, j) = FUNDAL;
				}
			}
		}
		for (int i = 0; i < x2.rows; i++) {
			for (int j = 0; j < x2.cols; j++) {
				if (x2.at<uchar>(i, j) != x1.at<uchar>(i, j)) {
					finished = false;
					break;
				}
			}
		}
		x1 = x2.clone();
		
	} while (!finished);

	(*result) = x2.clone();
	for (int i = 0; i < x2.rows; i++) {
		for (int j = 0; j < x2.cols; j++) {
			if (x2.at<uchar>(i, j) == OBJECT || src.at<uchar>(i, j) == OBJECT) {
				(*result).at<uchar>(i, j) = OBJECT;
			}
			else {
				(*result).at<uchar>(i, j) = FUNDAL;
			}
		}
	}
}

void testDifference() {
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname)){
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat result;
		difference(src, &result);
		imshow("Imagine sursa", src);
		imshow("Imagine destinatie", result);
		waitKey(0);
	}
}

void testFillRegions() {
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat result;
		fill_regions(src, &result);
		imshow("Imagine sursa", src);
		imshow("Imagine destinatie", result);
		waitKey(0);
	}
}

// **************************************** Lab 8 ************************************************
void compute_histograme(int*h, int *hc, float *p, int n, Mat src) {
	
	compute_histogram(h, n, src);
	compute_FDP(p, n, src);
	//compute comultative histogram
	hc[0] = h[0];
	for (int g = 1; g < 256; g++) {
		hc[g] = hc[g - 1] + h[g];
	}

}
 
void test_lab8() {
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat result;
		int n = 256;
		int h[256];
		int hc[256];
		float p[256];   //fdp
		compute_histograme(h,hc, p, n, src);
		showHistogram("Histograma", h, 256, 200);
		showHistogram("Histograma cumulativa", hc, 256, 200);

		waitKey(0);
	}
}
float val_medie_a_nivelurilor(float *p, int n, Mat src) {
	float m = 0.0f;;
	for (int g = 0; g < n; g++) {
		m += g*p[g];
	}
	return m;
}

float val_deviatiei_standard(float* p, int n, Mat src) {
	float v = 0.0f;
	float m = val_medie_a_nivelurilor(p, n, src);
	for (int g = 0; g < n; g++) {
		v += (g - m)*(g - m) * p[g];
	}
	
	return sqrt(v);   //returnare deviatie standard
}
void test_calcul_val_medie() {
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat result;
		int n = 256;
		int h[256];
		int hc[256];
		float p[256];   //fdp
		compute_histograme(h, hc, p, n, src);
		printf("Val medie: %.2f", val_medie_a_nivelurilor(p, n, src));

		waitKey(0);
	}
}

void test_deviatia_standard() {
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat result;
		int n = 256;
		int h[256];
		int hc[256];
		float p[256];   //fdp
		compute_histograme(h, hc, p, n, src);
		printf("Deviatie standard: %.2f", val_deviatiei_standard(p, n, src));

		waitKey(0);
	}
}
//determinarea pragului de binarizare globala
void binarizeImage(int threshold, Mat src, Mat *result) {
	
	int height = src.rows;
	int width = src.cols;
    *result = Mat(height, width, CV_8UC1);
		
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++)
		{
			(*result).at<uchar>(i, j) = src.at<uchar>(i, j) < threshold ? 0 : 255;

		}
	}
}
float  determinare_prag_binarizare_gobala(Mat src) {
	int n = 256;
	int h[256];
	int hc[256];
	float p[256];
	compute_histograme(h, hc, p, n, src);
	int Imin, Imax, gmin, gmax;
	//se parcurge h
	for (int g = 0; g < n; g++) {
		if (h[g] > 0) {
			Imin = g;
			break;
		}
	}
	//determinare Imax
	for (int g = n - 1; g >= 0; g--) {
		if (h[g] > 0) {
			Imax = g;
			break;
		}
	}

	float e = 0.5f;
	float Told = 0.0f;
	float T = (Imin + Imax) / 2.0f;
	do {
		Told = T;
		float m1=0.0f, m2=0.0f;
		for (int g = 0; g < Told; g++) {
			m1 += g * p[g];
		}
		for (int g = Told; g < n; g++) {
			m2 += g * p[g];
		}
		T = (m1 + m2) / 2.0f;
	} while (abs(T-Told)>e);
	return T;
}

void test_binarizare_imagine_threshold_global() {
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat result;
		float T = determinare_prag_binarizare_gobala(src);
		binarizeImage(T, src, &result);
		imshow("Imagine sursa", src);
		imshow("Imagine binarizata", result);
		waitKey(0);
	}

}


//exercitiul 3
//tranform =0 : identitate
//tranform =1 : negativul
//tranform =2 : latirea/ingucstare
// transform=3: constrast
void exercitiul3(Mat src, int transform) {

	Mat_<uchar> dst = Mat::zeros(src.size(), CV_8UC1);
	int gin_min=0.0f; 
	int gin_max=0.0f;
	int n = 256;
	int h[256];
	int hc[256];
	float p[256];
	float gamma=1.0f;
	compute_histograme(h, hc, p, n, src);

	for (int g = 0; g < n; g++) {
		if (h[g] > 0) {
			gin_min = g;
			break;
		}
	}
	//determinare Imax
	for (int g = n - 1; g >= 0; g--) {
		if (h[g] > 0) {
			gin_max = g;
			break;
		}
	}
	int gout_max=0.0f;
	int gout_min = 0.0f;
	if (transform == 2) {
		printf("Write gout_max: ");
		scanf("%d", &gout_max);
		printf("Write gout_min: ");
		scanf("%d", &gout_min);
	}
	if (transform == 3) {
		printf("Write gamma: ");
		scanf("%f", &gamma);
	}
	int L = 255;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			switch (transform) {
			case 0:
				dst(i, j) = src.at<uchar>(i, j);
				break;
			case 1:
				dst(i, j) = 255 - src.at<uchar>(i, j);
				break;
			case 2:
				dst(i, j) = gout_min + (src.at<uchar>(i, j) - gin_min) * (gout_max - gout_min) / (gin_max - gin_min);
				break;
			case 3:
				dst(i, j) = L * pow(((float)src.at<uchar>(i, j)) / L, gamma);
				break;
			default:
				dst(i, j) = src.at<uchar>(i, j);
				break;
			}
		}
	}
	int hnew[256];
	int hcnew[256];
	float pnew[256];
	compute_histograme(hnew, hcnew, pnew, n, dst);
	switch (transform) {
	case 0:
		imshow("Imagine sursa: ", src);
		imshow("Imagine identitate: ", dst);
		break;
	case 1:
		imshow("Imagine sursa: ", src);
		imshow("Imagine negativa: ", dst);
		break;
	case 2:
		imshow("Imagine sursa: ", src);
		imshow("Imagine contrast modificat: ", dst);
		showHistogram("Histograma sursei", h, 256, 200);
		showHistogram("Histograma destinatiei", hnew, 256, 200);
		break;
	case 3:
		imshow("Imagine sursa: ", src);
		imshow("Imagine corectie gamma: ", dst);
		showHistogram("Histograma sursei", h, 256, 200);
		showHistogram("Histograma corectiei gamma", hnew, 256, 200);
		break;
	default:
		imshow("Imagine sursa: ", src);
		imshow("Imagine identitate: ", dst);
		break;

	}
	waitKey(0);
}

void test_transformari_imagine() {
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int transform = 0;
		printf("Write tranform\n  0 : identitate\n  1 : negativul\n  2 : latirea/ingucstare\n  3: corectie gama\n transmform= ");
		scanf("%d", &transform);
		exercitiul3(src, transform);
	}
}

void egalizarea_histogramei(Mat src) {
	int n = 256;
	int h[256];
	int hc[256];
	float p[256];
	compute_histograme(h, hc, p, n, src);
	int M = src.rows * src.cols;
	byte tab[256];
	for (int g = 0; g < n; g++) {
		tab[g] = 255 * hc[g] / M;
	}

	Mat_<uchar> dst = Mat::zeros(src.size(), CV_8UC1);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dst(i, j) = tab[src.at<uchar>(i, j)];
		}
	}
	int hdst[256];
	int hcdst[256];
	float pdst[256];
	imshow("Imagine susrsa", src);
	imshow("imagine destinatia", dst);
	compute_histograme(hdst, hcdst,pdst, n, dst);
	showHistogram("Histograma sursei", h, 256, 200);
	showHistogram("Histograma destinatiei", hdst, 256, 200);
}

void test_egalizarea_histogramei() {
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		egalizarea_histogramei(src);
	}
}

// *************************************** LABORATOR 9  *****************************************
///////////////////////////////////////////////////////////////////////////////////////////////////

//definire filtru : filtru medie arimetica
int h[7][7] = {
	{ 1, 1, 1, 1, 1, 1, 1 },
	{ 1, 1, 1, 1, 1, 1, 1 },
	{ 1, 1, 1, 1, 1, 1, 1 },
	{ 1, 1, 1, 1, 1, 1, 1 },
	{ 1, 1, 1, 1, 1, 1, 1 },
	{ 1, 1, 1, 1, 1, 1, 1 },
	{ 1, 1, 1, 1, 1, 1, 1 } };

int HS[3][3] = {
	{0,-1,0},
	{-1, 4, -1},
	{0, -1, 0 }
};

enum {
	FILTRU_JOS_INT, 
	FILTRU_JOS_FLOAT, 
	FILTRU_SUS,
};

int compute_fs(int d) {
	int fs = 0;
	for (int y = 0; y <= 2 * d; y++) {
		for (int x = 0; x <= 2 * d; x++) {
			fs += h[y][x];
		}
	}
	return fs;
}

void compute_sp_sn(int d, int* sp, int* sn) {
	*sp = 0;
	*sn = 0;
	for (int y = 0; y < 2 * d; y++) {
		for (int x = 0; x < 2 * d; x++) {
			if (HS[y][x] < 0)
				(*sn) += -HS[y][x];
			else
				(*sp) += HS[y][x];
		}
	}
}


void convolutie_filtrare_generala(Mat_<uchar> src, Mat* dst, int w, int option) {
	int d = w / 2;
	int height = src.rows;
	int width = src.cols;
	int fs = 1;
	int sp, sn;
	switch (option) {
		case FILTRU_JOS_INT:
			fs = compute_fs(d);
			break;
		case FILTRU_JOS_FLOAT:
			fs = compute_fs(d);
			break;
		case FILTRU_SUS:
			compute_sp_sn(d, &sp, &sn);
			fs = 2*max_(sp, sn);
			break;
		default: 
			break;
	}
	printf("%d ", fs);
	for (int i = d; i < height-d; i++) {
		for (int j = d; j < width-d; j++) {
			float sum = 0.0;
			for (int y = -d; y <= d; y++) {
				for (int x = -d; x <= d; x++) {
					if(option == FILTRU_SUS)
						sum += HS[y + d][x + d] * src[i+y][j+x];
					else
						sum += h[y + d][x + d] * src[i + y][j + x];
				}
			}
			switch (option) {
				case FILTRU_JOS_INT: 
					(*dst).at<uchar>(i, j) =(uchar) (sum / fs);
					break;
				case FILTRU_JOS_FLOAT:
					(*dst).at<uchar>(i, j) = (uchar)(sum / fs);
					break;
				case FILTRU_SUS: 
					(*dst).at<uchar>(i, j) = (uchar)( sum / fs)+ 127;
					break;
			}
		}
	}
}


void test_filtru_trece_jos() {
	Mat_<uchar> src;
	char fname[MAX_PATH];
	openFileDlg(fname);
	src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	Mat dst = src.clone();
	int w;
	printf("Write w = ");
	scanf("%d", &w);
	convolutie_filtrare_generala(src, &dst, w, FILTRU_JOS_INT);
	imshow("Input image", src);
	imshow("Filtru jos image", dst);
	waitKey(0);
}

void test_filtru_trece_sus() {
	Mat_<uchar> src;
	char fname[MAX_PATH];
	openFileDlg(fname);
	src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	Mat dst = src.clone();
	int w;
	printf("Write w = ");
	scanf("%d", &w);
	convolutie_filtrare_generala(src, &dst, w, FILTRU_SUS);
	imshow("Input image", src);
	imshow("Filtru sus image", dst);
	waitKey(0);
}

void centering_transform(Mat img) {
	//expects floating point image
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
		}
	}
}

Mat generic_frequency_domain_filter(Mat src, int option)  //input de la tastatura pentru option
{
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);

	// Centering transformation 
	centering_transform(srcf);

	//perform forward transform with complex image output
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	//split into real and imaginary channels
	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels);  // chanels[0] = Re(DFT(I), chanels[1] = Im(DFT(I))

	//calculate magnitude and phase in floating point images mag and phi
	Mat mag, phi;
	magnitude(channels[0], channels[1], mag);
	phase(channels[0], channels[1], phi);

	// Dislplay here the phase and magnitude
	// ......
	mag += Scalar::all(1); 
	log(mag, mag);
	Mat spectrum, phase;
	normalize(mag, spectrum ,0 ,255, NORM_MINMAX, CV_8UC1);

	imshow("SPECTRUM ",spectrum);
	normalize(phi, phase, 0, 255, NORM_MINMAX, CV_8UC1);
	imshow("PHASE ", phase);
	waitKey(0);
	// Insert filtering operations here ( chanles[0] = Re(DFT(I), chanels[1] = Im(DFT(I) )
	int poz;
	float coef;
	float R = 10; // filter "radius"
	printf("Write radius R=");
	//scanf("%d", &R);
	int height = src.rows;
	int width = src.cols;
	
	switch (option) {
	case 1: break; // NO filter
	case 2:
		// FTJ ideal
		// inserati codul de filtrare aici ...
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				poz = (height/2 - i)  * (height/2 - i)  + (width/2 - j) * (width/2 - j);
				if (poz > R * R) {
					channels[0].at<float>(i, j) = 0.0f;
					channels[1].at<float>(i, j) = 0.0f;
				}
			}
		}
		break;
	case 3:
		// FTS ideal
		// inserati codul de filtrare aici ...
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				poz = (height / 2 - i) * (height / 2 - i) +
					(width / 2 - j) * (width / 2 - j);
				if (poz < R * R) {
					channels[0].at<float>(i, j) = 0.0f;
					channels[1].at<float>(i, j) = 0.0f;
				}
			}
		}
		break;
	case 4:
		// FTJ gauss
		// inserati codul de filtrare aici ...
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				poz = (height / 2 - i) * (height / 2 - i) +
					(width / 2 - j) * (width / 2 - j);
				float coef = exp(-poz / (R * R));
				channels[0].at<float>(i, j) *= coef;
				channels[1].at<float>(i, j) *= coef;
			}
		}
		break;
	case 5:
		// FTJ gauss
		// inserati codul de filtrare aici ...
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				poz = (height / 2 - i) * (height / 2 - i) +
					(width / 2 - j) * (width / 2 - j);
				float coef = 1-exp(-poz / (R * R));
				channels[0].at<float>(i, j) *= coef;
				channels[1].at<float>(i, j) *= coef;
			}
		}
		break;
	}

	//perform inverse transform and put results in dstf
	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
	centering_transform(dstf);
	//normalize the result and put in the destination image
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	//dstf.convertTo(dst, CV_8UC1);

	return dst;
}

void test_generic() {
	Mat src;
	char fname[MAX_PATH];
	openFileDlg(fname);
	src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	int option;
	printf("2 - FTJ\n");
	printf("3 - FTS\n");
	printf("4 - FGJ\n");
	printf("5 - FGS\n");
	printf("Option: ");
	scanf("%d", &option);
	Mat dst =generic_frequency_domain_filter(src, option);
	imshow("Input image", src);
	imshow("Output", dst);
	waitKey(0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                            LAB 10
// filtru median 
void swap(uchar* a, uchar* b)
{
	byte t = *a;
	*a = *b;
	*b = t;
}

int partition(uchar arr[], int low, int high)
{
	uchar pivot = arr[high]; // pivot
	int i = (low - 1); 

	for (int j = low; j <= high - 1; j++)
	{
		if (arr[j] < pivot)
		{
			i++; 
			swap(&arr[i], &arr[j]);
		}
	}
	swap(&arr[i + 1], &arr[high]);
	return (i + 1);
}
void quickSort(uchar arr[], int low, int high)
{
	if (low < high)
	{
		int pi = partition(arr, low, high);
		quickSort(arr, low, pi - 1);
		quickSort(arr, pi + 1, high);
	}
}


Mat medianFilter(Mat src) {
	int height = src.rows; 
	int width = src.cols;
	int w = 5; 
	printf("Write w=(3,5,7,9):");
	scanf("%d", &w);
	int d = 1;
    d = w / 2;
	Mat dst = src.clone();
	double t = (double)getTickCount();
	for (int i = d; i < height - d; i++) {
		for (int j = d; j < width - d; j++) {
			uchar* L = (uchar*)malloc(w*w);
			if (L == NULL) {
				printf("Nu s-a putut aloca memorie\n");
				exit(1);
			}
			int contor = 0;
			for (int m = -d; m <= d; m++) {
				for (int n = -d; n <= d; n++) {
					L[contor] = src.at<uchar>(i + m, j + n);
					contor++;
				}
			}
			quickSort(L, 0, w * w - 1);
			dst.at<uchar>(i, j) = L[w*w/2];
			free(L);
		}
		
	}
	t = ((double)getTickCount() - t) / getTickFrequency();
	//Print the proccessing time
	printf("Time = %.3f [ms]\n", t * 1000);
	return dst;
}

void testMedianFiltre() {
	Mat src;
	char fname[MAX_PATH];
	openFileDlg(fname);
	src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	double t = (double)getTickCount();
	Mat dst = medianFilter(src);
	imshow("Imagine sursa:", src);
	imshow("Imagine destinatie", dst);
	waitKey(0);
}

//Zgomit Gaussian
Mat getConvolutie(Mat src, int d, float fs, float S[9][9]) {
	int height = src.rows;
	int width = src.cols;
	Mat dst = src.clone();
	for (int i = d; i < height - d; i++) {
		for (int j = d; j < width - d; j++) {
			float sum = 0.0;
			for (int y = -d; y <= d; y++) {
				for (int x = -d; x <= d; x++) {
						sum += (S[y + d][x + d] * src.at<uchar>(i + y,j + x));
				}
			}
			dst.at<uchar>(i, j) = (uchar)(sum / fs);
			}
	}
	return dst;
}
Mat gaussianFiltre(Mat src) {
	int height = src.rows;
	int width = src.cols;
	int w = 3; 
	printf("Write w=(3,5,7):");
	scanf("%d", &w);
	int d = 1;
	d = w / 2;
	float S[9][9]; 
	float sum = 0.0f;
	float sigma = ((float)w) / 6.0f;
	Mat dst;
	for (int y = 0; y < w; y++) {
		for (int x = 0; x < w; x++) {
			float E = exp(-(x - d) * (x - d) + (y - d) * (y - d) / (2 * sigma * sigma));
			float N = 2 * PI * sigma * sigma;
			S[y][x] = E / N;
			sum += S[y][x];
		}
	}
	printf("Suma %f\n", sum);
	//convlolutia
	double t = (double)getTickCount();
	dst = getConvolutie(src, d, sum, S);
	t = ((double)getTickCount() - t) / getTickFrequency();
	//Print the proccessing time
	printf("Time = %.3f [ms]\n", t * 1000);
	return dst;
}

void testGaussianFiltre() {
	Mat src;
	char fname[MAX_PATH];
	openFileDlg(fname);
	src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	double t = (double)getTickCount();
	Mat dst = gaussianFiltre(src);
	imshow("Imagine sursa:", src);
	imshow("Imagine destinatie", dst);
	waitKey(0);
}

Mat optimizedGaussian(Mat src) {
	int height = src.rows;
	int width = src.cols;
	float sum = 0.0f;
	int w = 3;
	printf("Write w=(3,5,7):");
	scanf("%d", &w);
	int d = 1;
	d = w / 2;
	float S[9];
	float sigma = ((float)w) / 6.0f;
	for (int x = 0; x < w; x++) {
		float E = exp(-(x - d) * (x - d) / (2 * sigma * sigma));
		float N = sigma * sqrt(2 * PI);
		S[x] = E / N;
		sum += S[x];
	}
	printf("Sum : %f\n", sum);
	Mat temp = src.clone();
	double t = (double)getTickCount();
	for (int i = d; i < height - d; i++) {
		for (int j = d; j < width - d; j++) {
			float ps = 0.0f;
			for (int m = -d; m <= d; m++) {
				ps += src.at<uchar>(i + m, j) * S[m + d];
			}
			temp.at<uchar>(i, j) = ps / sum;
		}
	}
	//pentru destinatie
	Mat dst = src.clone();
	for (int i = d; i < height - d; i++) {
		for (int j = d; j < width - d; j++) {
			float ps = 0.0f;
			for (int m = -d; m <= d; m++) {
				ps += temp.at<uchar>(i , j+m) * S[m + d];
			}
			dst.at<uchar>(i, j) = ps / sum;
		}
	}
	t = ((double)getTickCount() - t) / getTickFrequency();
	//Print the proccessing time
	printf("Time = %.3f [ms]\n", t * 1000);
	return dst;
}

void testOptimizedGaussianFiltre() {
	Mat src;
	char fname[MAX_PATH];
	openFileDlg(fname);
	src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	double t = (double)getTickCount();
	Mat dst = optimizedGaussian(src);
	imshow("Imagine sursa:", src);
	imshow("Imagine destinatie", dst);
	waitKey(0);
}


//---------------------------------------------------------LAB11---------------------------------------------
Mat gaussianFiltre(Mat src, int w) {
	int height = src.rows;
	int width = src.cols;
	int d = 1;
	d = w / 2;
	float S[9][9];
	float sum = 0.0f;
	float sigma = ((float)w) / 6.0f;
	Mat dst;
	for (int y = 0; y < w; y++) {
		for (int x = 0; x < w; x++) {
			float E = exp(-(x - d) * (x - d) + (y - d) * (y - d) / (2 * sigma * sigma));
			float N = 2 * PI * sigma * sigma;
			S[y][x] = E / N;
			sum += S[y][x];
		}
	}
	printf("Suma %f\n", sum);
	//convlolutia
	double t = (double)getTickCount();
	dst = getConvolutie(src, d, sum, S);
	t = ((double)getTickCount() - t) / getTickFrequency();
	//Print the proccessing time
	printf("Time = %.3f [ms]\n", t * 1000);
	return dst;
}

void computeModulAndDirection(Mat temp, Mat* modul, Mat* directie, int d) {
	int Sx[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
	int Sy[3][3] = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };
	int height = temp.rows;
	int width = temp.cols;
	
	
	for (int i = d; i < height - d; i++) {
		for (int j = d; j < width - d; j++) {
			int gradX = 0;
			int gradY = 0;
			for (int y = -d; y <= d; y++) {
				for (int x = -d; x <= d; x++) {
					gradX += (Sx[y + d][x + d] * temp.at<uchar>(i + y, j + x));
					gradY += (Sy[y + d][x + d] * temp.at<uchar>(i + y, j + x));
				}
			}
			(*modul).at<uchar>(i,j) = sqrt(gradX * gradX + gradY * gradY) / 5.65;
			int dir=0;
			float teta = atan2((float)gradY, (float)gradX);
			if ((teta > 3 * PI / 8 && teta < 5 * PI / 8) || (teta > -5 * PI / 8 && teta < -3 * PI / 8))
				dir = 0;

			if ((teta > PI / 8 && teta < 3 * PI / 8) || (teta > -7 * PI / 8 && teta < -5 * PI / 8)) 
				dir = 1;

			if ((teta > -PI / 8 && teta < PI / 8) || teta > 7 * PI / 8 && teta < -7 * PI / 8)
				dir = 2;

			if ((teta > 5 * PI / 8 && teta < 7 * PI / 8) || (teta > -3 * PI / 8 && teta < -PI / 8))
				dir = 3;

			(*directie).at<uchar>(i, j) = dir;
		}
	}


}

void nonMaximumSuppresion(Mat* modul, Mat direction) {
	int height = modul->rows;
	int width = modul->cols;
	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			switch (direction.at<uchar>(i, j)) {
			case 0: 
				if (modul->at<uchar>(i, j) < modul->at<uchar>(i - 1, j) ||
					modul->at<uchar>(i, j) < modul->at<uchar>(i + 1, j))
					modul->at<uchar>(i, j) = 0;
				break;
			case 1:
				if (modul->at<uchar>(i, j) < modul->at<uchar>(i - 1, j-1) ||
					modul->at<uchar>(i, j) < modul->at<uchar>(i + 1, j+1))
					modul->at<uchar>(i, j) = 0;
				break;
			case 2:
				if (modul->at<uchar>(i, j) < modul->at<uchar>(i , j - 1) ||
					modul->at<uchar>(i, j) < modul->at<uchar>(i, j + 1))
					modul->at<uchar>(i, j) = 0;
				break;
			case 3:
				if (modul->at<uchar>(i, j) < modul->at<uchar>(i-1, j + 1) ||
					modul->at<uchar>(i, j) < modul->at<uchar>(i+1, j - 1))
					modul->at<uchar>(i, j) = 0;
				break;
			}
		}
	}
}
#define WEAK 128 
#define STRONG 255 
void compute_histogram2(int* histogram, int n, Mat modul) {
	int height = modul.rows;
	int width = modul.cols;
	for (int i = 0; i < n; i++) {
		histogram[i] = 0;
	}
	for (int i = 1; i < height-1; i++)
	{
		for (int j = 1; j < width-1; j++)
		{
			histogram[modul.at<uchar>(i, j)] += 1;
		}
	}
}
void computeAdaptiveThreshold(Mat modul, int *histogram, int *pH, int *pL) {
	float p = 0.1f;
	float k = 0.4f;
	int height = modul.rows;
	int width = modul.cols;
	compute_histogram2(histogram, 256, modul);
	//calculare nr pixeli cu modul diferit de zero care nu vor fi puncte de muchie
	float nrNonMuchie = (1 - p) * ((height - 2) * (width - 2) - histogram[0]);
	//Calculati Pragul adaptiv insumand elementele din Hist(incepand de la Hist[1]),
	//oprindu - va in momentul in care suma > NrNonMuchie.
	float sum = 0.0f;
	int adaptiveThreshold=0;
	for (int i = 1; i < 256; i++) {
		sum += histogram[i];
		if (sum > nrNonMuchie) {
			adaptiveThreshold = i;
			break;
		}
	}
	*pH=adaptiveThreshold;
	*pL = k * adaptiveThreshold;
}

void histerezaBinarize(Mat* modul, int pl, int ph) {
	int height = modul->rows;
	int width = modul->cols;
	for (int i = 0; i < height ; i++) {
		for (int j =0; j < width; j++) {
			if (modul->at<uchar>(i, j) < pl) 
				modul->at<uchar>(i, j) = 0;
			else if (modul->at<uchar>(i, j) > ph) 
				modul->at<uchar>(i, j) = STRONG;
			else 
				modul->at<uchar>(i, j) = WEAK;
		}
	}
}

void edgesExtension(Mat* modul) {
	Mat_<uchar> visited = Mat::zeros(modul->size(), CV_8UC1);
	queue<Point> que;
	int height = modul->rows;
	int width = modul->cols;
	int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 }; // row 
	int di[8] = { 0, -1, -1, -1,  0,  1, 1, 1 }; // col 

	for (int i = 2; i < height - 3; i++) {
		for (int j = 2; j < width - 3; j++) {
			if (modul->at<uchar>(i, j) == STRONG && visited(i, j) == 0) {
				que.push(Point(j, i));
				visited(i, j) = 1;
			}
			while (!que.empty()) {
				Point oldest = que.front();
				int jj = oldest.x;
				int ii = oldest.y;
				que.pop();
				int mag = 0;
				for (int d = 0; d < 8; d++) {
					if (modul->at<uchar>(ii + di[d], jj + dj[d]) == WEAK) {
						modul->at<uchar>(ii + di[d], jj + dj[d]) = STRONG;
						que.push(Point(jj + dj[d], ii + di[d]));
						visited(ii + di[d], jj + dj[d]) = 1;
					}
				}
			}
		}
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (modul->at<uchar>(i, j) == WEAK) {
				modul->at<uchar>(i, j) = 0;
			}
		}
	}

}
void showDirections(Mat modul, Mat directie) {
	Scalar colorLUT[4] = { 0 };
	colorLUT[0] = Scalar(0, 0, 255); //red 
	colorLUT[1] = Scalar(0, 255, 255); // yellow 
	colorLUT[2] = Scalar(255, 0, 0); // blue 
	colorLUT[3] = Scalar(0, 255, 0); // green 
	Mat_<Vec3b> ImgDir = Mat::zeros(modul.size(), CV_8UC3);
	int d = 1;
	int height = modul.rows;
	int width = modul.cols;
	for (int i = d; i < height - d; i++) // d=1 
		for (int j = d; j < width - d; j++)
			if (modul.at<uchar>(i, j)){
				Scalar color = colorLUT[directie.at<uchar>(i, j)];
				ImgDir(i, j)[0] = color[0];
				ImgDir(i, j)[1] = color[1];
				ImgDir(i, j)[2] = color[2];
			}
	imshow("Imagine directii", ImgDir);
}
void  cannyFunction(Mat src) {
	Mat temp = src.clone();
	Mat modul = Mat::zeros(src.size(), CV_8UC1);
	Mat directie = Mat::zeros(src.size(), CV_8UC1);
	int w = 3; 
	int d = w/2;
	temp = gaussianFiltre(src, w);
	
	computeModulAndDirection(temp, &modul, &directie,d);
	imshow("Modul raw", modul);
	nonMaximumSuppresion(&modul, directie);
	imshow("Modu lafter NMS", modul);
	//compute adaptive threshold 
	int histogram[256];
	//stabilire praguri de binarizare
	int pH, pL;
    computeAdaptiveThreshold(modul, histogram, &pH, &pL);
	//binarizare cu histereza
	printf("PH %d", pH);
	printf("P%d", pL);

	histerezaBinarize(&modul, pL, pH);
	imshow("Modul cu binarizare histereza", modul);
	//extinderea muchiilor
	edgesExtension(&modul);
	imshow("Modul dup edges extension", modul);

	//afisare cu cod de culoari
	showDirections(modul, directie);
	showHistogram("Histogram", histogram, 256, 300);
	waitKey(0);
}
void testCannyFunction() {
	Mat src;
	char fname[MAX_PATH];
	openFileDlg(fname);
	src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	cannyFunction(src);
}
int main()
{
	int a[256];
	float p[256];
	int op;
	int sum = 0;

	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" --------Laborator 1-----------\n");
		printf(" --------Laborator 1-----------\n");
		printf(" 10 - Additive brightness\n");
		printf(" 11 - Multiplicative brightness\n");
		printf(" 12 - Cadrane\n");
		printf(" 13 - Inversa\n");

		printf(" --------Laborator 2-----------\n");
		printf(" 14 - Copiere canale RGB(3 cadrane)\n");
		printf(" 15 - Color to grayscale\n");
		printf(" 16 - Graysvale to white-black\n");
		printf(" 17 - ExtractHSV\n");
		printf(" 18 - Test function isInside\n");
		printf(" --------Laborator 3-----------\n");
		printf(" 19 - Compute histogram\n");
		printf(" 20 - Compute FDP\n");
		printf(" 21 - Show histogram \n");
		printf(" 22 - Reduce number of acumulators for histogram\n");
		printf(" 23 - Multiple thresholds\n");
		printf(" 24 - Floyd-Steinberg\n");
		printf(" --------Laborator 4-----------\n");
		printf(" 25 - Lab4\n");
		printf(" --------Laborator 5-----------\n");
		printf(" 26 - Lab5 -labelingAlgorithm\n");
		printf(" --------Laborator 6-----------\n");
		printf(" 27 - Lab6 -Contur tracing\n");
		printf(" 28 - Lab6 -Contur reconstruct\n");
		printf(" --------Laborator 7-----------\n");
		printf(" 29 - Lab7\n");
		printf(" 30 - Difference\n");
		printf(" 31 - Fill regions\n");
		printf(" --------Laborator 8-----------\n");
		printf(" 32-Lab8-histograme\n");
		printf(" 33-Val medie a nivelurilor de intensitate\n");
		printf(" 34-Test deviatie standard\n");
		printf(" 35-Test threshold binarizare\n");
		printf(" 36-Test transformari imagine\n");
		printf(" 37-Test egalizarea histogramei\n");
		printf(" --------Laborator 9 -----------\n");
		printf("38 - Test filtru jos\n");
		printf("39 - Test filtru sus\n");
		printf("40 - Test generic\n");
		printf(" --------Laborator 10 -----------\n");
		printf("41 - Test median filtre\n");
		printf("42 - Gaussian  filtre\n");
		printf("43 - Optimized Gaussian  filtre\n");
		printf(" -------- Laborator 11 -----------\n");
		printf("44- Test canny\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op); 
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testParcurgereSimplaDiblookStyle(); //diblook style
			break;
		case 4:
			//testColor2Gray();
			testBGR2HSV();
			break;
		case 5:
			testResize();
			break;
		case 6:
			testCanny();
			break;
		case 7:
			testVideoSequence();
			break;
		case 8:
			testSnap();
			break;
		case 9:
			testMouseClick();
			break;
		case 10:
			printf("Write the additive: ");
			int additive;
			scanf("%d", &additive);
			brightnessAditive(additive);
			break;
		case 11:
			printf("Write the multiplicative: ");
			int multipliclative;
			scanf("%d", &multipliclative);
			brightnessMultiplicative(multipliclative);
			break;
		case 12:
			createMatrix_256();
			break;
		case 13:
			createMatrixFloat();
			break;
		case 14:
			extractRGB();
			break;
		case 15:
			colorToGrayscale();
			break;
		case 16:
			binarizeImage();
			break;
		case 17:
			extractHSV();
			break;
		case 18:
			testIsInside();
			break;
		case 19:
			compute_histogram(a, 256);
			break;
		case 20: 
			compute_FDP(p, 256);
			break;
		case 21:
			compute_histogram(a, 256);
			showHistogram("Histograma", a, 256, 300);
			break;
		case 22:
			lower_histogram_acc();
			break;
		case 23:
			multiple_thresholds();
			break;
		case 24:
			floydSteinberg_alg();
			break;
		case 25:
			lab4();
			break;
		case 26:
			lab5();
			break;
		case 27:
			lab6();
			break;
		case 28:
			lab6_reconstruct();
			break;
		case 29:
			lab7();
			break;
		case 30:
			testDifference();
			break;
		case 31:
			testFillRegions();
			break;
		case 32:
			test_lab8();
			break;
		case 33:
			test_calcul_val_medie();
			break;
		case 34:
			test_deviatia_standard();
			break;
		case 35:
			test_binarizare_imagine_threshold_global();
			break;
		case 36:
			test_transformari_imagine();
			break;
		case 37:
			test_egalizarea_histogramei();
			break;
		case 38:
			test_filtru_trece_jos();
			break;
		case 39:
			test_filtru_trece_sus();
			break;
		case 40:
			test_generic();
			break;
		case 41:
			testMedianFiltre();
			break;
		case 42:
			testGaussianFiltre();
			break;
		case 43:
			testOptimizedGaussianFiltre();
			break;
		case 44:
			testCannyFunction();
			break;
			
			
		default:
			break;
		}
		
	} while (op != 0);
	
	return 0;
}