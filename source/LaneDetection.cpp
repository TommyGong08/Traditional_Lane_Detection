#include<opencv2/opencv.hpp>
#include<iostream>
#include<cmath>
#include<fstream>
using namespace std;
using namespace cv;
#define PI 3.1416
#define min(a,b) (a<b?a:b)

//CheckMode: 0代表去除黑区域，1代表去除白区域; NeihborMode：0代表4邻域，1代表8邻域;  
void RemoveSmallRegion(Mat& Src, Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode)
{
	int RemoveCount = 0;       //记录除去的个数  
	//记录每个像素点检验状态的标签，0代表未检查，1代表正在检查,2代表检查不合格（需要反转颜色），3代表检查合格或不需检查  
	Mat Pointlabel = Mat::zeros(Src.size(), CV_8UC1);

	if (CheckMode == 1)
	{
		cout << "Mode: 去除小区域. ";
		for (int i = 0; i < Src.rows; ++i)
		{
			uchar* iData = Src.ptr<uchar>(i);
			uchar* iLabel = Pointlabel.ptr<uchar>(i);
			for (int j = 0; j < Src.cols; ++j)
			{
				if (iData[j] < 10)
				{
					iLabel[j] = 3;
				}
			}
		}
	}
	else
	{
		cout << "Mode: 去除孔洞. ";
		for (int i = 0; i < Src.rows; ++i)
		{
			uchar* iData = Src.ptr<uchar>(i);
			uchar* iLabel = Pointlabel.ptr<uchar>(i);
			for (int j = 0; j < Src.cols; ++j)
			{
				if (iData[j] > 10)
				{
					iLabel[j] = 3;
				}
			}
		}
	}

	vector<Point2i> NeihborPos;  //记录邻域点位置  
	NeihborPos.push_back(Point2i(-1, 0));
	NeihborPos.push_back(Point2i(1, 0));
	NeihborPos.push_back(Point2i(0, -1));
	NeihborPos.push_back(Point2i(0, 1));
	if (NeihborMode == 1)
	{
		cout << "Neighbor mode: 8邻域." << endl;
		NeihborPos.push_back(Point2i(-1, -1));
		NeihborPos.push_back(Point2i(-1, 1));
		NeihborPos.push_back(Point2i(1, -1));
		NeihborPos.push_back(Point2i(1, 1));
	}
	else cout << "Neighbor mode: 4邻域." << endl;
	int NeihborCount = 4 + 4 * NeihborMode;
	int CurrX = 0, CurrY = 0;
	//开始检测  
	for (int i = 0; i < Src.rows; ++i)
	{
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 0)
			{
				//********开始该点处的检查**********  
				vector<Point2i> GrowBuffer;                                      //堆栈，用于存储生长点  
				GrowBuffer.push_back(Point2i(j, i));
				Pointlabel.at<uchar>(i, j) = 1;
				int CheckResult = 0;                                               //用于判断结果（是否超出大小），0为未超出，1为超出  

				for (int z = 0; z < GrowBuffer.size(); z++)
				{

					for (int q = 0; q < NeihborCount; q++)                                      //检查四个邻域点  
					{
						CurrX = GrowBuffer.at(z).x + NeihborPos.at(q).x;
						CurrY = GrowBuffer.at(z).y + NeihborPos.at(q).y;
						if (CurrX >= 0 && CurrX < Src.cols && CurrY >= 0 && CurrY < Src.rows)  //防止越界  
						{
							if (Pointlabel.at<uchar>(CurrY, CurrX) == 0)
							{
								GrowBuffer.push_back(Point2i(CurrX, CurrY));  //邻域点加入buffer  
								Pointlabel.at<uchar>(CurrY, CurrX) = 1;           //更新邻域点的检查标签，避免重复检查  
							}
						}
					}

				}
				if (GrowBuffer.size() > AreaLimit) CheckResult = 2;                 //判断结果（是否超出限定的大小），1为未超出，2为超出  
				else { CheckResult = 1;   RemoveCount++; }
				for (int z = 0; z < GrowBuffer.size(); z++)                         //更新Label记录  
				{
					CurrX = GrowBuffer.at(z).x;
					CurrY = GrowBuffer.at(z).y;
					Pointlabel.at<uchar>(CurrY, CurrX) += CheckResult;
				}


			}
		}
	}

	CheckMode = 255 * (1 - CheckMode);
	//开始反转面积过小的区域  
	for (int i = 0; i < Src.rows; ++i)
	{
		uchar* iData = Src.ptr<uchar>(i);
		uchar* iDstData = Dst.ptr<uchar>(i);
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 2)
			{
				iDstData[j] = CheckMode;
			}
			else if (iLabel[j] == 3)
			{
				iDstData[j] = iData[j];
			}
		}
	}
	Pointlabel.release();
	cout << RemoveCount << " objects removed." << endl;
}

void myGaus2Binary(cv::Mat& src, cv::Mat& dst)
{
	if (!src.data) return;
	cv::Mat temp(src.size(), src.type());
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (i >= 0 && i <= 350)
			{
				temp.at<uchar>(i, j) = 0;
			}
			else
			{
				if(src.at<uchar>(i, j) > 150 && src.at<uchar>(i, j) < 255)
				{
					temp.at<uchar>(i, j) = 255;
				}
				else
					temp.at<uchar>(i, j) = 0;
			}
		}
	}
	temp.copyTo(dst);
	temp.release();
}

void myRGB2GRAY(cv::Mat& src, cv::Mat& gray, int width, int height)
{
	vector<cv::Mat> channels;
	cv::split(src, channels);
	cv::Mat red, green, blue;
	blue = channels.at(0);
	green = channels.at(1);
	red = channels.at(2);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			gray.at<uchar>(i, j) = blue.at<uchar>(i, j) * 0.1140 + green.at<uchar>(i, j) * 0.5870 + red.at<uchar>(i, j) * 0.2989;
		}
	}
	//cv::imshow("gray image", gray_img);
	red.release();
	green.release();
	blue.release();
}

void myHist(cv::Mat& src, cv::Mat& dst)
{
	int height, width = 0;
	height = src.rows;
	width = src.cols;
	int gray[256] = { 0 };  //记录每个灰度级别下的像素个数
	double gray_prob[256] = { 0 };  //记录灰度分布密度
	double gray_distribution[256] = { 0 };  //记录累计密度
	int gray_new[256] = { 0 };  //均衡化后的灰度值
	int value;
	int sum = width * height;
	//统计每个灰度下的像素个数
	for (int i = 0; i < height; i++)
	{
		uchar* p = src.ptr<uchar>(i);
		for (int j = 0; j < width; j++)
		{
			value = p[j];
			gray[value]++;
		}
	}

	//统计灰度频率
	for (int i = 0; i < 256; i++)
	{
		gray_prob[i] = ((double)gray[i] / sum);
	}

	//计算累计密度
	gray_distribution[0] = gray_prob[0];
	for (int i = 1; i < 256; i++)
	{
		gray_distribution[i] = gray_prob[i] + gray_distribution[i - 1];
	}

	//重新计算均衡化后的灰度值，四舍五入
	for (int i = 0; i < 256; i++)
	{
		gray_new[i] = round(gray_distribution[i] * 255);
	}
	for (int i = 0; i < width; i++)
	{
		uchar* p = dst.ptr<uchar>(i);
		for (int j = 0; j < height; j++)
		{
			p[j] = gray_new[p[j]];
		}
	}
	src.copyTo(dst);
	src.release();
}

double** getGuassionArray(int size, double sigma) {
	int i, j;
	double sum = 0.0;
	int center = size; //以第一个点的坐标为原点，求出中心点的坐标

	double** arr = new double* [size];//建立一个size*size大小的二维数组
	for (i = 0; i < size; ++i)
		arr[i] = new double[size];

	for (i = 0; i < size; ++i)
		for (j = 0; j < size; ++j) {
			arr[i][j] = exp(-((i - center) * (i - center) + (j - center) * (j - center)) / (sigma * sigma * 2));
			sum += arr[i][j];
		}
	for (i = 0; i < size; ++i)
		for (j = 0; j < size; ++j)
			arr[i][j] /= sum;
	return arr;
}

void myGaussian(cv::Mat& src, cv::Mat& dst, int ksize, int sigma)
{
	if (!src.data) return;
	double** arr;
	cv::Mat temp(src.size(), src.type());
	for (int i = 0; i < src.rows; ++i)
		for (int j = 0; j < src.cols; ++j) {
			//边缘不进行处理
			if ((i - 1) > 0 && (i + 1) < src.rows && (j - 1) > 0 && (j + 1) < src.cols) {
				//输入 size & sigma
				arr = getGuassionArray(ksize, sigma);
				temp.at<uchar>(i, j) = 0;
				for (int x = 0; x < 3; ++x) {
					for (int y = 0; y < 3; ++y) {
						temp.at<uchar>(i, j) += arr[x][y] * src.at<uchar>(i + 1 - x, j + 1 - y);
					}
				}
			}
		}
	temp.copyTo(dst);
	temp.release();
}

void myCloseOp(cv::Mat& src, cv::Mat& dst)
{
	Mat temp = Mat::zeros(src.size(), CV_8UC1);
	src.copyTo(temp);
	int height_B = 2;
	int width_B = 2;
	int height = src.rows;
	int width = src.cols;
	int flag = 0;
	//先膨胀
	src.copyTo(temp);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			flag = 1;
			for (int m = i - height_B / 2; m < i + height_B / 2 + 1; m++)
			{
				for (int n = j - width_B / 2; n < j + width_B / 2 + 1; n++)
				{
					if (m < 0 || n < 0 || m >= height || n >= width) continue;
					//自身及领域中若有一个为0
					//则将该点设为0
					if (temp.at<uchar>(i, j) != 0 || temp.at<uchar>(m, n) != 0)
					{
						flag = 0;
						break;
					}
				}
				if (flag == 0)
				{
					break;
				}
			}
			if (flag == 0)
			{
				src.at<uchar>(i, j) = 255;
			}
			else
			{
				src.at<uchar>(i, j) = 0;
			}
		}
	}
	//再腐蚀
	height_B = 1;
	width_B = 1;
	//对于每一个kernel，如果全是白色，则将白色全部变为黑色
	for (int i = 250; i < height - height_B / 2; i++)
	{
		for (int j = width_B / 2; j < width - width_B / 2; j++)
		{
			flag = 0;
			for (int m = i - height_B / 2; m < i + height_B / 2 + 1; m++)
			{
				for (int n = j - width_B / 2; n < j + width_B/2 + 1; n++)
				{
					if(temp.at<uchar>(i, j) == 255 && temp.at<uchar>(m, n) == 255)
					{
						flag = 1;
						continue;
					}
					//有一个不等于255
					else if (temp.at<uchar>(i, j) ==255 && temp.at<uchar>(m, n) == 0)
					{
						flag = 3;
						break;
					}
				}
				if(flag == 3)
				{
					break;
				}
			}
			if (flag == 1)
			{
				for (int n = j - width_B; n < j + width_B; n++)
				{
					src.at<uchar>(i, j) = 0;
				}
			}
			else if(flag == 3)
			{
				src.at<uchar>(i, j) = 255;
			}
		}
	}
	src.copyTo(dst);
}


//求九个数的中值
uchar Median(uchar n1, uchar n2, uchar n3, uchar n4, uchar n5,
	uchar n6, uchar n7, uchar n8, uchar n9) {
	uchar arr[9];
	arr[0] = n1;
	arr[1] = n2;
	arr[2] = n3;
	arr[3] = n4;
	arr[4] = n5;
	arr[5] = n6;
	arr[6] = n7;
	arr[7] = n8;
	arr[8] = n9;
	for (int gap = 9 / 2; gap > 0; gap /= 2)//希尔排序
		for (int i = gap; i < 9; ++i)
			for (int j = i - gap; j >= 0 && arr[j] > arr[j + gap]; j -= gap)
				swap(arr[j], arr[j + gap]);
	return arr[4];//返回中值
}

//中值滤波函数
void MedianFlitering(const Mat& src, Mat& dst) 
{
	if (!src.data)return;
	Mat _dst(src.size(), src.type());
	for (int i = 0; i < src.rows; ++i)
		for (int j = 0; j < src.cols; ++j) {
			if ((i - 1) > 0 && (i + 1) < src.rows && (j - 1) > 0 && (j + 1) < src.cols) {
				_dst.at<uchar>(i, j) = Median(src.at<uchar>(i, j), src.at<uchar>(i + 1, j + 1),
					src.at<uchar>(i + 1, j), src.at<uchar>(i, j + 1), src.at<uchar>(i + 1, j - 1),
					src.at<uchar>(i - 1, j + 1), src.at<uchar>(i - 1, j), src.at<uchar>(i, j - 1),
					src.at<uchar>(i - 1, j - 1));
			}
			else
				_dst.at<uchar>(i, j) = src.at<uchar>(i, j);
		}
	_dst.copyTo(dst);
	_dst.release();
}


std::vector<int> hough_line_detect(cv::Mat img, int threshold)
{
	int row, col;
	int i, k;
	//参数空间的参数极角angle(角度)，极径p;
	int angle, p;

	//累加器
	int** socboard;
	int* buf;
	int w, h;
	w = img.cols;
	h = img.rows;
	int Size;
	int offset;
	std::vector<int> lines;
	//申请累加器空间并初始化
	Size = w * w + h * h;
	Size = 2 * sqrt(Size) + 100;
	offset = Size / 2;
	cout << "offset: " << offset << endl;
	socboard = (int**)malloc(Size * sizeof(int*));
	if (!socboard)
	{
		printf("mem err\n");
		return lines;
	}

	for (i = 0; i < Size; i++)
	{
		socboard[i] = (int*)malloc(181 * sizeof(int));
		if (socboard[i] == NULL)
		{
			printf("buf err\n");
			return lines;
		}
		memset(socboard[i], 0, 181 * sizeof(int));
	}

	//遍历图像并投票
	int src_data;
	p = 0;
	for (row = 0; row < img.rows; row++)
	{
		for (col = 0; col < img.cols; col++)
		{
			//获取像素点
			src_data = img.at<uchar>(row, col);
			if (src_data == 255)
			{
				for (angle = 0; angle < 181; angle++)
				{
					p = col * cos(angle * PI / 180.0) + row * sin(angle * PI / 180.0) + offset;
					//错误处理
					if (p < 0)
					{
						printf("at (%d,%d),angle:%d,p:%d\n", col, row, angle, p);
						printf("warrning!");
						printf("size:%d\n", Size / 2);
						continue;
					}
					//投票计分
					socboard[p][angle]++;
				}
			}
		}
	}

	//遍历计分板，选出符合阈值条件的直线
	//int count = 0;
	int Max = 0;
	int kp, kt, r;
	kp = 0;
	kt = 0;
	for (i = 0; i < Size; i++)//p
	{
		for (k = 0; k < 181; k++)//角度0-180
		{
			if (socboard[i][k] > Max)
			{
				Max = socboard[i][k];
				kp = i - offset;
				kt = k;
			}

			if (socboard[i][k] >= threshold)
			{
				r = i - offset;
				//lines_w.push_back(std::);
				//lines.push_back(-1.0 * float(std::cos(k * PI / 180) / std::sin(k * PI / 180)));
				//lines.push_back(float(r) / std::sin(k * PI / 180));
				lines.push_back(i - Size / 2);//rho
				lines.push_back(k);//angle
				//count++;
			}
		}
	}
	//释放资源
	for (int e = 0; e < Size; e++)
	{
		free(socboard[e]);
	}
	free(socboard);
	return lines;
}

int draw_lane(cv::Mat& src, vector<int> lines, int lines_num)
{
	cout << "lines_num: " << lines_num << endl;
	int final_line_num = 0;
	if (lines.empty() != 0) return 0;
	float rho0 = lines[0], theta0 = PI * lines[1] / 180;
	Point pt1, pt2;
	double a = cos(theta0), b = sin(theta0);
	double x0 = a * rho0, y0 = b * rho0;
	pt1.x = cvRound(x0 + 1000 * (-b));
	pt1.y = cvRound(y0 + 1000 * (a));
	pt2.x = cvRound(x0 - 1500 * (-b));
	pt2.y = cvRound(y0 - 1500 * (a));
	line(src, pt1, pt2, Scalar(0, 255, 0), 1);
	cout << "rho: " << lines[0] << " theta: " << lines[1] << endl;
	final_line_num++;
	for (size_t i = 0; i < lines_num; i++)
	{
		//平行的直线忽略
		if ((lines[2 * i + 1] > 75 && lines[2 * i + 1] < 105))
		{
			break;
		}
		//处理正常的直线
		float rho = lines[2 * i], theta = PI * lines[2 * i + 1] / 180;
		float deita_rho = float(abs(rho0-rho) / abs(rho0));
		float deita_theta = float(abs(theta0 - theta) / abs(theta));
		std::cout << "deita_rho : " << deita_rho << endl;
		std::cout << "deita_theta : " << deita_theta << endl;
		if (deita_rho < 1.5 && deita_theta < 1.5)
		{
			continue;
		}
		else
		{
			rho0 = rho;
			theta0 = theta;
		}
		cout << "rho: " << lines[2 * i] << " theta: " << lines[2 * i + 1] << endl;
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1500 * (-b));
		pt2.y = cvRound(y0 - 1500 * (a));
		line(src, pt1, pt2, Scalar(0, 255, 0), 1);
		final_line_num++;
	}
	 return  final_line_num;
}


void Write2txt(int line_num, vector<int> lines)
{

	ofstream file("C:/Users/PC/Desktop/DIP_homework/predict.txt");
	file << line_num;
	for(size_t i = 0; i < line_num ; i++)
	{
		file << " " << lines[2*i] << " " << lines[2*i + 1];
	}
	file << endl;
	file.close();
}

int main(void)
{
	string src = "C:/Users/PC/Desktop/dip/src/";
	string file = "clips/0531/1492628791892679406/20.jpg";
	int hough_threshold = 50;
	string raw_iamge = src + file;
	//ifstream plist("C:/Users/PC/Desktop/DIP_homework/picture_list.txt", ios::in);
	//100个测试集
	cv::Mat image = cv::imread(raw_iamge);
	int width, height = 0;
	height = image.rows;
	width = image.cols;
	//std::cout << width << endl << height << endl;
	//imshow("src", image);
	//cv::waitKey(6000);

	cv::Mat gray_img = cv::Mat::zeros(height, width, CV_8UC1);
	cv::Mat Gaus_img = cv::Mat::zeros(height, width, CV_8UC1);
	cv::Mat Canny_img = cv::Mat::zeros(height, width, CV_8UC1);
	cv::Mat Hist_gray_img = cv::Mat::zeros(height, width, CV_8UC1);
	cv::Mat Binary_img = cv::Mat::zeros(height, width, CV_8UC1);
	cv::Mat remove_small_img = cv::Mat::zeros(height, width, CV_8UC1);
	cv::Mat close_img = cv::Mat::zeros(height, width, CV_8UC1);
	cv::Mat blur_img = cv::Mat::zeros(height, width, CV_8UC1);
	cv::Mat hough_img = cv::Mat::zeros(height, width, CV_8UC1);
	cv::Mat lane_img = cv::Mat::zeros(height, width, CV_8UC1);

	//创建mask
	Mat mask_up = cv::Mat::zeros(height, width, CV_8UC1);
	Mat up_image = cv::Mat::zeros(height, width, CV_8UC1);
	Rect r1(0, 0, 1280, 250);
	mask_up(r1).setTo(255);
	image.copyTo(up_image, mask_up);
	//imshow("mask", up_image);
	//转灰度图
	myRGB2GRAY(image, gray_img, width, height);
	//imshow("gray", gray_img);
	//直方图均衡化
	myHist(gray_img, Hist_gray_img);
	//imshow("Hist_gray", Hist_gray_img);

	//高斯滤波,输入卷积核size和sigma
	int ksize = 3, sigma = 1.5;
	myGaussian(Hist_gray_img, Gaus_img, ksize, sigma);
	//cv::imshow("Gaus_img",Gaus_img);

	//双阈值二值化
	myGaus2Binary(Gaus_img, Binary_img);
	cv::imshow("New Binary", Binary_img);

	//边缘平滑
	MedianFlitering(Binary_img, blur_img);
	//cv::imshow("blur", blur_img);

	//去除小区域
	RemoveSmallRegion(blur_img, remove_small_img, 20, 1, 1);
	//cv::imshow("remove_small", remove_small_img);

	//闭运算
	myCloseOp(remove_small_img, close_img);
	cv::imshow("close", close_img);

	//霍夫变换
	vector<int> lines;
	lines = hough_line_detect(close_img, hough_threshold);
	int lines_num = lines.size() / 2;
	cout << lines_num << endl;
	//画车道线
	int final_line_num = 0;
	final_line_num = draw_lane(image, lines, lines_num);
	//cv::imshow("res", image);

	//制作完整的图
	Mat mask_down = cv::Mat::zeros(height, width, CV_8UC1);
	Rect r2(0, 250, 1280, 470);
	mask_down(r2).setTo(255);
	image.copyTo(up_image, mask_down);
	cv::imshow("final", up_image);

	//将直线信息存入txt文件
	cout << final_line_num << endl;
	string out_path = "C:/Users/PC/Desktop/dip/result/";
	string final_path = out_path + file;
	imwrite(final_path, up_image);
	waitKey(0);
	return 0;
}
