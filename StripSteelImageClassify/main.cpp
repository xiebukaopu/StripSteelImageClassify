#include "ImgCommonAlgorithm.h"
#include "ImgPreJudge.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// 前期用作测试函数，用于测试项目 ImgFeatureExtration 中的算法的正确性
// 后期主要用于整个项目的入口函数
int main( )
{
	Mat src = imread("1.tif");

	if( src.empty() )
	{
		cout << "图片加载失败" << endl;
		system("pause");
		return -1;
	}

	Mat filterPic;
	GaussianBlur(src,filterPic,Size(5,5),0,0);

	ImgPreJudge ipj(filterPic);
	ImgCommonAlgorithm::showImage("高斯模糊图像",ipj.getInputImg());

	ipj.calculateGrad();
	ImgCommonAlgorithm::showImage("梯度图像",ipj.getGradImg());

	ipj.binaryImg(2.6,13);
	ImgCommonAlgorithm::showImage("二值化图像",ipj.getBinImg());

	/*Mat guussBinImg;
	GaussianBlur(ipj.getBinImg(),guussBinImg,Size(5,5),0,0);
	imshow("二值化模糊图像",guussBinImg);*/

	waitKey(0);

	return 0;
}