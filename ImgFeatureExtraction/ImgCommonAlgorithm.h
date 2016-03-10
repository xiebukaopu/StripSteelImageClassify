#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

// 数字图像的公用的算法
class __declspec(dllexport) ImgCommonAlgorithm
{
public:
	ImgCommonAlgorithm( void );
	
	~ImgCommonAlgorithm( void );

	static bool weightedAverageGrayScale( cv::Mat &src, cv::Mat &det );	// 加权平均灰度化处理

	static double averageImagePixel( cv::Mat &src );	// 求图像中所有元素的平均值

	static void showImage( std::string name, cv::Mat &src );			// 显示图像
};

