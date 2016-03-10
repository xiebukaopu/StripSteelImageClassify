#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

// ����ͼ��Ĺ��õ��㷨
class __declspec(dllexport) ImgCommonAlgorithm
{
public:
	ImgCommonAlgorithm( void );
	
	~ImgCommonAlgorithm( void );

	static bool weightedAverageGrayScale( cv::Mat &src, cv::Mat &det );	// ��Ȩƽ���ҶȻ�����

	static double averageImagePixel( cv::Mat &src );	// ��ͼ��������Ԫ�ص�ƽ��ֵ

	static void showImage( std::string name, cv::Mat &src );			// ��ʾͼ��
};

