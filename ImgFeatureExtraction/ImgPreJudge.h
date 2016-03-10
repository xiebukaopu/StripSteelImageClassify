#pragma once
// #pragma warning(disable: 4251)

#include <opencv2/opencv.hpp>

#define IMGPREJUDGE_API __declspec(dllexport)


// 缺陷图像预判断的主类，主要用于快速实时地判断输入的图像是否有缺陷
class IMGPREJUDGE_API ImgPreJudge
{
	// template class IMGPREJUDGE_API cv::Mat;

public:
	ImgPreJudge( cv::Mat &inputImg );		
	
	~ImgPreJudge(void);	
	

private: 
	cv::Mat m_mInputImg;						// 需要进行预处理的图像
	cv::Mat m_mGradImg;							// 对原图像进行梯度化处理后的梯度图像
	cv::Mat m_mBinImg;							// 二值化图像
	// int m_iThreshold;						// 二值化阈值，大于该阈值符合统计条件
	// int m_iPointCnt;							// 二值化阈值，大于该值认为是异常点


public:
	cv::Mat getInputImg();						// 输出原图像
	cv::Mat getGradImg();						// 输出梯度图像
	cv::Mat getBinImg();						// 输出二值化图像
	bool isFaultImg( double scale );			// 判断是否是缺陷图像
	void calculateGrad();						// 计算梯度图像
	void binaryImg( double k = 2.6, int pointCnt = 12 );	// 对梯度图像进行二值化图像

private:
	int faltung( int a[], int aLen, int b[], int bLen );		// 卷积运算


};

