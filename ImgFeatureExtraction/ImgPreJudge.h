#pragma once
// #pragma warning(disable: 4251)

#include <opencv2/opencv.hpp>

#define IMGPREJUDGE_API __declspec(dllexport)


// ȱ��ͼ��Ԥ�жϵ����࣬��Ҫ���ڿ���ʵʱ���ж������ͼ���Ƿ���ȱ��
class IMGPREJUDGE_API ImgPreJudge
{
	// template class IMGPREJUDGE_API cv::Mat;

public:
	ImgPreJudge( cv::Mat &inputImg );		
	
	~ImgPreJudge(void);	
	

private: 
	cv::Mat m_mInputImg;						// ��Ҫ����Ԥ�����ͼ��
	cv::Mat m_mGradImg;							// ��ԭͼ������ݶȻ��������ݶ�ͼ��
	cv::Mat m_mBinImg;							// ��ֵ��ͼ��
	// int m_iThreshold;						// ��ֵ����ֵ�����ڸ���ֵ����ͳ������
	// int m_iPointCnt;							// ��ֵ����ֵ�����ڸ�ֵ��Ϊ���쳣��


public:
	cv::Mat getInputImg();						// ���ԭͼ��
	cv::Mat getGradImg();						// ����ݶ�ͼ��
	cv::Mat getBinImg();						// �����ֵ��ͼ��
	bool isFaultImg( double scale );			// �ж��Ƿ���ȱ��ͼ��
	void calculateGrad();						// �����ݶ�ͼ��
	void binaryImg( double k = 2.6, int pointCnt = 12 );	// ���ݶ�ͼ����ж�ֵ��ͼ��

private:
	int faltung( int a[], int aLen, int b[], int bLen );		// �������


};

