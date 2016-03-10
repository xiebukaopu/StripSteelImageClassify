#include "ImgCommonAlgorithm.h"
#include "ImgPreJudge.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// ǰ���������Ժ��������ڲ�����Ŀ ImgFeatureExtration �е��㷨����ȷ��
// ������Ҫ����������Ŀ����ں���
int main( )
{
	Mat src = imread("1.tif");

	if( src.empty() )
	{
		cout << "ͼƬ����ʧ��" << endl;
		system("pause");
		return -1;
	}

	Mat filterPic;
	GaussianBlur(src,filterPic,Size(5,5),0,0);

	ImgPreJudge ipj(filterPic);
	ImgCommonAlgorithm::showImage("��˹ģ��ͼ��",ipj.getInputImg());

	ipj.calculateGrad();
	ImgCommonAlgorithm::showImage("�ݶ�ͼ��",ipj.getGradImg());

	ipj.binaryImg(2.6,13);
	ImgCommonAlgorithm::showImage("��ֵ��ͼ��",ipj.getBinImg());

	/*Mat guussBinImg;
	GaussianBlur(ipj.getBinImg(),guussBinImg,Size(5,5),0,0);
	imshow("��ֵ��ģ��ͼ��",guussBinImg);*/

	waitKey(0);

	return 0;
}