#include "ImgCommonAlgorithm.h"

ImgCommonAlgorithm::ImgCommonAlgorithm(void)
{

}


ImgCommonAlgorithm::~ImgCommonAlgorithm(void)
{

}


/**
 * 	FullName:   ImgCommonAlgorithm::weightedAverageGrayScale
 * 	Author:		zhouqingshan	2016/03/08
 * 
 *	\access		
 *		public static 
 *
 * 	\param			
 *		cv::Mat &src     �����ԭͼ��
 *		cv::Mat &det     �ҶȻ����ͼ��
 *
 * 	\return		
 *		bool		����ҶȻ�ʧ�ܣ�����false�� �ҶȻ��ɹ�������true��
 *
 * 	\description
 *		��Ȩƽ���ķ�����ͼ����лҶȻ��Ĵ���	 
 **/
bool ImgCommonAlgorithm::weightedAverageGrayScale( cv::Mat &src, cv::Mat &det )
{
	// ԭͼ��Ϊ�� 
	if ( src.empty() )
	{
		return false;
	}

	// ԭͼ����ǻҶ�ͼ��ֱ�ӷ���
    if ( src.channels() == 1 )
	{
		det = src.clone();		// ֱ�Ӹ���ͼ�񼴿� or   src.copyTo(det)

		return true;
	}

	int srcImgRows = src.rows;					// ����
	int srcImgCols = src.cols;					// ����

	// ��ʼ���Ҷ�ͼ��
	det = cv::Mat::zeros(srcImgRows,srcImgCols,CV_8UC1);

	// �������ݵ�����������ͬ�Ĳ���(�˴�Ӧ���ع�����ʽ���򵥵ķ�ʽ�����������﷨���ԣ���δ�뵽���ʵķ����ع���
	switch(src.depth())
	{
	case CV_8U:
		// ����ͼ��
		for( int i = 0 ; i < srcImgRows ; ++i )
		{
			uchar *srcData = src.ptr<uchar>(i);
			uchar *detData = det.ptr<uchar>(i);
			for( int j = 0 ; j < srcImgCols ; ++j )
			{
				detData[j] = (int)(0.114 * srcData[3*j] + 
					0.587 * srcData[3*j+1] + 0.299 * srcData[3*j+2]);
			}
		}
		break;

	case CV_8S:
		// ����ͼ��
		for( int i = 0 ; i < srcImgRows ; ++i )
		{
			char *srcData = src.ptr<char>(i);
			char *detData = det.ptr<char>(i);
			for( int j = 0 ; j < srcImgCols ; ++j )
			{
				detData[j] = (int)(0.114 * srcData[3*j] + 
					0.587 * srcData[3*j+1] + 0.299 * srcData[3*j+2]);
			}
		}
		break;

	case CV_16U:
		// ����ͼ��
		for( int i = 0 ; i < srcImgRows ; ++i )
		{
			unsigned short *srcData = src.ptr<unsigned short>(i);
			unsigned short *detData = det.ptr<unsigned short>(i);
			for( int j = 0 ; j < srcImgCols ; ++j )
			{
				detData[j] = (int)(0.114 * srcData[3*j] + 
					0.587 * srcData[3*j+1] + 0.299 * srcData[3*j+2]);
			}
		}
		break;

	
	case CV_16S:
		// ����ͼ��
		for( int i = 0 ; i < srcImgRows ; ++i )
		{
			short *srcData = src.ptr<short>(i);
			short *detData = det.ptr<short>(i);
			for( int j = 0 ; j < srcImgCols ; ++j )
			{
				detData[j] = (int)(0.114 * srcData[3*j] + 
					0.587 * srcData[3*j+1] + 0.299 * srcData[3*j+2]);
			}
		}
		break;

	case CV_32F:
		// ����ͼ��
		for( int i = 0 ; i < srcImgRows ; ++i )
		{
			float *srcData = src.ptr<float>(i);
			float *detData = det.ptr<float>(i);
			for( int j = 0 ; j < srcImgCols ; ++j )
			{
				detData[j] = (int)(0.114 * srcData[3*j] + 
					0.587 * srcData[3*j+1] + 0.299 * srcData[3*j+2]);
			}
		}
		break;

	case CV_32S:
		// ����ͼ��
		for( int i = 0 ; i < srcImgRows ; ++i )
		{
			int *srcData = src.ptr<int>(i);
			int *detData = det.ptr<int>(i);
			for( int j = 0 ; j < srcImgCols ; ++j )
			{
				detData[j] = (int)(0.114 * srcData[3*j] + 
					0.587 * srcData[3*j+1] + 0.299 * srcData[3*j+2]);
			}
		}
		break;

	case CV_64F:
		// ����ͼ��
		for( int i = 0 ; i < srcImgRows ; ++i )
		{
			double *srcData = src.ptr<double>(i);
			double *detData = det.ptr<double>(i);
			for( int j = 0 ; j < srcImgCols ; ++j )
			{
				detData[j] = (int)(0.114 * srcData[3*j] + 
					0.587 * srcData[3*j+1] + 0.299 * srcData[3*j+2]);
			}
		}
		break;
	}

	return true;
}

/**
 * 	FullName:   ImgCommonAlgorithm::averageImagePixel
 * 	Author:		zhouqingshan	2016/03/08
 * 
 *	\access		
 *		public static 
 *
 * 	\param			
 *		cv::Mat &src
 *
 * 	\return		
 *		double   �������ص�ƽ��ֵ
 *
 * 	\description
 *		��ͼ�����������ص�ƽ��ֵ	 
 **/
double ImgCommonAlgorithm::averageImagePixel( cv::Mat &src )
{
	if ( src.empty() )
	{
		return 0.0;
	}

	int srcImgRows = src.rows;
	int srcImgCols = src.cols * src.channels();

	double sum = 0.0;

	// �������ݵ�����(�˴�Ӧ���ع�����ʽ���򵥵ķ�ʽ�����������﷨���ԣ���δ�뵽���ʵķ����ع���
	switch(src.depth())
	{
	case CV_8U:
		// ����ͼ��
		for( int i = 0 ; i < srcImgRows ; ++i )
		{
			uchar *srcData = src.ptr<uchar>(i);
			for( int j = 0 ; j < srcImgCols ; ++j )
			{
				sum += srcData[j];
			}
		}
		break;

	case CV_8S:
		// ����ͼ��
		for( int i = 0 ; i < srcImgRows ; ++i )
		{
			char *srcData = src.ptr<char>(i);
			for( int j = 0 ; j < srcImgCols ; ++j )
			{
				sum += srcData[j];
			}
		}
		break;

	case CV_16U:
		// ����ͼ��
		for( int i = 0 ; i < srcImgRows ; ++i )
		{
			unsigned short *srcData = src.ptr<unsigned short>(i);
			
			for( int j = 0 ; j < srcImgCols ; ++j )
			{
				sum += srcData[j];
			}
		}
		break;


	case CV_16S:
		// ����ͼ��
		for( int i = 0 ; i < srcImgRows ; ++i )
		{
			short *srcData = src.ptr<short>(i);
	
			for( int j = 0 ; j < srcImgCols ; ++j )
			{
				sum += srcData[j];
			}
		}
		break;

	case CV_32F:
		// ����ͼ��
		for( int i = 0 ; i < srcImgRows ; ++i )
		{
			float *srcData = src.ptr<float>(i);
			
			for( int j = 0 ; j < srcImgCols ; ++j )
			{
				sum += srcData[j];
			}
		}
		break;

	case CV_32S:
		// ����ͼ��
		for( int i = 0 ; i < srcImgRows ; ++i )
		{
			int *srcData = src.ptr<int>(i);
			
			for( int j = 0 ; j < srcImgCols ; ++j )
			{
				sum += srcData[j];
			}
		}
		break;

	case CV_64F:
		// ����ͼ��
		for( int i = 0 ; i < srcImgRows ; ++i )
		{
			double *srcData = src.ptr<double>(i);
		
			for( int j = 0 ; j < srcImgCols ; ++j )
			{
				sum += srcData[j];
			}
		}
		break;
	}

	return (sum / srcImgRows / srcImgCols) ;
}

// ��ʾͼ��
void ImgCommonAlgorithm::showImage( std::string name, cv::Mat &src )
{
	if ( src.empty() )
	{
		std::cout << "ͼƬΪ�գ��޷���ʾ��" << std::endl;
		return;
	}

	imshow(name,src);
}
