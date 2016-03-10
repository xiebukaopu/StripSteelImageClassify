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
 *		cv::Mat &src     输入的原图像
 *		cv::Mat &det     灰度化后的图像
 *
 * 	\return		
 *		bool		如果灰度化失败，返回false； 灰度化成功，返回true；
 *
 * 	\description
 *		加权平均的方法对图像进行灰度化的处理	 
 **/
bool ImgCommonAlgorithm::weightedAverageGrayScale( cv::Mat &src, cv::Mat &det )
{
	// 原图像为空 
	if ( src.empty() )
	{
		return false;
	}

	// 原图像就是灰度图像，直接返回
    if ( src.channels() == 1 )
	{
		det = src.clone();		// 直接复制图像即可 or   src.copyTo(det)

		return true;
	}

	int srcImgRows = src.rows;					// 行数
	int srcImgCols = src.cols;					// 列数

	// 初始化灰度图像
	det = cv::Mat::zeros(srcImgRows,srcImgCols,CV_8UC1);

	// 根据数据的类型做出不同的操作(此处应该重构成形式更简单的方式，但是由于语法特性，暂未想到合适的方法重构）
	switch(src.depth())
	{
	case CV_8U:
		// 遍历图像
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
		// 遍历图像
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
		// 遍历图像
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
		// 遍历图像
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
		// 遍历图像
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
		// 遍历图像
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
		// 遍历图像
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
 *		double   返回像素的平均值
 *
 * 	\description
 *		求图像中所有像素的平均值	 
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

	// 根据数据的类型(此处应该重构成形式更简单的方式，但是由于语法特性，暂未想到合适的方法重构）
	switch(src.depth())
	{
	case CV_8U:
		// 遍历图像
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
		// 遍历图像
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
		// 遍历图像
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
		// 遍历图像
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
		// 遍历图像
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
		// 遍历图像
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
		// 遍历图像
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

// 显示图像
void ImgCommonAlgorithm::showImage( std::string name, cv::Mat &src )
{
	if ( src.empty() )
	{
		std::cout << "图片为空，无法显示！" << std::endl;
		return;
	}

	imshow(name,src);
}
