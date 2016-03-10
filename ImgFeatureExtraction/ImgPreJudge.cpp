#include "ImgPreJudge.h"
#include "ImgCommonAlgorithm.h"

using namespace cv;

ImgPreJudge::ImgPreJudge(Mat &inputImg )
{
	if( inputImg.empty() )
	{
		return;
	}

	this->m_mInputImg = inputImg.clone();
}


ImgPreJudge::~ImgPreJudge(void)
{

}


Mat ImgPreJudge::getInputImg()
{
	return this->m_mInputImg;
}


Mat ImgPreJudge::getGradImg()
{
	return this->m_mGradImg;
}

Mat ImgPreJudge::getBinImg()
{
	return this->m_mBinImg;
}


/**
 * 	FullName:   ImgPreJudge::calculateGrad
 * 	Author:		zhouqingshan	2016/03/08
 * 
 *	\access		
 *		public 
 *
 * 	\param			
 *		
 *
 * 	\return		
 *		void		
 *
 * 	\description
 *		计算梯度图像	 
 **/
void ImgPreJudge::calculateGrad()
{
	// 没有输入图像
	if( m_mInputImg.empty() )
	{
		return;
	}

	// 将 m_mInputImg 转化为灰度图
	Mat grayImg;
	ImgCommonAlgorithm::weightedAverageGrayScale(m_mInputImg,grayImg);

	int grayImgRows = grayImg.rows;
	int grayImgCols = grayImg.cols;

	m_mGradImg = Mat::zeros(grayImgRows,grayImgCols,CV_8UC1);	// 初始化梯度图

	int grayImgPixel[9];				// 存放中心像素及其周围的9个像素
	memset(grayImgPixel,0,9);

	// Kirsch算子的八个模板
	int kirschTemplate[8][9] = {
		{5,-3,-3,5,0,-3,5,-3,-3},
		{5,5,5,-3,0,-3,-3,-3,-3},
		{5,5,-3,5,0,-3,-3,-3,-3},
		{-3,-3,-3,5,0,-3,5,5,-3},
		{-3,-3,-3,-3,0,-3,5,5,5},
		{-3,-3,-3,-3,0,5,-3,5,5},
		{-3,-3,5,-3,0,5,-3,-3,5},
		{-3,5,5,-3,0,5,-3,-3,-3}
	};

	// 遍历原图像，进行卷积运算( 边界就只有一个像素，未做边界处理 )
	for ( int i = 1 ; i < grayImgRows - 1 ; ++i )		// 忽略边界的像素
	{
		// 已经知道是灰度图像了，直接用 uchar 类型
		uchar *gradData = m_mGradImg.ptr<uchar>(i);

		uchar *grayData = grayImg.ptr<uchar>(i - 1 );
		uchar *grayDatai = grayImg.ptr<uchar>(i);
		uchar *grayDataii = grayImg.ptr<uchar>(i + 1);
		for ( int j = 1 ; j < grayImgCols - 1 ; ++j )
		{
			grayImgPixel[0] = grayData[j-1];	// 灰度图像 i - 1 行的3个元素
			grayImgPixel[1] = grayData[j];
			grayImgPixel[2] = grayData[j+1];

			grayImgPixel[3] = grayDatai[j-1];	// 灰度图像 i 行的3个元素
			grayImgPixel[4] = grayDatai[j];
			grayImgPixel[5] = grayDatai[j+1];

			grayImgPixel[6] = grayDataii[j-1];	// 灰度图像 i + 1 行的3个元素
			grayImgPixel[7] = grayDataii[j];
			grayImgPixel[8] = grayDataii[j+1];

			// 将灰度图像中的像素与模板做卷积运算
			int v0 = faltung(grayImgPixel,9,kirschTemplate[0],9);
			int v1 = faltung(grayImgPixel,9,kirschTemplate[1],9);
			int v2 = faltung(grayImgPixel,9,kirschTemplate[2],9);
			int v3 = faltung(grayImgPixel,9,kirschTemplate[3],9);
			
			int tempMax = max(max(v0,v1),max(v2,v3));

			gradData[j] = max(0,min(255,tempMax));		// 四个中的最大值赋给梯度图像
		}
	}

	// 尝试高斯模糊梯度图像
	/*Mat det;
	GaussianBlur(m_mGradImg,det,Size(5,5),0,0);

	m_mGradImg = det.clone();*/
}

/**
 * 	FullName:   ImgPreJudge::faltung
 * 	Author:		zhouqingshan	2016/03/08
 * 
 *	\access		
 *		private 
 *
 * 	\param			
 *		int a[],	a数组
 *		int aLen,   a数组的长度
 *		int b[],	b数组
 *		int bLen	b数组的长度
 *
 * 	\return		
 *		int
 *
 * 	\description
 *		卷积运算，就是把数组a与数组b的元素依次相乘得到一个结果	 
 **/
int ImgPreJudge::faltung( int a[], int aLen, int b[], int bLen )
{
	// 两个数组必须相等
	if( aLen != bLen || a == NULL || b == NULL )
	{
		return -1;
	}

	int ret = 0;
	for( int i = 0 ; i < aLen ; ++i )
	{
		ret += a[i] * b[i];
	}

	return ret;
}


/**
 * 	FullName:   ImgPreJudge::binaryImg
 * 	Author:		zhouqingshan	2016/03/08
 * 
 *	\access		
 *		public 
 *
 * 	\param			
 *		k:	阈值的系数，在5*5像素的区域内统计区域内梯度大于该阈值的像素个数
 *		pointCnt:	若像素个数大于pointCnt值，则认为改点是异常点
 *		
 * 	\return		
 *		void
 *
 * 	\description
 *		对图像进行二值化处理	 
 **/
void ImgPreJudge::binaryImg( double k, int pointCnt )
{
	// 阈值等于： k * Average(GradImg) , 即与梯度图像的像素平均值成一定的比例

	// 如果梯度图的数据为空，则计算梯度图
	if( m_mGradImg.empty() )
	{
		calculateGrad();	
	}

	int gradImgRows = m_mGradImg.rows;
	int gradImgCols = m_mGradImg.cols;

	// 初始化为所有元素为255的矩阵
	m_mBinImg = Mat(gradImgRows,gradImgCols,CV_8UC1,Scalar::all(255));	

	// 求阈值
	int threshold = k * ImgCommonAlgorithm::averageImagePixel(m_mGradImg);

	// 在 5*5 的范围内寻找满足相应条件的值
	for ( int i = 0 ; i <= gradImgRows - 5 ; ++i )		// 二值化图像的右边界和下边界无内容
	{
		uchar *binImgData = m_mBinImg.ptr<uchar>(i);

		for( int j = 0 ; j <= gradImgCols - 5 ; ++j )
		{
			// 开始统计
			int cnt = 0;
			for ( int m = 0 ; m < 5 ; ++m )
			{
				uchar *gradImgData = m_mGradImg.ptr<uchar>( i + m );
				for( int n = 0 ; n < 5 ; ++n )
				{
					if ( gradImgData[j+n] > threshold )
					{
						++cnt;		// 大于指定的阈值则增加计数值
					}
				}
			}

			// 计数值大于指定的点数，则为异常点，置为0； 反之，置为255
			cnt > pointCnt ? (binImgData[j] = 0) : (binImgData[j] = 255);
		}
	}

}


/**
 * 	FullName:   ImgPreJudge::isFaultImg
 * 	Author:		zhouqingshan	2016/03/08
 * 
 *	\access		
 *		public 
 *
 * 	\param			
 *		double scalar,  比例值，当图片中的异常点的数量超过这一比例的时候
 *						表明该图片中存在缺陷，需要进行之后的处理
 *
 * 	\return		
 *		bool	存在缺陷，返回true; 不存在缺陷，返回false
 *
 * 	\description
 *		判断一副二值化的图像是否存在缺陷	 
 **/
bool ImgPreJudge::isFaultImg( double scalar )
{
	if( m_mGradImg.empty() )
	{	
		// 计算图像的梯度图像
		calculateGrad();
	}

	if( m_mBinImg.empty() )
	{
		// 二值化图像
		binaryImg( );
	}

	int binImgRows = m_mBinImg.rows;
	int binImgCols = m_mBinImg.cols;

	// 根据参数的比例值确定确定缺陷的异常点的个数是多少
	int threshold = (int)( binImgRows * binImgCols * scalar );

	int cnt = 0;
	for( int i = 0 ; i < binImgRows ; ++i )
	{
		uchar *binImgData = m_mBinImg.ptr<uchar>(i);
		for( int j = 0 ; j < binImgCols ; ++j )
		{
			if ( binImgData[j] == 0 )
			{
				++cnt;
			}
		}
	}

	return cnt > threshold ? true : false;
}

