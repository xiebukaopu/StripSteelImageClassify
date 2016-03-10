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
 *		�����ݶ�ͼ��	 
 **/
void ImgPreJudge::calculateGrad()
{
	// û������ͼ��
	if( m_mInputImg.empty() )
	{
		return;
	}

	// �� m_mInputImg ת��Ϊ�Ҷ�ͼ
	Mat grayImg;
	ImgCommonAlgorithm::weightedAverageGrayScale(m_mInputImg,grayImg);

	int grayImgRows = grayImg.rows;
	int grayImgCols = grayImg.cols;

	m_mGradImg = Mat::zeros(grayImgRows,grayImgCols,CV_8UC1);	// ��ʼ���ݶ�ͼ

	int grayImgPixel[9];				// ����������ؼ�����Χ��9������
	memset(grayImgPixel,0,9);

	// Kirsch���ӵİ˸�ģ��
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

	// ����ԭͼ�񣬽��о������( �߽��ֻ��һ�����أ�δ���߽紦�� )
	for ( int i = 1 ; i < grayImgRows - 1 ; ++i )		// ���Ա߽������
	{
		// �Ѿ�֪���ǻҶ�ͼ���ˣ�ֱ���� uchar ����
		uchar *gradData = m_mGradImg.ptr<uchar>(i);

		uchar *grayData = grayImg.ptr<uchar>(i - 1 );
		uchar *grayDatai = grayImg.ptr<uchar>(i);
		uchar *grayDataii = grayImg.ptr<uchar>(i + 1);
		for ( int j = 1 ; j < grayImgCols - 1 ; ++j )
		{
			grayImgPixel[0] = grayData[j-1];	// �Ҷ�ͼ�� i - 1 �е�3��Ԫ��
			grayImgPixel[1] = grayData[j];
			grayImgPixel[2] = grayData[j+1];

			grayImgPixel[3] = grayDatai[j-1];	// �Ҷ�ͼ�� i �е�3��Ԫ��
			grayImgPixel[4] = grayDatai[j];
			grayImgPixel[5] = grayDatai[j+1];

			grayImgPixel[6] = grayDataii[j-1];	// �Ҷ�ͼ�� i + 1 �е�3��Ԫ��
			grayImgPixel[7] = grayDataii[j];
			grayImgPixel[8] = grayDataii[j+1];

			// ���Ҷ�ͼ���е�������ģ�����������
			int v0 = faltung(grayImgPixel,9,kirschTemplate[0],9);
			int v1 = faltung(grayImgPixel,9,kirschTemplate[1],9);
			int v2 = faltung(grayImgPixel,9,kirschTemplate[2],9);
			int v3 = faltung(grayImgPixel,9,kirschTemplate[3],9);
			
			int tempMax = max(max(v0,v1),max(v2,v3));

			gradData[j] = max(0,min(255,tempMax));		// �ĸ��е����ֵ�����ݶ�ͼ��
		}
	}

	// ���Ը�˹ģ���ݶ�ͼ��
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
 *		int a[],	a����
 *		int aLen,   a����ĳ���
 *		int b[],	b����
 *		int bLen	b����ĳ���
 *
 * 	\return		
 *		int
 *
 * 	\description
 *		������㣬���ǰ�����a������b��Ԫ��������˵õ�һ�����	 
 **/
int ImgPreJudge::faltung( int a[], int aLen, int b[], int bLen )
{
	// ��������������
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
 *		k:	��ֵ��ϵ������5*5���ص�������ͳ���������ݶȴ��ڸ���ֵ�����ظ���
 *		pointCnt:	�����ظ�������pointCntֵ������Ϊ�ĵ����쳣��
 *		
 * 	\return		
 *		void
 *
 * 	\description
 *		��ͼ����ж�ֵ������	 
 **/
void ImgPreJudge::binaryImg( double k, int pointCnt )
{
	// ��ֵ���ڣ� k * Average(GradImg) , �����ݶ�ͼ�������ƽ��ֵ��һ���ı���

	// ����ݶ�ͼ������Ϊ�գ�������ݶ�ͼ
	if( m_mGradImg.empty() )
	{
		calculateGrad();	
	}

	int gradImgRows = m_mGradImg.rows;
	int gradImgCols = m_mGradImg.cols;

	// ��ʼ��Ϊ����Ԫ��Ϊ255�ľ���
	m_mBinImg = Mat(gradImgRows,gradImgCols,CV_8UC1,Scalar::all(255));	

	// ����ֵ
	int threshold = k * ImgCommonAlgorithm::averageImagePixel(m_mGradImg);

	// �� 5*5 �ķ�Χ��Ѱ��������Ӧ������ֵ
	for ( int i = 0 ; i <= gradImgRows - 5 ; ++i )		// ��ֵ��ͼ����ұ߽���±߽�������
	{
		uchar *binImgData = m_mBinImg.ptr<uchar>(i);

		for( int j = 0 ; j <= gradImgCols - 5 ; ++j )
		{
			// ��ʼͳ��
			int cnt = 0;
			for ( int m = 0 ; m < 5 ; ++m )
			{
				uchar *gradImgData = m_mGradImg.ptr<uchar>( i + m );
				for( int n = 0 ; n < 5 ; ++n )
				{
					if ( gradImgData[j+n] > threshold )
					{
						++cnt;		// ����ָ������ֵ�����Ӽ���ֵ
					}
				}
			}

			// ����ֵ����ָ���ĵ�������Ϊ�쳣�㣬��Ϊ0�� ��֮����Ϊ255
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
 *		double scalar,  ����ֵ����ͼƬ�е��쳣�������������һ������ʱ��
 *						������ͼƬ�д���ȱ�ݣ���Ҫ����֮��Ĵ���
 *
 * 	\return		
 *		bool	����ȱ�ݣ�����true; ������ȱ�ݣ�����false
 *
 * 	\description
 *		�ж�һ����ֵ����ͼ���Ƿ����ȱ��	 
 **/
bool ImgPreJudge::isFaultImg( double scalar )
{
	if( m_mGradImg.empty() )
	{	
		// ����ͼ����ݶ�ͼ��
		calculateGrad();
	}

	if( m_mBinImg.empty() )
	{
		// ��ֵ��ͼ��
		binaryImg( );
	}

	int binImgRows = m_mBinImg.rows;
	int binImgCols = m_mBinImg.cols;

	// ���ݲ����ı���ֵȷ��ȷ��ȱ�ݵ��쳣��ĸ����Ƕ���
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

