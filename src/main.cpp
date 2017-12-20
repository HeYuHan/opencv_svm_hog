//显示图像文件  
#include <iostream>    
#include <fstream>    
#include <string>    
#include <vector> 
#include <opencv2/opencv.hpp>  
#include<opencv2/ml.hpp>
using namespace std;
using namespace cv;

#pragma comment(linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"")  
void train_data(const char* data_path,const char* save_path);
void svm_test(const char* svn_data_path, const char* test_data_path);

int main()
{
	train_data("Resource/train_data.txt","svm_data.xml");
	return 1;
	vector<string> img_path;
	vector<int> img_label;
	
	const char* air_label = "airplanes";
	const char* train_dir_path = "Resource/train_images";
	char data_path[128] = {0};
	sprintf(data_path, "%s/%s.txt", train_dir_path, air_label);
	ifstream svm_data(data_path);
	if (svm_data.fail())return -1;
	string fileName;
	while (getline(svm_data, fileName))
	{
		
		char full_path[128] = { 0 };
		sprintf(full_path, "%s/%s/%s", train_dir_path, air_label, fileName.c_str());
		printf("%s\n", full_path);
		img_path.push_back(string(full_path));
	}
	svm_data.close();
	Mat data_mat, res_mat;
	int nImgNum = img_path.size();
	res_mat = Mat::zeros(nImgNum, 1, CV_32FC1);
	Mat src;
	Mat trainImg = Mat::zeros(64, 64, CV_8UC3);//需要分析的图片  
	
	for (string::size_type i = 0; i != img_path.size(); i++)
	{
		src = imread(img_path[i].c_str(), 1);
		resize(src, trainImg, Size(64, 64), 0, 0, INTER_CUBIC);
		
		HOGDescriptor hog = HOGDescriptor(cvSize(64, 64), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);  //具体意思见参考文章1,2       
		vector<float>descriptors;//结果数组  
		hog.compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //调用计算函数开始计算    
		if (i == 0)
		{
			data_mat = Mat::zeros(nImgNum, descriptors.size(), CV_32FC1); //根据输入图片大小进行分配空间
		}
		int n = 0;
		for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
		{
			data_mat.at<float>(i, n) = *iter;
			n++;
		}
		res_mat.at<float>(i, 0) = i%2;

	}

	CvSVM svm;//新建一个SVM      
	CvSVMParams param;//这里是参数  
	CvTermCriteria criteria;
	criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria);
	/*
	SVM种类：CvSVM::C_SVC
	Kernel的种类：CvSVM::RBF
	degree：10.0（此次不使用）
	gamma：8.0
	coef0：1.0（此次不使用）
	C：10.0
	nu：0.5（此次不使用）
	p：0.1（此次不使用）
	然后对训练数据正规化处理，并放在CvMat型的数组里。
	*/
	//☆☆☆☆☆☆☆☆☆(5)SVM学习☆☆☆☆☆☆☆☆☆☆☆☆           
	svm.train(data_mat, res_mat, Mat(), Mat(), param);//训练啦      
													//☆☆利用训练数据和确定的学习参数,进行SVM学习☆☆☆☆       
	svm.save("SVM_DATA.xml");
	
	return 1;

	//const char *pstrImageName = "Resource/train_images/airplanes/image_0001.jpg";
	//const char *pstrWindowsTitle = "OpenCV";

	////从文件中读取图像  
	//IplImage *pImage = cvLoadImage(pstrImageName, CV_LOAD_IMAGE_UNCHANGED);

	////创建窗口  
	//cvNamedWindow(pstrWindowsTitle, CV_WINDOW_AUTOSIZE);

	////在指定窗口中显示图像  
	//cvShowImage(pstrWindowsTitle, pImage);

	////等待按键事件  
	//cvWaitKey();

	//cvDestroyWindow(pstrWindowsTitle);
	//cvReleaseImage(&pImage);
	return 0;
}
void train_data(const char* data_path, const char* save_path)
{
	vector<string> img_path;
	vector<int> img_label;
	int index = 0;
	ifstream svm_data(data_path);
	if (svm_data.fail())return;
	string line;
	while (getline(svm_data, line))
	{
		if (index % 2 == 0)
		{
			img_label.push_back(atoi(line.c_str()));
		}
		else
		{
			img_path.push_back(line);
		}
		
		index++;
	}
	svm_data.close();
	Mat data_mat, res_mat;
	int nImgNum = img_label.size();
	res_mat = Mat::zeros(nImgNum, 1, CV_32FC1);
	Mat src;
	Mat trainImg = Mat::zeros(64, 64, CV_8UC3);//需要分析的图片  

	for (string::size_type i = 0; i != nImgNum; i++)
	{
		src = imread(img_path[i].c_str(), 1);
		resize(src, trainImg, Size(64, 64), 0, 0, INTER_CUBIC);

		HOGDescriptor hog = HOGDescriptor(cvSize(64, 64), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);  //具体意思见参考文章1,2       
		vector<float>descriptors;//结果数组  
		hog.compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //调用计算函数开始计算    
		if (i == 0)
		{
			data_mat = Mat::zeros(nImgNum, descriptors.size(), CV_32FC1); //根据输入图片大小进行分配空间
		}
		int n = 0;
		for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
		{
			data_mat.at<float>(i, n) = *iter;
			n++;
		}
		res_mat.at<float>(i, 0) = img_label[i];

	}

	CvSVM svm;//新建一个SVM      
	CvSVMParams param;//这里是参数  
	CvTermCriteria criteria;
	criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria);
	/*
	SVM种类：CvSVM::C_SVC
	Kernel的种类：CvSVM::RBF
	degree：10.0（此次不使用）
	gamma：8.0
	coef0：1.0（此次不使用）
	C：10.0
	nu：0.5（此次不使用）
	p：0.1（此次不使用）
	然后对训练数据正规化处理，并放在CvMat型的数组里。
	*/
	//☆☆☆☆☆☆☆☆☆(5)SVM学习☆☆☆☆☆☆☆☆☆☆☆☆           
	svm.train(data_mat, res_mat, Mat(), Mat(), param);//训练啦      
													  //☆☆利用训练数据和确定的学习参数,进行SVM学习☆☆☆☆       
	svm.save(save_path);

}
void svm_test(const char* svm_data_path, const char* test_data_path)
{
	CvSVM svm;
	svm.load(svm_data_path);
	vector<string> img_test_path;
	ifstream img_path_input(test_data_path);
	if (img_path_input.fail())return;
	string line;
	while (getline(img_path_input,line))
	{
		img_test_path.push_back(line);
	}
	int nImgNum = img_test_path.size();
	
	for (string::size_type i = 0; i != nImgNum; i++)
	{
		Mat src = imread(img_test_path[i].c_str(), 1);
		Mat trainImg = Mat::zeros(64, 64, CV_8UC3);
		resize(src, trainImg, Size(64, 64), 0, 0, INTER_CUBIC);
		HOGDescriptor hog = HOGDescriptor(cvSize(64, 64), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);  //具体意思见参考文章1,2       
		vector<float>descriptors;//结果数组  
		hog.compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //调用计算函数开始计算
		Mat svm_mat = Mat::zeros(nImgNum, descriptors.size(), CV_32FC1);
		int n = 0;
		for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
		{
			svm_mat.at<float>(i, n) = *iter;
			n++;
		}
		int ret = svm.predict(svm_mat);
		printf("predict:%d | path:%s\n", ret, img_test_path[i].c_str());
	}
}
//bool svmTest()
//{
//	string buf;
//	CvSVM svm;
//	svm.load(DEFAULT_SVMMODEL_PATH);//加载训练好的xml文件
//									//检测样本    
//	IplImage *test;
//	char result[512];
//	vector<string> img_tst_path;
//	ifstream img_tst(DEFAULT_TESTSAMPLES_TXT_DECRIBE_PATH);  //加载需要预测的图片集合，这个文本里存放的是图片全路径，不要标签
//	if (!img_tst)
//		return FALSE;
//	while (img_tst)
//	{
//		if (getline(img_tst, buf))
//		{
//			img_tst_path.push_back(buf);
//		}
//	}
//	img_tst.close();
//
//	ofstream predict_txt(DEFAULT_TESTSAMPLES_RECOGNITION_RESULT_TXT_DECRIBE_PATH);//把预测结果存储在这个文本中   
//	for (string::size_type j = 0; j != img_tst_path.size(); j++)//依次遍历所有的待检测图片    
//	{
//		test = cvLoadImage(img_tst_path[j].c_str(), 1);
//		if (test == NULL)
//		{
//			cout << " can not load the image: " << img_tst_path[j].c_str() << endl;
//			continue;//结束本次循环
//		}
//		IplImage* trainTempImg = cvCreateImage(cvSize(40, 32), 8, 3);
//		cvZero(trainTempImg);
//		cvResize(test, trainTempImg);
//		HOGDescriptor *hog = new HOGDescriptor(cvSize(40, 32), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
//		vector<float>descriptors;//结果数组       
//		hog->compute(trainTempImg, descriptors, Size(1, 1), Size(0, 0));
//		cout << "HOG dims: " << descriptors.size() << endl;
//		CvMat* SVMtrainMat = cvCreateMat(1, descriptors.size(), CV_32FC1);
//		int n = 0;
//		for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
//		{
//			cvmSet(SVMtrainMat, 0, n, *iter);
//			n++;
//		}
//
//		int ret = svm.predict(SVMtrainMat);//检测结果
//		sprintf(result, "%s  %d\r\n", img_tst_path[j].c_str(), ret);
//		predict_txt << result;  //输出检测结果到文本
//	}
//	predict_txt.close();
//	cvReleaseImage(&test);
//	return TRUE;
//}