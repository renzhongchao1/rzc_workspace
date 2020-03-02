#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include <cv.h>
// #include <highgui.h>
// #include <cvaux.h>
#include <iostream>
#include <vector>
#include<string.h>
using namespace std;
using namespace cv;

int main ( int argc, char** argv )
{
    cout << "OpenCV Training SVM Automatic Number Plate Recognition\n";
    cout << "\n";

    char* path_Plates;
    char* path_NoPlates;
    int numPlates;
    int numNoPlates;
    int imageWidth=150;
    int imageHeight=150;

    //Check if user specify image to process
    if(1)
    {
        numPlates= 12;
        numNoPlates= 90 ;
        path_Plates= "/home/kaushik/opencv_work/Manas6/Pics/Positive_Images/";
        path_NoPlates= "/home/kaushik/opencv_work/Manas6/Pics/Negative_Images/i";

    }else{
        cout << "Usage:\n" << argv[0] << " <num Plate Files> <num Non Plate Files> <path to plate folder files> <path to non plate files> \n";
        return 0;
    }

    Mat classes;//(numPlates+numNoPlates, 1, CV_32FC1);
    Mat trainingData;//(numPlates+numNoPlates, imageWidth*imageHeight, CV_32FC1 );

    Mat trainingImages;
    vector<int> trainingLabels;

    for(int i=1; i<= numPlates; i++)
    {

        stringstream ss(stringstream::in | stringstream::out);
        ss<<path_Plates<<i<<".jpg";
        try{

            const char* a = ss.str().c_str();
            printf("\n%s\n",a);
            Mat img = imread(ss.str(), CV_LOAD_IMAGE_UNCHANGED);
            img= img.clone().reshape(1, 1);
            //imshow("Window",img);
            //cout<<ss.str();
            trainingImages.push_back(img);
            trainingLabels.push_back(1);
        }
        catch(Exception e){;}
    }

    for(int i=0; i< numNoPlates; i++)
    {
        stringstream ss(stringstream::in | stringstream::out);
        ss << path_NoPlates<<i << ".jpg";
        try
        {
            const char* a = ss.str().c_str();
            printf("\n%s\n",a);
            Mat img=imread(ss.str(),CV_LOAD_IMAGE_UNCHANGED);
            //imshow("Win",img);
            img= img.clone().reshape(1, 1);
            trainingImages.push_back(img);
            trainingLabels.push_back(0);
            //cout<<ss.str();
        }
        catch(Exception e){;}
    }

    Mat(trainingImages).copyTo(trainingData);
    //trainingData = trainingData.reshape(1,trainingData.rows);
    trainingData.convertTo(trainingData, CV_32FC1);
    Mat(trainingLabels).copyTo(classes);

    FileStorage fs("SVM.xml", FileStorage::WRITE);
    fs << "TrainingData" << trainingData;
    fs << "classes" << classes;
    fs.release();

    return 0;
}