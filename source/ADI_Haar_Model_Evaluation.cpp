// ADI_Haar_Model_Evaluation.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "adi_image_tool_box.h"
#include "adi_haarfeatures.h"
#include "profile.h"

#include <opencv.hpp>  
#include <opencv2/core/core.hpp>  
#include <iostream> 
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <functional>

using namespace std;

/*=============  D A T A  =============*/
ALIGN(4)
	MEMORY_SECTION(adi_slow_noprio_disp_rw)
	int8_t  aInputImage[ADI_IMAGE_WIDTH * ADI_IMAGE_HEIGHT * 3];

ALIGN(4)
	MEMORY_SECTION(adi_slowb1_prio0_rw)
	uint16_t    aImageSquare[ADI_IMAGE_WIDTH * ADI_IMAGE_HEIGHT];

ALIGN(4)
	MEMORY_SECTION(vidout_section0)
	uint8_t aGrayImage[ADI_IMAGE_WIDTH * ADI_IMAGE_HEIGHT];

ALIGN(4)
	MEMORY_SECTION(vidout_section0)
	uint32_t    aIntegralImage[ADI_ROI_IMAGE_WIDTH * ADI_ROI_IMAGE_HEIGHT];

ALIGN(4)
	MEMORY_SECTION(vidout_section0)
	uint64_t    aIntegralImageSquare[ADI_ROI_IMAGE_WIDTH * ADI_ROI_IMAGE_HEIGHT];

ALIGN(4)
	MEMORY_SECTION(vidout_section0)
	uint32_t    aImageTemp[ADI_IMAGE_WIDTH * ADI_IMAGE_HEIGHT];

ALIGN(4)
	MEMORY_SECTION(adi_slowb1_prio0_rw)
	uint16_t    aDetectedFaces[ADI_MAX_FACES * 4];

ALIGN(4)
	MEMORY_SECTION(adi_slowb1_prio0_rw)
	char_t  aTrainedDataMemory[ADI_MEMORYFOR_TRAINEDDATA];

ALIGN(4)
	MEMORY_SECTION(adi_slowb1_prio0_rw)
	int32_t aTrainedFileMemory[ADI_TRAINED_FILE_SIZE];

ALIGN(4)
	MEMORY_SECTION(adi_slow_noprio_disp_rw)
	char_t          aString[PPM_HEADER_SIZE];

ALIGN(4)
	MEMORY_SECTION(adi_appl_slow_noprio_rw)
	static char_t   aOutFilename[256];

ALIGN(4)
	MEMORY_SECTION(adi_appl_slow_noprio_rw)
	static char_t   atempFilename[256];

ALIGN(4)
	MEMORY_SECTION(adi_appl_slow_noprio_rw)
	static char_t   aPathToMedia[256];
/******************************************************************************

  Function              : main

  Function description  : Setup path for media files,
                          Check Input file type,
                          Open Input file and read data into L3 buffer,
                          Call required module,
                          Open Output file and write processed data


  Parameters            : None


  Returns               : Status (Zero/Non-Zero)

  Notes                 :

******************************************************************************/

extern int32_t haar_features_params[];
int parase_csv_file(const char* file_name, vector<cv::Point>& points);
float calcute_overlap_ratio(cv::Rect& rect_one, cv::Rect& rect_two);

int _tmain(int argc, _TCHAR* argv[])
{
	//FILE                        *pInputFile;
	//FILE                        *pOutFile;
	//char_t                      aType[32];
	//char_t                      aExtension[8];

	int32_t                     nResult;
	uint32_t                    nFacesDetecetd;
	ADI_IMAGE_HAAR_DATA         oImageData;
	int32_t                     i, j;
	float32_t                   nScaleIncrement;
	ADI_IMAGE_SIZE              oMinObjectSize;

	//uint32_t                    nBytesRead;
	//uint32_t                    nDisplayColor;
	//ADI_POINT                   nCenter;
	//int32_t                     nRadius;

	uint8_t                     *pImagePtr;
	uint32_t                    nXOffset;
	uint32_t                    nYOffset;
	uint32_t                    nStride;
	uint32_t                    nNumRawFaces;
	uint32_t                    nMinNeighbours;
	uint32_t                    nOvelapPercentage;
	
	ADI_HAARCLASSIFIERCASCADE   *pClassifierCascade;


	char input_image_name[100];
	char input_csv_name[100];

	memcpy(aTrainedFileMemory,haar_features_params,ADI_TRAINED_FILE_SIZE * sizeof(int32_t));

	nResult = adi_HaarFeaturesInit((ADI_HAARCLASSIFIERCASCADE *)aTrainedDataMemory,
		ADI_MEMORYFOR_TRAINEDDATA,
		(int8_t *)aTrainedFileMemory,
		ADI_TRAINED_FILE_SIZE);

	if (nResult < 0)
	{
		printf("Memory Given For Storing Trained data was not sufficient. %d bytes short", -(nResult));
		return -1;
	}

	int right_detect_num = 0;
	int wrong_detect_num = 0;

	for (i = 1; i < NUM_INPUT_FILES; i++)
	{
		// Parase input csv
		sprintf(input_image_name,"../%s\\%d.csv","csv", i); 
		vector<cv::Point> key_points;
		int flag = parase_csv_file(input_csv_name, key_points);
		cv::Rect annotation_rect(0, 0, 0, 0);
		if (key_points[0].x >= 0 && key_points[0].y >= 0 && key_points[1].x >= 0 && key_points[1].y >= 0)
		{
			int x_mid = ((key_points[0].x - 5) + (key_points[1].x + 5)) / 2;
			int y_mid = (key_points[0].y + key_points[1].y) / 2;
			int width = ((key_points[1].x + 5) - (key_points[0].x - 5));
			int height = width / 2;

			annotation_rect.x = key_points[0].x - 5;
			annotation_rect.y = y_mid - height / 2;
			annotation_rect.width = width;
			annotation_rect.height = height;
		}

		// Load input image
		sprintf(input_image_name,"../%s\\%d.jpg","image", i); 
		cv::Mat input_image = cv::imread(input_image_name);

		if (input_image.empty())
		{
			break;
		}
		memcpy(aInputImage, input_image.data, ADI_IMAGE_WIDTH * ADI_IMAGE_HEIGHT * 3 * sizeof(uint8_t));

		printf("Image Num = %d, process => Haar Feature Face Detection\n", i);

		nScaleIncrement = 1.4567;
		nXOffset = 0;
		nYOffset = 0;
		nMinNeighbours = 0;
		nOvelapPercentage = 75;

		adi_RGB2GRAY((uint8_t *)aInputImage,
			aGrayImage,
			ADI_IMAGE_WIDTH ,
			ADI_IMAGE_HEIGHT);

		pImagePtr = aGrayImage;
		pImagePtr = pImagePtr + (nYOffset * ADI_IMAGE_WIDTH) + nXOffset;
		nStride = ADI_IMAGE_WIDTH - ADI_ROI_IMAGE_WIDTH;

		adi_HaarPreProcess(&oImageData,
			pImagePtr,
			aImageSquare,
			aIntegralImage,
			aIntegralImageSquare,
			aImageTemp,
			ADI_ROI_IMAGE_WIDTH,
			ADI_ROI_IMAGE_HEIGHT,
			nStride);

		pClassifierCascade = (ADI_HAARCLASSIFIERCASCADE *)aTrainedDataMemory;

		/* Optionally You can add adi_EqualizeHist() Here to improve the results */
		oMinObjectSize.nWidth = ADI_MIN_OBJECTWIDTH;
		oMinObjectSize.nHeight = ADI_MIN_OBJECTHEIGHT;

		nNumRawFaces = adi_HaarDetectObjects(&oImageData,
			nScaleIncrement,
			pClassifierCascade,
			&oMinObjectSize,
			aDetectedFaces,
			(uint8_t *)aImageTemp);

		nFacesDetecetd = adi_HaarPostProcess(aDetectedFaces,
			(uint16_t *)aImageTemp,
			nNumRawFaces,
			nOvelapPercentage,
			nMinNeighbours);

		std::vector<float> ratios;
		//Analyze the results
		for (j = 0; j < nFacesDetecetd * 4; j += 4)
		{
			cv::Rect adi_result_rect;
			adi_result_rect.x = (int)(aDetectedFaces[j]);
			adi_result_rect.y = (int)(aDetectedFaces[j + 1]);
			adi_result_rect.width = (int)(aDetectedFaces[j + 2]);
			adi_result_rect.height = (int)(aDetectedFaces[j + 3]);

			if (annotation_rect.x > 0 && annotation_rect.y > 0 
				&& (annotation_rect.x + annotation_rect.width) < input_image.cols
				&& (annotation_rect.y + annotation_rect.height) < input_image.rows)
			{
				float ratio_value = calcute_overlap_ratio(adi_result_rect, annotation_rect);
				if (ratio_value >= 0.8)
				{
					ratios.push_back(ratio_value);
				}
				else
				{
					wrong_detect_num++;
				}
				
			}
			else
			{
				wrong_detect_num++;
			}
		}

		if (ratios.size() >= 1)
		{
			right_detect_num++;

			//order the values from largest to smallest
			std::sort(ratios.begin(), ratios.end(), std::greater<float>());
		}
		
	}
	
	printf("*******************************************************************\n");
	printf("Done\n");
	return 0;
}

int parase_csv_file(const char* file_name, vector<cv::Point>& points)
{
	// Load a csv file
	ifstream inFile(file_name, ios::in);
	if (inFile.fail())
	{
		points.push_back(cv::Point(-1, -1));
		points.push_back(cv::Point(-1, -1));
		return -1;
	}

	string lineStr;
	vector<vector<string>> strArray;
	while (getline(inFile, lineStr))
	{
		// Print the whole line string
		cout << lineStr << endl;

		// Save as dimensional table structure
		stringstream ss(lineStr);
		string str;
		vector<string> lineArray;

		// Separated the value by a comma
		while (getline(ss, str, ','))
		{
			lineArray.push_back(str);
		}
		strArray.push_back(lineArray);
	}

	int flag = stoi(strArray[0][0]);

	switch (flag)
	{
	case 0:
		points.push_back(cv::Point(stoi(strArray[37][0]), stoi(strArray[37][1])));
		points.push_back(cv::Point(stoi(strArray[46][0]), stoi(strArray[46][1])));
		break;

	case 1:
		points.push_back(cv::Point(stoi(strArray[69][0]), stoi(strArray[69][1])));
		points.push_back(cv::Point(stoi(strArray[70][0]), stoi(strArray[70][1])));
		break;

	case 2:
		points.push_back(cv::Point(stoi(strArray[1][0]), stoi(strArray[1][1])));
		points.push_back(cv::Point(stoi(strArray[2][0]), stoi(strArray[2][1])));
		break;

	default:
		points.push_back(cv::Point(-1, -1));
		points.push_back(cv::Point(-1, -1));
		break;
	}

	return flag;
}

float calcute_overlap_ratio(cv::Rect& rect_one, cv::Rect& rect_two)
{
	int left = max(rect_one.x, rect_two.x);
	int top = max(rect_one.y, rect_two.y);
	int right = min(rect_one.x + rect_one.width, rect_two.x + rect_two.width);
	int bottom = min(rect_one.y + rect_one.height, rect_two.y + rect_two.height);
	int max_area = max(rect_one.width * rect_one.height, rect_two.width * rect_two.height);

	float ratio = (float)(right - left) * (bottom - top) / max_area;

	return ratio;
}