#pragma once


#include "ContextManager.hpp"
#include "Average.h"
#include "ClParser.hpp"
#include <CL/cl.h>

class SMA_Analyzer
{
public:
	SMA_Analyzer(char *inputFilepath, int numObs);
	SMA_Analyzer();
	~SMA_Analyzer();
	void getAverage(int numElements, int * inputData, float ** outputData);
	void setFilepath(char * filepath);
	void setNumobs(int numObs);
	char * getFilepath();
	int getNumObs();
	int getSelectedPlatform();
	int getSelectedDevice();
	void printDeviceInfo();
	cl_int parseCSV(char *inputFilepath);
protected:


private:
	char * inputFilepath;
	int numObs;
	const bool debug = true;
	const bool debug_block1 = true;

	Context_Manager * contextManager;
	Average * averager;
	ClParser * parser;

	cl_uint selectedPlatform = NULL;
	cl_uint selectedDevice = NULL;



	// setup opencl objects and variables;
	cl_int errNum;
	cl_uint numPlatforms = 0;
	cl_uint numDevices = 0;
	cl_platform_id * platformIDs = NULL;
	cl_device_id * deviceIDs = NULL;
	cl_context context = NULL;
	

	void checkErr(cl_int err, const char * name);

};