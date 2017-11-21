#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include<chrono>

#include "ContextManager.hpp"
#include "Average.h"
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
protected:

private:
	char * inputFilepath;
	int numObs;
	const char * KERNEL_FILE = "Larkin_module12.cl";
	const bool debug = true;
	const bool debug_block1 = true;

	Context_Manager contextManager = Context_Manager(debug_block1);
	Average * averager;

	cl_uint selectedPlatform = -1;
	cl_uint selectedDevice = -1;



	// setup opencl objects and variables;
	cl_int errNum;
	cl_uint numPlatforms = 0;
	cl_uint numDevices = 1;
	cl_platform_id * platformIDs = NULL;
	cl_device_id * deviceIDs = NULL;
	cl_context context = NULL;
	cl_program program;
	std::vector<cl_kernel> kernels;
	std::vector<cl_command_queue> queues;
	std::vector<cl_mem> buffers;
	std::vector<cl_event> events;

	void checkErr(cl_int err, const char * name);

};