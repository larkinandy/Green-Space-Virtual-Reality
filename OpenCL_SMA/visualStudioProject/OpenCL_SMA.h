#pragma once



#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include<chrono>

#include "info.hpp"
#include <CL/cl.h>


class SMA_Analyzer
{
public:
	SMA_Analyzer(char *inputFilepath, int numObs);
	SMA_Analyzer();
	void getAverage(int numElements, int * inputData);
	void createInput(int ** inputOutput, int numBufferElements);
	void setFilepath(char * filepath);
	void setNumobs(int numObs);
	char * getFilepath();
	int getNumObs();
protected:

private:
	cl_int errNum;
	char * inputFilepath;
	int numObs;
	const char * KERNEL_FILE = "Larkin_module12.cl";
	const bool debug = true;



	void checkErr(cl_int err, const char * name);
	void setupPlatform(cl_platform_id ** platformIDs, cl_uint numPlatforms, int platform);
	void setupDevices(cl_platform_id * platformIDs, cl_device_id ** deviceIDs, int platform, cl_uint * numDevices);
	void setupContext(cl_platform_id * platformIDs, cl_context * context, cl_device_id * deviceIDs, int platform, cl_uint numDevices);
	void createProgram(cl_program * program, cl_context context, cl_int numDevices, cl_device_id * deviceIDs);
	void printOutput(int numDevices, float * results, int numElements);
	void createBuffers(int ** inputOutput, int numBuffers, cl_context context, std::vector<cl_mem> * buffers, int numElementsPerBuffer);
	
	void ceateCommandQueues(int numBuffers, cl_device_id * deviceIDs, cl_context context, std::vector<cl_command_queue> * queues, std::vector<cl_mem> buffers,
		cl_program program, std::vector<cl_kernel> *kernels);
	void copyDataToBuffer(std::vector<cl_command_queue> * queues, std::vector<cl_mem> * buffers, int * inputOutput, int numElements);
	void callKernels(std::vector<cl_command_queue> * queues, std::vector<cl_kernel> * kernels, std::vector<cl_event> * events, int numBufferElements);
	void copyDataToHost(std::vector<cl_command_queue> * queues, std::vector<cl_mem> * buffers, cl_uint numBuffers, float * outputVals);
	

};