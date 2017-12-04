#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <CL/cl.h>

#include "parsedStruct.h"


class DeviceBaseClass 
{

	public:
		DeviceBaseClass(cl_context * contextPtr, cl_device_id * deviceIDs, cl_uint numDevices,cl_uint perferredDevice);
		DeviceBaseClass();
		~DeviceBaseClass();

protected:

	char * deviceFunctionFile = "NULL fill with child file";
	const cl_bool debug = true;

	// setup opencl objects and variables;
	cl_int errNum;
	cl_uint numDevices;
	cl_uint *numWorkItems;
	cl_uint preferredDevice;
	cl_device_id * deviceIDs = NULL;
	
	cl_context context = NULL;
	cl_program program;
	std::vector<cl_kernel> kernels;
	std::vector<cl_command_queue> queues;
	std::vector<cl_mem> buffers;
	std::vector<cl_event> events;

	void releaseBuffers();
	void releaseProgram();
	void releaseCommandQueues();
	void releaseKernels();
	void releaseEvents();

	void checkErr(cl_int err, const char * name);
	void createProgram(cl_uint numDevices, cl_device_id * deviceIDs,cl_uint deviceNum);
	void createCommandQueue(cl_uint deviceNum);
	void enqeueKernel(cl_uint queueNum, cl_uint kernelNum, cl_uint numThreads, cl_uint deviceNum);
	void enqeueKernel(cl_uint queueNum, cl_uint kernelNum, cl_uint numThreads, cl_uint deviceNum, cl_event * priorEvent);
	void enqeueKernel(cl_uint queueNum, cl_uint kernelNum, cl_uint numThreads, cl_uint numLocal, cl_uint deviceNum, cl_event * priorEvent);

	void createBuffer(cl_uint elemSize, cl_uint numElements, cl_mem_flags typeFlag);
	void createBuffer(cl_uint elemSize, std::vector<cl_mem> * varBuffers, cl_uint numElements, cl_mem_flags typeFlag);
	
	void copyDataToBuffer(cl_uint queueNumber, cl_mem * buffer, char * hostData, cl_uint numElements);
	void copyDataToBuffer(cl_uint queueNumber, cl_uint bufferNumber, cl_int * hostData, cl_uint numElements);
	void copyDataToBuffer(cl_uint queueNumber, cl_uint bufferNumber, cl_float * hostData, cl_uint numElements);
	void copyDataToBuffer(cl_uint queueNumber, cl_uint bufferNumber, char * hostData, cl_uint numElements);
	void copyDataToBuffer(cl_uint queueNumber, cl_mem * buffer, char * hostData, cl_uint numElements, cl_event * event);

	void copyDataToHost(cl_uint queueNum, cl_mem bufferNum, cl_int * outputVals, cl_uint numElements);
	void copyDataToHost(cl_uint queueNum, cl_mem bufferNum, cl_char * outputVals, cl_uint numElements);
	
private:
};