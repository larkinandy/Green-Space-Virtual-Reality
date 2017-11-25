#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include<chrono>
#include <algorithm>
#include <CL/cl.h>



class DeviceBaseClass {

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
	cl_device_id * deviceIDs = NULL;
	cl_uint preferredDevice;
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
	
	void copyDataToBuffer(cl_uint queueNumber, cl_uint bufferNumber, cl_int * hostData, cl_uint numElements);
	void copyDataToBuffer(cl_uint queueNumber, cl_uint bufferNumber, cl_float * hostData, cl_uint numElements);
	void copyDataToBuffer(cl_uint queueNumber, cl_uint bufferNumber, char * hostData, cl_uint numElements);
	void copyDataToHost(cl_uint queueNum, cl_uint bufferNum, cl_float * outputVals, cl_uint numElements);
	void copyDataToHost(cl_uint queueNum, cl_uint bufferNum, cl_int * outputVals, cl_uint numElements);
	void copyDataToHost(cl_uint queueNum, cl_uint bufferNum, char * outputVals, cl_uint numElements);
	void createBuffer(cl_uint elemSize, cl_uint numElements, cl_mem_flags typeFlag);
	void createCommandQueue(cl_uint deviceNum);
	void enqeueKernel(cl_uint kernelNum, cl_uint numThreads, cl_uint deviceNum);
	void enqeueKernel(cl_uint queueNum, cl_uint kernelNum, cl_uint numThreads, cl_uint deviceNum);
private:
};