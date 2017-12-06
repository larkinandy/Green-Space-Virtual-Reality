/* DeviceBaseClass.hpp
* Header file for base class with boiler plate for OpenCL functions 
* Author: Andrew Larkin
* December 5, 2017 */

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

// base class containing abstracted functions for many common OpenCL functions
// never implemented as a standalone class
class DeviceBaseClass
{
	public:
		DeviceBaseClass(cl_context * contextPtr, cl_device_id * deviceIDs, cl_uint numDevices,cl_uint perferredDevice);
		DeviceBaseClass();
		~DeviceBaseClass();

protected:

	char * deviceFunctionFile = "NULL fill with OpenCL kernel filename";
	cl_bool debug = true;

	// setup opencl objects and variables;
	cl_int errNum;
	cl_uint numDevices;
	cl_uint *numWorkItems;
	cl_uint preferredDevice;
	cl_device_id * deviceIDs = NULL;
	cl_context context = NULL;
	cl_program program;

	// vectors for storing common OpenCL objects
	std::vector<cl_kernel> kernels;
	std::vector<cl_command_queue> queues;
	std::vector<cl_mem> buffers;
	std::vector<cl_event> events;

	// kernel and queue operations
	void checkErr(cl_int err, const char * name);
	void createProgram(cl_uint numDevices, cl_device_id * deviceIDs,cl_uint deviceNum);
	void createCommandQueue(cl_uint deviceNum);
	void enqeueKernel(cl_uint queueNum, cl_uint kernelNum, cl_uint numThreads, cl_uint deviceNum);
	void enqeueKernel(cl_uint queueNum, cl_uint kernelNum, cl_uint numThreads, cl_uint deviceNum, cl_event * priorEvent);
	void enqeueKernel(cl_uint queueNum, cl_uint kernelNum, cl_uint numThreads, cl_uint numLocal, cl_uint deviceNum, cl_event * priorEvent);

	// buffer operations
	void createBuffer(cl_uint elemSize, cl_uint numElements, cl_mem_flags typeFlag);
	void createBuffer(cl_uint elemSize, std::vector<cl_mem> * varBuffers, cl_uint numElements, cl_mem_flags typeFlag);
	
	// operations to copy data from host to device.  Multiple variation for different buffer types (e.g. int, char, etc)
	void copyDataToBuffer(cl_uint queueNumber, cl_mem * buffer, char * hostData, cl_uint numElements);
	void copyDataToBuffer(cl_uint queueNumber, cl_uint bufferNumber, cl_int * hostData, cl_uint numElements);
	void copyDataToBuffer(cl_uint queueNumber, cl_uint bufferNumber, cl_float * hostData, cl_uint numElements);
	void copyDataToBuffer(cl_uint queueNumber, cl_uint bufferNumber, char * hostData, cl_uint numElements);
	void copyDataToBuffer(cl_uint queueNumber, cl_mem * buffer, char * hostData, cl_uint numElements, cl_event * event);

	// copy data from device to host
	void copyDataToHost(cl_uint queueNum, cl_mem bufferNum, cl_int * outputVals, cl_uint numElements);
	void copyDataToHost(cl_uint queueNum, cl_mem bufferNum, cl_char * outputVals, cl_uint numElements);

	// cleanup
	void releaseBuffers();
	void releaseBuffers(std::vector<cl_mem> * buffersToRelease);
	void releaseProgram();
	void releaseCommandQueues();
	void releaseKernels();
	void releaseEvents();


private:
};

// end of DeviceBaseClass.hpp