#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include<chrono>

#include <CL/cl.h>


class DeviceBaseClass {

public:
	DeviceBaseClass(cl_context * contextPtr, cl_device_id * deviceIDs, cl_uint numDevices);
	DeviceBaseClass();
	~DeviceBaseClass();

protected:

	char * deviceFunctionFile;
	const char * KERNEL_FILE = "Larkin_module12.cl";
	const bool debug = true;

	cl_uint selectedPlatform = -1;
	cl_uint selectedDevice = -1;
	cl_uint numDevices;

	// setup opencl objects and variables;
	cl_int errNum;
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


	void createProgram(cl_int numDevices, cl_device_id * deviceIDs);
	void createBuffers(int numBuffers, int numElementsPerBuffer);
	void ceateCommandQueues(int numBuffers, cl_device_id * deviceIDs, std::vector<cl_command_queue> * queues,
		std::vector<cl_mem> buffers, std::vector<cl_kernel> *kernels);
	void copyDataToBuffer(std::vector<cl_command_queue> * queues, std::vector<cl_mem> * buffers, int * inputData, int numElements);
	void callKernels(std::vector<cl_command_queue> * queues, std::vector<cl_kernel> * kernels, std::vector<cl_event> * events, int numBufferElements);
	void copyDataToHost(std::vector<cl_command_queue> * queues, std::vector<cl_mem> * buffers, cl_uint numBuffers, float * outputVals);


private:
};