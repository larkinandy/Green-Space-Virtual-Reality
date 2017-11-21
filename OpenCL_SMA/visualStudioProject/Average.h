#pragma once



#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include<chrono>

#include <CL/cl.h>



class Average 

{

public:
	void getAverage(int numElements, int * inputData, float ** outputData);
	Average(cl_context * contextPtr, cl_device_id * deviceIDs);
	Average();
	~Average();


protected:

private:

	char * inputFilepath;
	int numObs;
	const char * KERNEL_FILE = "Larkin_module12.cl";
	const bool debug = true;
	const bool debug_block1 = true;

	cl_uint selectedPlatform = -1;
	cl_uint selectedDevice = -1;



	// setup opencl objects and variables;
	cl_int errNum;
	cl_device_id * deviceIDs = NULL;
	cl_context context = NULL;
	cl_program program;
	std::vector<cl_kernel> kernels;
	std::vector<cl_command_queue> queues;
	std::vector<cl_mem> buffers;
	std::vector<cl_event> events;

	void checkErr(cl_int err, const char * name);
	void createProgram(cl_program * program, cl_context context, cl_int numDevices, cl_device_id * deviceIDs);
	void createBuffers(int ** inputOutput, int numBuffers, cl_context context, std::vector<cl_mem> * buffers, int numElementsPerBuffer);

	void ceateCommandQueues(int numBuffers, cl_device_id * deviceIDs, cl_context context, std::vector<cl_command_queue> * queues, std::vector<cl_mem> buffers,
		cl_program program, std::vector<cl_kernel> *kernels);
	void copyDataToBuffer(std::vector<cl_command_queue> * queues, std::vector<cl_mem> * buffers, int * inputOutput, int numElements);
	void callKernels(std::vector<cl_command_queue> * queues, std::vector<cl_kernel> * kernels, std::vector<cl_event> * events, int numBufferElements);
	void copyDataToHost(std::vector<cl_command_queue> * queues, std::vector<cl_mem> * buffers, cl_uint numBuffers, float * outputVals);



};