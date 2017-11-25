#pragma once




#include "DeviceBaseClass.hpp"


class Average : public DeviceBaseClass

{

public:
	void getAverage(cl_uint numElements, cl_int * inputData, cl_float ** outputData);
	Average(cl_context * contextPtr, cl_device_id * deviceIDs, cl_uint numDevices, cl_uint preferredDevice);
	Average();
	~Average();


protected:

private:
	
	cl_int numObs;
	const cl_bool debug_block1 = true;
	void callKernels();
	void createBuffers(cl_uint numBuffers, cl_uint numElementsPerBuffer);
	void ceateCommandQueues(cl_uint numBuffers);

};