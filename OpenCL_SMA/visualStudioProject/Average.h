#pragma once



#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include<chrono>
#include "DeviceBaseClass.hpp"

#include <CL/cl.h>


class Average : public DeviceBaseClass

{

public:
	void getAverage(int numElements, int * inputData, float ** outputData);
	Average(cl_context * contextPtr, cl_device_id * deviceIDs, cl_uint numDevices);
	Average();
	~Average();


protected:

private:
	
	int numObs;
	const bool debug_block1 = true;



};