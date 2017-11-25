#pragma once


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include<chrono>
#include <algorithm>

#include <CL/cl.h>

using namespace std;


class Context_Manager 
{
	public:
		Context_Manager();
		~Context_Manager();
		void getOptimalDevices(cl_device_id ** inDeviceIDs, cl_uint * optimalDevices, cl_uint  * numDevices);
		void getOptimalPlatform(cl_platform_id * optimalPlatform);
		void getOptimalContext(cl_context * optimalContext);
		void printDeviceInfo();
		void printDeviceInfo(cl_uint deviceNum);
	protected:

	private:
		cl_platform_id * platformIDs = NULL;
		cl_device_id * deviceIDs = NULL;
		cl_context context = NULL;
		cl_uint numDevices;
		cl_int errNum;
		cl_uint selectedDevice;
		cl_uint selectedPlatform;
		cl_uint  numPlatforms = 0;

		const cl_bool debug = true;

		void checkErr(cl_int err, const char * name);
		void getPlatformInfo(cl_platform_id ** platformIds);
		void getBestDeviceOnPlatform(cl_platform_id * platformIDs, cl_uint platformNum,
			cl_uint * maxCompute, cl_uint *selectedPlatform, cl_uint * selectedDevice);
		int selectOptimalDevice(cl_platform_id * platformIDs, cl_device_id ** deviceIDs,
			cl_uint * selectedPlatform, cl_uint * selectedDevice, cl_uint * numDevices, cl_uint numPlatforms);
		void setupContext(cl_platform_id * platformIDs, cl_context * context, cl_device_id * deviceIDs, cl_uint platform, cl_uint numDevices);
	
		void releaseContext(cl_context context);
		void releaseDevices(cl_device_id * deviceIDs, cl_uint numDevices);
		void releaseDevices();
};