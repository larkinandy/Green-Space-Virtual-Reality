/* ContextManager.cpp
* Implementation file for class to identify and query OpenCL platforms and select best device
* Author: Andrew Larkin
* December 5, 2017 */

#include "ContextManager.hpp"


Context_Manager::Context_Manager() 
{
	getPlatformInfo(&platformIDs);
	errNum = selectOptimalDevice(platformIDs, &deviceIDs,&selectedPlatform,
		&selectedDevice, &numDevices, numPlatforms);
	checkErr(errNum, "selectOptimalDevice");
	setupContext(platformIDs, &context, &(deviceIDs[selectedDevice]), selectedPlatform, 1);
}

Context_Manager::~Context_Manager() 
{
	free(deviceIDs);
	releaseContext(context);
	free(platformIDs);
	if (debug) { cout << "destroying context manager" << endl; }
}

void Context_Manager::releaseDevices()
{
	// only needed if using subdevices.  Reserve place for potential implementation later
}

//return device id for device previously determined to be optimal by the ContextManager
void Context_Manager::getOptimalDevices(cl_device_id ** indeviceIDs, cl_uint * optimalDevices, cl_uint * numDevices)
{
	*indeviceIDs = deviceIDs;
	*optimalDevices = this->selectedDevice;
	*numDevices = this->numDevices;
}

// return platform id for platform previously determined to be optimal
void Context_Manager::getOptimalPlatform(cl_platform_id * platformID)
{
	platformID = &(this->platformIDs[selectedPlatform]);
}

// return context created by manager 
void Context_Manager::getOptimalContext(cl_context * context)
{
	*context = this->context;
}

// Function to check and handle OpenCL errors
void Context_Manager::checkErr(cl_int err, const char * name)
{
	if (err != CL_SUCCESS) 
	{
		std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}

// identify platforms and choose first platform on list
void Context_Manager::getPlatformInfo(cl_platform_id ** platformIDs)
{
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	std::cout << "num platforms: " << numPlatforms << std::endl;
	checkErr(
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
		"clGetPlatformIDs");

	*platformIDs = (cl_platform_id *)malloc(sizeof(cl_platform_id)* numPlatforms);

	errNum = clGetPlatformIDs(numPlatforms, *platformIDs, NULL);
	checkErr(
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
		"clGetPlatformIDs");
}

// evaluate GPU devices on a platform and determine which GPU device has the best work group size
void Context_Manager::getBestDeviceOnPlatform(cl_platform_id * platformIDs, cl_uint platformNum,
	cl_uint * maxCompute, cl_uint *selectedPlatform, cl_uint * selectedDevice)
{
	cl_uint numTempDevices = 0;
	errNum = clGetDeviceIDs(platformIDs[platformNum], CL_DEVICE_TYPE_GPU, 0, NULL, &numTempDevices);
	if (errNum != 0) { std::cout << "no GPUs on platform" << platformNum; return; }
	if (numTempDevices > 0)
	{
		cl_device_id *tempDeivceIDs = (cl_device_id *)malloc(sizeof(cl_device_id) * numTempDevices);
		errNum = clGetDeviceIDs(platformIDs[platformNum],CL_DEVICE_TYPE_GPU, numTempDevices,tempDeivceIDs,NULL);
		if (errNum == -1) { free(tempDeivceIDs); return; }
		checkErr(errNum, "clGetDeviceIDs");

		cl_uint maxComputeNewDevice = 0;
		size_t size;
		for (cl_uint deviceIndex = 0; deviceIndex < numTempDevices; deviceIndex++)
		{
			errNum = clGetDeviceInfo(
				tempDeivceIDs[deviceIndex],
				CL_DEVICE_MAX_WORK_GROUP_SIZE,
				sizeof(cl_uint),
				&maxComputeNewDevice,
				&size);

			if (maxComputeNewDevice > *maxCompute)
			{
				*selectedPlatform = platformNum;
				*selectedDevice = deviceIndex;
				*maxCompute = maxComputeNewDevice;
			}
		}
		free(tempDeivceIDs);
	}
}

// evaluate GPU devices on all platforms and selecct platform and GPU with best work group size
cl_int Context_Manager::selectOptimalDevice(cl_platform_id * platformIDs, cl_device_id ** deviceIDs,
	cl_uint * selectedPlatform, cl_uint * selectedDevice, cl_uint * numDevices, cl_uint numPlatforms)
{
	cl_uint maxCompute = 0;
	for (cl_uint platformNum = 0; platformNum < numPlatforms; platformNum++)
	{
		getBestDeviceOnPlatform(platformIDs, platformNum, &maxCompute, selectedPlatform, selectedDevice);
	}

	if (maxCompute == 0)
	{
		std::cout << " Error, no OpenCL compatible GPU device found " << std::endl;
		return -1;
	}

	if (debug)
	{
		std::cout << "best selected device " << *selectedDevice << " found on platform " << *selectedPlatform << std::endl;
	}

	*numDevices = *selectedDevice + 1;
	*deviceIDs = (cl_device_id *)malloc(sizeof(cl_device_id)* *numDevices);

	errNum = clGetDeviceIDs(platformIDs[*selectedPlatform],CL_DEVICE_TYPE_GPU,*numDevices,*deviceIDs,NULL);
	checkErr(errNum, "clGetDeviceIDs");

	printDeviceInfo();
	return 0;
}

// print device info for all devices in the associated context
void Context_Manager::printDeviceInfo()
{
	for (cl_uint deviceIndex = 0; deviceIndex < numDevices; deviceIndex++)
	{
		printDeviceInfo(deviceIndex);
	}
}

// print select parameters for a GPU device
void Context_Manager::printDeviceInfo(cl_uint deviceNum)
{
	char buffer[1024];
	size_t max_work_size;
	size_t size;
	cl_ulong global_mem_size;
	{
		std::cout << "properties for device number " << deviceNum << std::endl;
		clGetDeviceInfo(deviceIDs[deviceNum], CL_DEVICE_NAME, sizeof(buffer), buffer, &size);
		std::cout << "Selected Device: " << buffer << std::endl;
		clGetDeviceInfo(deviceIDs[deviceNum], CL_DEVICE_VENDOR, sizeof(buffer), buffer, &size);
		std::cout << "Device Vendor: " << buffer << std::endl;
		clGetDeviceInfo(deviceIDs[deviceNum], CL_DEVICE_VERSION, sizeof(buffer), buffer, &size);
		std::cout << "Device Version: " << buffer << std::endl;
		clGetDeviceInfo(deviceIDs[deviceNum], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_size, &size);
		std::cout << "Max work group units: " << max_work_size << std::endl;
		clGetDeviceInfo(deviceIDs[deviceNum], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, &size);
		std::cout << "Global memory size (mb) " << global_mem_size / (1024 * 1024) << std::endl << std::endl;
	}
}

// create context and attach selected devices
void Context_Manager::setupContext(cl_platform_id * platformIDs, cl_context * context, cl_device_id * deviceIDs, cl_uint platform, cl_uint numDevices)
{
	std::cout << "platform number used to generate context: " << platform << std::endl;

	cl_int errNum;
	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platformIDs[platform],
		0
	};

	*context = clCreateContext(contextProperties,numDevices,deviceIDs,NULL,NULL,&errNum);
	checkErr(errNum, "clCreateContext");
}

void Context_Manager::releaseContext(cl_context context) 
{
	cl_uint numReleaes;
	clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT, sizeof(cl_uint), &numReleaes, NULL);
	for (cl_int releaseCount = numReleaes; releaseCount > 0; releaseCount--) 
	{
		errNum = clReleaseContext(context);
		checkErr(errNum, "release context");
	}
}

void Context_Manager::releaseDevices(cl_device_id * deviceIDs, cl_uint numDevices) 
{
	cl_uint numReleaes;
	for (cl_uint deviceIndex = 0; deviceIndex < numDevices; deviceIndex++)
	{
		clGetDeviceInfo(deviceIDs[deviceIndex], CL_DEVICE_REFERENCE_COUNT, sizeof(cl_uint), &numReleaes, NULL);
		for (cl_int releaseCount = numReleaes; releaseCount >= 0; releaseCount--)
		{
			errNum = clReleaseDevice(deviceIDs[deviceIndex]);
			checkErr(errNum, "release devices");
			std::cout << " releasing device" << deviceIndex << std::endl;
		}
	}
	free(deviceIDs);
}

// end of ContextManager.cpp