
#include "ContextManager.hpp"

Context_Manager::Context_Manager(bool debug) 
{
	this->debug = debug;
	getPlatformInfo(&platformIDs, &numPlatforms, selectedPlatform, selectedDevice);
	selectOptimalDevice(platformIDs, &deviceIDs,
		&selectedPlatform, &selectedDevice, &numDevices, numPlatforms);
	setupContext(platformIDs, &context, &(deviceIDs[selectedDevice]), selectedPlatform, 1);
}

Context_Manager::Context_Manager()
{
	this->debug = false;
	getPlatformInfo(&platformIDs, &numPlatforms, selectedPlatform, selectedDevice);
	selectOptimalDevice(platformIDs, &deviceIDs,
		&selectedPlatform, &selectedDevice, &numDevices, numPlatforms);
	setupContext(platformIDs, &context, &(deviceIDs[selectedDevice]), selectedPlatform, 1);
}

Context_Manager::~Context_Manager() 
{
	free(platformIDs);
	free(deviceIDs);
}


cl_device_id * Context_Manager::getOptimalDevice() 
{
	//return &(deviceIDs[selectedDevice]);
	return deviceIDs;
}


cl_platform_id * Context_Manager::getOptimalPlatform() 
{
	return &(platformIDs[0]);
}


cl_context * Context_Manager::getOptimalContext() 
{
	return &context;
}




// Function to check and handle OpenCL errors
void Context_Manager::checkErr(cl_int err, const char * name)
{
	if (err != CL_SUCCESS) {
		std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}




// identify platforms and choose first platform on list
void Context_Manager::getPlatformInfo(cl_platform_id ** platformIDs, cl_uint * numPlatforms, cl_uint selectedPlatform, cl_uint selectedDevice)
{
	errNum = clGetPlatformIDs(0, NULL, numPlatforms);
	checkErr(
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
		"clGetPlatformIDs");

	*platformIDs = (cl_platform_id *)malloc(sizeof(cl_platform_id)* *numPlatforms);

	int maxComputeUnits;

	errNum = clGetPlatformIDs(*numPlatforms, *platformIDs, NULL);
	checkErr(
		(errNum != CL_SUCCESS) ? errNum : (*numPlatforms <= 0 ? -1 : CL_SUCCESS),
		"clGetPlatformIDs");

}

void Context_Manager::getBestDeviceOnPlatform(cl_platform_id ** platformIDs, cl_uint platformNum,
	cl_uint * maxCompute, cl_uint *selectedPlatform, cl_uint * selectedDevice)
{
	cl_uint numDevices;
	errNum = clGetDeviceIDs((*platformIDs)[platformNum], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	cl_device_id *tempDeivceIDs = (cl_device_id *)malloc(sizeof(cl_device_id) * numDevices);
	if (numDevices > 0) 
	{
		errNum = clGetDeviceIDs(
			(*platformIDs)[platformNum],
			CL_DEVICE_TYPE_GPU,
			numDevices,
			tempDeivceIDs,
			NULL);
		checkErr(errNum, "clGetDeviceIDs");

		for (unsigned int deviceIndex = 0; deviceIndex < numDevices; deviceIndex++)
		{
			cl_uint maxComputeNewDevice = 0;
			size_t size;
			errNum = clGetDeviceInfo(tempDeivceIDs[deviceIndex], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_uint), &maxComputeNewDevice, &size);
			if (maxComputeNewDevice > *maxCompute)
			{
				*selectedPlatform = platformNum;
				*selectedDevice = deviceIndex;
				*maxCompute = maxComputeNewDevice;
			}
		}
	}
	free(tempDeivceIDs);
}


// identify devices and choose first OpenCL compatible device on list
int Context_Manager::selectOptimalDevice(cl_platform_id * platformIDs, cl_device_id ** deviceIDs,
	cl_uint * selectedPlatform, cl_uint * selectedDevice, cl_uint * numDevices, cl_uint numPlatforms)
{
	unsigned int maxCompute = 0;
	for (unsigned int platformNum = 0; platformNum < numPlatforms; platformNum++)
	{
		getBestDeviceOnPlatform(&platformIDs, platformNum, &maxCompute, selectedPlatform, selectedDevice);
	}

	if (maxCompute == 0)
	{
		std::cout << " Error, no OpenCL compatible GPU device found " << std::endl;
		return 0;
	}

	if (debug)
	{
		std::cout << "best selected device " << *selectedDevice << " found on platform " << *selectedPlatform << std::endl;
	}

	*numDevices = *selectedDevice + 1;
	*deviceIDs = (cl_device_id *)malloc(sizeof(cl_device_id)* *numDevices);

	errNum = clGetDeviceIDs(
		platformIDs[*selectedDevice],
		CL_DEVICE_TYPE_GPU,
		*numDevices,
		*deviceIDs,
		NULL);
	checkErr(errNum, "clGetDeviceIDs");

	if (debug) { printDeviceInfo(); }

}

void Context_Manager::printDeviceInfo()
{
	for (unsigned int deviceIndex = 0; deviceIndex < numDevices; deviceIndex++)
	{
		printDeviceInfo(deviceIndex);
	}
}

void Context_Manager::printDeviceInfo(cl_uint deviceNum)
{
	char buffer[1024];
	size_t max_work_size;
	cl_uint maxDimensions;
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
		std::cout << "Global memory size (mb) " << global_mem_size / (1024 * 1024) << std::endl;

	}
}


// create context and attach selected devices
void Context_Manager::setupContext(cl_platform_id * platformIDs, cl_context * context, cl_device_id * deviceIDs, cl_uint platform, cl_uint numDevices)
{
	cl_int errNum;
	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platformIDs[platform],
		0
	};
	size_t size;
	int maxComputeNewDevice;


	*context = clCreateContext(
		contextProperties,
		numDevices,
		deviceIDs,
		NULL,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateContext");
}