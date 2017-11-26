#include "DeviceBaseClass.hpp"

DeviceBaseClass::DeviceBaseClass(cl_context * contextPtr, cl_device_id * deviceIDs, cl_uint numDevices,cl_uint preferredDevice)
{
	this->context = *contextPtr;
	this->deviceIDs = deviceIDs;
	this->numDevices = numDevices;
	this->preferredDevice = preferredDevice;
	numWorkItems = (cl_uint*)malloc(sizeof(cl_uint)*numDevices);
	cl_uint maxDimensions;
	size_t size;
	//cl_ulong global_mem_size;
	for (cl_uint deviceIndex = 0; deviceIndex < numDevices; deviceIndex++)
	{
		clGetDeviceInfo(deviceIDs[deviceIndex], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &numWorkItems[deviceIndex], &size);
		if (debug) {
			std::cout << "Max work group units: " << numWorkItems[deviceIndex] << std::endl;
			//clGetDeviceInfo(deviceIDs[deviceIndex], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, &size); //maybe use later during optimization
			//std::cout << "Global memory size (mb) " << global_mem_size / (1024 * 1024) << std::endl << std::endl;
		}
	}
}

DeviceBaseClass::DeviceBaseClass() 
{
}


DeviceBaseClass::~DeviceBaseClass()
{
	releaseBuffers();
	releaseEvents();
	releaseKernels();
	releaseCommandQueues();
	releaseProgram();
	free(numWorkItems);
}



// load program and .cl file and build
void DeviceBaseClass::createProgram(cl_uint numDevices, cl_device_id * deviceIDs,cl_uint deviceNum)
{
	std::ifstream srcFile(deviceFunctionFile);
	checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading kernel file");

	std::string srcProg(
		std::istreambuf_iterator<char>(srcFile),
		(std::istreambuf_iterator<char>()));

	const char * src = srcProg.c_str();
	size_t length = srcProg.length();


	// Create program from source
	program = clCreateProgramWithSource(context,1,&src,&length,&errNum);
	checkErr(errNum, "clCreateProgramWithSource");

	// Build program
	errNum = clBuildProgram(program,numDevices,deviceIDs,"-I.",NULL,NULL);
	if (errNum != CL_SUCCESS)
	{
		// Determine the reason for the error
		char buildLog[16384];
		
		clGetProgramBuildInfo(program,
			deviceIDs[deviceNum],
			CL_PROGRAM_BUILD_LOG,
			sizeof(buildLog),
			buildLog,
			NULL);

		std::cerr << "Error in OpenCL C source: " << std::endl;
		std::cerr << buildLog;
		checkErr(errNum, "clBuildProgram");
	}
}


void  DeviceBaseClass::createBuffer(cl_uint elemSize, std::vector<cl_mem> * varBuffers, cl_uint numElements, cl_mem_flags typeFlag)
{
	// create a single buffer to cover all the input data
	cl_mem buffer = clCreateBuffer(context, typeFlag, elemSize * numElements, NULL, &errNum);
	checkErr(errNum, "clCreateBuffer");
	varBuffers->push_back(buffer);
}


void  DeviceBaseClass::createBuffer(cl_uint elemSize, cl_uint numElements, cl_mem_flags typeFlag)
{
	// create a single buffer to cover all the input data
	cl_mem buffer = clCreateBuffer(context, typeFlag, elemSize * numElements, NULL, &errNum);
	checkErr(errNum, "clCreateBuffer");
	buffers.push_back(buffer);
}




void DeviceBaseClass::createCommandQueue(cl_uint deviceNum) 
{
	cl_command_queue queue =
		clCreateCommandQueue(
			context,
			deviceIDs[deviceNum],
			0,
			&errNum);
	checkErr(errNum, "clCreateCommandQueue");
	queues.push_back(queue);
}




// copy data from device to the main buffer. 
void DeviceBaseClass::copyDataToBuffer(cl_uint queueNumber, cl_uint bufferNumber, cl_int * hostData, cl_uint numElements)
{
	//Write input data
	errNum = clEnqueueWriteBuffer(
		queues[queueNumber],
		buffers[bufferNumber],
		CL_FALSE,
		0,
		sizeof(int) * numElements,
		(void*)hostData,
		0,
		NULL,
		NULL);

	checkErr(errNum, "host to device");
}


// copy data from device to the main buffer. 
void DeviceBaseClass::copyDataToBuffer(cl_uint queueNumber, cl_uint bufferNumber, cl_float * hostData, cl_uint numElements)
{
	//Write input data
	errNum = clEnqueueWriteBuffer(
		queues[queueNumber],
		buffers[bufferNumber],
		CL_FALSE,
		0,
		sizeof(float) * numElements,
		(void*)hostData,
		0,
		NULL,
		NULL);

	checkErr(errNum, "host to device");
}

// copy data from device to the main buffer. 
void DeviceBaseClass::copyDataToBuffer(cl_uint queueNumber, cl_uint bufferNumber, char * hostData, cl_uint numElements)
{
	//Write input data
	errNum = clEnqueueWriteBuffer(
		queues[queueNumber],
		buffers[bufferNumber],
		CL_FALSE,
		0,
		sizeof(char) * numElements,
		(void*)hostData,
		0,
		NULL,
		NULL);

	checkErr(errNum, "host to device");
}


// copy data from device to the main buffer. 
void DeviceBaseClass::copyDataToBuffer(cl_uint queueNumber, cl_mem * buffer, char * hostData, cl_uint numElements)
{
	//Write input data
	errNum = clEnqueueWriteBuffer(
		queues[queueNumber],
		*buffer,
		CL_FALSE,
		0,
		sizeof(char) * numElements,
		(void*)hostData,
		0,
		NULL,
		NULL);

	checkErr(errNum, "host to device");
}


// copy data from device to the main buffer. 
void DeviceBaseClass::copyDataToBuffer(cl_uint queueNumber, cl_uint bufferNumber, char * hostData, cl_uint numElements, cl_event * event)
{
	//Write input data
	errNum = clEnqueueWriteBuffer(
		queues[queueNumber],
		buffers[bufferNumber],
		CL_FALSE,
		0,
		sizeof(char) * numElements,
		(void*)hostData,
		0,
		NULL,
		event);

	checkErr(errNum, "host to device");

	events.push_back(*event);

}

// execute all kernels


void DeviceBaseClass::enqeueKernel(cl_uint kernelNum, cl_uint numThreads, cl_uint deviceNum) {
	cl_event event;
	cl_uint numLocalThreads = std::min(numThreads,numWorkItems[deviceNum]);
	
	size_t globalSize[1] = {numThreads };
	size_t localSize[1] = { numLocalThreads };
	
	errNum = clEnqueueNDRangeKernel(
		queues[kernelNum],
		kernels[kernelNum],
		1,
		NULL,
		globalSize,
		localSize,
		0,
		NULL,
		&event);

	events.push_back(event);

}

void DeviceBaseClass::enqeueKernel(cl_uint queueNum,cl_uint kernelNum, cl_uint numThreads, cl_uint deviceNum) {
	cl_event event;
	cl_uint numLocalThreads = std::min(numThreads, numWorkItems[deviceNum]/2) ;
	cl_uint globalThreads = ((numThreads + numLocalThreads - 1) / numLocalThreads)*numLocalThreads;
	size_t globalSize[1] = { globalThreads };
	size_t localSize[1] = { numLocalThreads };

	errNum = clEnqueueNDRangeKernel(
		queues[queueNum],
		kernels[kernelNum],
		1,
		NULL,
		globalSize,
		localSize,
		0,
		NULL,
		&event);
	checkErr(errNum, "enqueue kernel");
	events.push_back(event);

}


void DeviceBaseClass::enqeueKernel(cl_uint queueNum, cl_uint kernelNum, cl_uint numThreads, cl_uint deviceNum,cl_event * priorEvent) {
	cl_event event;
	cl_uint numLocalThreads = std::min(numThreads, numWorkItems[deviceNum] / 2);
	cl_uint globalThreads = ((numThreads + numLocalThreads - 1) / numLocalThreads)*numLocalThreads;
	size_t globalSize[1] = { globalThreads };
	size_t localSize[1] = { numLocalThreads };

	errNum = clEnqueueNDRangeKernel(
		queues[queueNum],
		kernels[kernelNum],
		1,
		NULL,
		globalSize,
		localSize,
		1,
		priorEvent,
		&event);
	checkErr(errNum, "enqueue kernel");
	events.push_back(event);
}



void DeviceBaseClass::copyDataToHost(cl_uint queueNum,cl_uint bufferNum, cl_float * outputVals, cl_uint numElements)
{
	errNum = clEnqueueReadBuffer(
		queues[queueNum],
		buffers[bufferNum],
		CL_FALSE,
		0,
		sizeof(cl_float)*numElements,
		(void*)outputVals,
		0,
		NULL,
		NULL);
	checkErr(errNum, "device to host");

}


void DeviceBaseClass::copyDataToHost(cl_uint queueNum, cl_uint bufferNum, cl_int * outputVals, cl_uint numElements)
{
	errNum = clEnqueueReadBuffer(
		queues[queueNum],
		buffers[bufferNum],
		CL_FALSE,
		0,
		sizeof(cl_int)*numElements,
		(void*)outputVals,
		0,
		NULL,
		NULL);
	checkErr(errNum, "device to host");
}

void DeviceBaseClass::copyDataToHost(cl_uint queueNum, cl_mem buffer, cl_int * outputVals, cl_uint numElements)
{
	errNum = clEnqueueReadBuffer(
		queues[queueNum],
		buffer,
		CL_FALSE,
		0,
		sizeof(cl_int)*numElements,
		(void*)outputVals,
		0,
		NULL,
		NULL);
	checkErr(errNum, "device to host");
}

void DeviceBaseClass::copyDataToHost(cl_uint queueNum, cl_uint bufferNum, cl_char * outputVals, cl_uint numElements)
{
	errNum = clEnqueueReadBuffer(
		queues[queueNum],
		buffers[bufferNum],
		CL_FALSE,
		0,
		sizeof(char)*numElements,
		(void*)outputVals,
		0,
		NULL,
		NULL);
	checkErr(errNum, "device to host");
}

void DeviceBaseClass::copyDataToHost(cl_uint queueNum, cl_mem buffer, cl_char * outputVals, cl_uint numElements)
{
	errNum = clEnqueueReadBuffer(
		queues[queueNum],
		buffer,
		CL_FALSE,
		0,
		sizeof(char)*numElements,
		(void*)outputVals,
		0,
		NULL,
		NULL);
	checkErr(errNum, "device to host");
}



void DeviceBaseClass::releaseCommandQueues()
{
	cl_uint numReleases;
	for (cl_int queueIndex = queues.size()-1; queueIndex >=0; queueIndex--)
	{
		clGetCommandQueueInfo(queues[queueIndex], CL_QUEUE_REFERENCE_COUNT, sizeof(cl_uint), &numReleases, NULL);
		for (cl_int releaseCount = numReleases; releaseCount > 0; releaseCount--) {
			errNum = clReleaseCommandQueue(queues[queueIndex]);
			checkErr(errNum, "command queue release");
		}
	}


}
void DeviceBaseClass::releaseKernels()
{
	cl_uint numReleaes;
	for (cl_int kernelIndex = kernels.size() - 1; kernelIndex >= 0; kernelIndex--)
	{
		clGetKernelInfo(kernels[kernelIndex], CL_KERNEL_REFERENCE_COUNT, sizeof(cl_uint), &numReleaes, NULL);
		for (cl_int releaseCount = numReleaes; releaseCount > 0; releaseCount--)
		{
			errNum = clReleaseKernel(kernels[kernelIndex]);
			checkErr(errNum, "release Kernel");
		}
	}
}



void DeviceBaseClass::releaseBuffers()
{
	cl_uint numReleaes;
	for (cl_int bufferIndex = buffers.size()-1; bufferIndex >= 0 ; bufferIndex--)
	{
		errNum = clGetMemObjectInfo(buffers[bufferIndex], CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &numReleaes, NULL);
		checkErr(errNum, "release Buffer");
		for (cl_int releaseCount = numReleaes; releaseCount > 0; releaseCount--)
		{
			errNum = clReleaseMemObject(buffers[bufferIndex]);
			checkErr(errNum, "release Buffer");
		}
	}

}

void DeviceBaseClass::releaseProgram()
{
	cl_uint numReleaes;
	clGetProgramInfo(program, CL_PROGRAM_REFERENCE_COUNT, sizeof(cl_uint), &numReleaes, NULL);
	for (cl_int releaseCount = numReleaes; releaseCount > 0; releaseCount--)
	{
		std::cout << "releasing program" << std::endl;
		errNum = clReleaseProgram(program);
		checkErr(errNum, "release program");
	}
}

void DeviceBaseClass::releaseEvents()
{
	cl_uint numReleaes;
	for (cl_int eventIndex = events.size() - 1; eventIndex >= 0; eventIndex--)
	{
		clGetEventInfo(events[eventIndex], CL_EVENT_REFERENCE_COUNT, sizeof(cl_uint), &numReleaes, NULL);
		for (cl_int releaseCount = numReleaes; releaseCount > 0; releaseCount--)
		{
			errNum = clReleaseEvent(events[eventIndex]);
			checkErr(errNum, "release events");
		}
	}
}
// Function to check and handle OpenCL errors
void DeviceBaseClass::checkErr(cl_int err, const char * name)
{
	if (err != CL_SUCCESS) {
		std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}