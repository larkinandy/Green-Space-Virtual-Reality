#include "DeviceBaseClass.hpp"


DeviceBaseClass::DeviceBaseClass(cl_context * contextPtr, cl_device_id * deviceIDs, cl_uint numDevices)
{
	this->context = *contextPtr;
	this->deviceIDs = deviceIDs;
	this->numDevices = numDevices;
}

DeviceBaseClass::DeviceBaseClass() {

}


DeviceBaseClass::~DeviceBaseClass()
{
	releaseBuffers();
	releaseEvents();
	releaseKernels();
	releaseCommandQueues();
	releaseProgram();
}





// load program and .cl file and build
void DeviceBaseClass::createProgram(cl_int numDevices, cl_device_id * deviceIDs)
{
	std::ifstream srcFile(KERNEL_FILE);
	checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading kernel file");

	std::string srcProg(
		std::istreambuf_iterator<char>(srcFile),
		(std::istreambuf_iterator<char>()));

	const char * src = srcProg.c_str();
	size_t length = srcProg.length();


	// Create program from source
	program = clCreateProgramWithSource(
		context,
		1,
		&src,
		&length,
		&errNum);
	checkErr(errNum, "clCreateProgramWithSource");

	// Build program
	errNum = clBuildProgram(
		program,
		numDevices,
		deviceIDs,
		"-I.",
		NULL,
		NULL);
	if (errNum != CL_SUCCESS)
	{
		// Determine the reason for the error
		char buildLog[16384];
		clGetProgramBuildInfo(
			program,
			deviceIDs[0],
			CL_PROGRAM_BUILD_LOG,
			sizeof(buildLog),
			buildLog,
			NULL);

		std::cerr << "Error in OpenCL C source: " << std::endl;
		std::cerr << buildLog;
		checkErr(errNum, "clBuildProgram");
	}
}

// create buffers and sub-buffers
void DeviceBaseClass::createBuffers(int numBuffers, int numElementsPerBuffer)
{
	cl_int errNum;

	// create a single buffer to cover all the input data
	cl_mem buffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		sizeof(int) * numElementsPerBuffer*numBuffers,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer");
	buffers.push_back(buffer);

	// now for all buffersRects other than the first create a sub-buffer
	cl_mem newBuffer;
	for (unsigned int i = 1; i < numBuffers; i++)
	{
		cl_buffer_region region =
		{
			numElementsPerBuffer * i * sizeof(int),
			numElementsPerBuffer * sizeof(int)
		};
		newBuffer = clCreateSubBuffer(
			buffer,
			CL_MEM_READ_ONLY,
			CL_BUFFER_CREATE_TYPE_REGION,
			&region,
			&errNum);
		checkErr(errNum, "clCreateSubBuffer");

		buffers.push_back(newBuffer);
	}


	cl_mem outputBuffer;
	outputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*(numBuffers), NULL, &errNum);
	checkErr(errNum, "clCreateBuffer");
	buffers.push_back(outputBuffer);
}

// create command queues, one queue for each main and sub-buffer
void DeviceBaseClass::ceateCommandQueues(int numBuffers, cl_device_id * deviceIDs, std::vector<cl_command_queue> * queues,
	std::vector<cl_mem> buffers, std::vector<cl_kernel> *kernels)
{
	cl_int errNum;

	// Create command queues
	for (unsigned int i = 0; i < numBuffers; i++)
	{
		cl_command_queue queue =
			clCreateCommandQueue(
				context,
				deviceIDs[0],
				0,
				&errNum);
		checkErr(errNum, "clCreateCommandQueue");

		queues->push_back(queue);

		cl_kernel kernel = clCreateKernel(
			program,
			"rect_based_avg",
			&errNum);
		checkErr(errNum, "clCreateKernel(rect_based_avg)");

		// set arguments for kernels.  Input buffer changes, but output buffer remains constant 
		errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffers[i]);
		errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffers[buffers.size() - 1]);
		errNum = clSetKernelArg(kernel, 2, sizeof(cl_int), &i);

		checkErr(errNum, "clSetKernelArg(rect_based_avg)");
		kernels->push_back(kernel);
	}
}

// copy data from device to the main buffer. 
void DeviceBaseClass::copyDataToBuffer(std::vector<cl_command_queue> * queues, std::vector<cl_mem> * buffers, int * inputOutput, int numElements)
{
	cl_int errNum;

	//Write input data
	errNum = clEnqueueWriteBuffer(
		(*queues)[0],
		(*buffers)[0],
		CL_TRUE,
		0,
		sizeof(int) * numElements,
		(void*)inputOutput,
		0,
		NULL,
		NULL);

	checkErr(errNum, "host to device");
}

// execute all kernels
void DeviceBaseClass::callKernels(std::vector<cl_command_queue> * queues, std::vector<cl_kernel> * kernels, std::vector<cl_event> * events, int numBufferElements)
{
	for (unsigned int i = 0; i < queues->size(); i++)
	{
		cl_event event;

		size_t globalSize[1] = { 1 };
		size_t localSize[1] = { 1 };

		errNum = clEnqueueNDRangeKernel(
			(*queues)[i],
			(*kernels)[i],
			1,
			NULL,
			globalSize,
			localSize,
			0,
			NULL,
			&event);

		checkErr(errNum, "enqeue kernel");
		events->push_back(event);

	}
}

//return results from GPU to CPU memory
void DeviceBaseClass::copyDataToHost(std::vector<cl_command_queue> * queues, std::vector<cl_mem> * buffers, cl_uint numBuffers, float * outputVals)
{
	// Read back computed data
	errNum = clEnqueueReadBuffer(
		(*queues)[0],
		(*buffers)[buffers->size() - 1],
		CL_TRUE,
		0,
		sizeof(float)*numBuffers,
		(void*)outputVals,
		0,
		NULL,
		NULL);

	checkErr(errNum, "device to host");
}

void DeviceBaseClass::releaseCommandQueues()
{
	for (cl_uint queueIndex = 0; queueIndex < queues.size(); queueIndex++)
	{
		errNum = clReleaseCommandQueue(queues[queueIndex]);
		checkErr(errNum, "command queue release");

	}


}
void DeviceBaseClass::releaseKernels()
{
	for (cl_uint kernelIndex = 0; kernelIndex < kernels.size(); kernelIndex++)
	{
		errNum = clReleaseKernel(kernels[kernelIndex]);
		checkErr(errNum, "kernel release");
	}
}



void DeviceBaseClass::releaseBuffers()
{
	for (int bufferIndex = 0; bufferIndex < buffers.size(); bufferIndex++)
	{
		errNum = clReleaseMemObject(buffers[bufferIndex]);
		checkErr(errNum, "release Buffer");
	}

}

void DeviceBaseClass::releaseProgram()
{
	clReleaseProgram(program);
}

void DeviceBaseClass::releaseEvents()
{
	for (int eventIndex = 0; eventIndex < events.size(); eventIndex++)
	{
		errNum = clReleaseEvent(events[eventIndex]);
		checkErr(errNum, "release events");
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