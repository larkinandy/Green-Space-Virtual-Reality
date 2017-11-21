# include "Average.h"




Average::Average(cl_context * contextPtr, cl_device_id * deviceIDs) 
{
	this->context = *contextPtr;
	this->deviceIDs = deviceIDs;
}

Average::Average() {

}


Average::~Average() 
{

}






void Average::getAverage(int numElements,int * inputData, float ** outputData) 
{

	// for measuring the elapsed time to completion
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::milliseconds milliseconds;
	Clock::time_point t0 = Clock::now();

	*outputData = (float *)malloc(sizeof(float) * 4);


	// setup other objects and variables
	const int NUM_ELEMENTS_PER_BUFFER = 4;
	const int numBuffers = numElements / NUM_ELEMENTS_PER_BUFFER;

	createProgram(&program, context, 1, deviceIDs);
	createBuffers(&inputData, numBuffers, context, &buffers, NUM_ELEMENTS_PER_BUFFER);
	ceateCommandQueues(numBuffers, deviceIDs, context, &queues, buffers, program, &kernels);
	copyDataToBuffer(&queues, &buffers, inputData, numElements);

	// Run
	callKernels(&queues, &kernels, &events, NUM_ELEMENTS_PER_BUFFER);

	// Technically don't need this as we are doing a blocking read
	// with in-order queue.
	clWaitForEvents(events.size(), &events[0]);

	// Get results and print
	copyDataToHost(&queues, &buffers, numBuffers, *outputData);

	//measure elapsed time
	Clock::time_point t1 = Clock::now();
	milliseconds ms = std::chrono::duration_cast<milliseconds>(t1 - t0);
	std::cout << "Time to completion: " << ms.count() << "ms\n" << std::endl;
}

// Function to check and handle OpenCL errors
void Average::checkErr(cl_int err, const char * name)
{
	if (err != CL_SUCCESS) {
		std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}



// load program and .cl file and build
void Average::createProgram(cl_program * program, cl_context context, cl_int numDevices, cl_device_id * deviceIDs)
{
	cl_int errNum;

	std::ifstream srcFile(KERNEL_FILE);
	checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading kernel file");

	std::string srcProg(
		std::istreambuf_iterator<char>(srcFile),
		(std::istreambuf_iterator<char>()));

	const char * src = srcProg.c_str();
	size_t length = srcProg.length();


	// Create program from source
	*program = clCreateProgramWithSource(
		context,
		1,
		&src,
		&length,
		&errNum);
	checkErr(errNum, "clCreateProgramWithSource");

	// Build program
	errNum = clBuildProgram(
		*program,
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
			*program,
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
void Average::createBuffers(int ** inputOutput, int numBuffers, cl_context context, std::vector<cl_mem> * buffers, int numElementsPerBuffer)
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
	buffers->push_back(buffer);

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

		buffers->push_back(newBuffer);
	}


	cl_mem outputBuffer;
	outputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*(numBuffers), NULL, &errNum);
	checkErr(errNum, "clCreateBuffer");
	buffers->push_back(outputBuffer);
}

// create command queues, one queue for each main and sub-buffer
void Average::ceateCommandQueues(int numBuffers, cl_device_id * deviceIDs, cl_context context, std::vector<cl_command_queue> * queues, std::vector<cl_mem> buffers,
	cl_program program, std::vector<cl_kernel> *kernels)
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
void Average::copyDataToBuffer(std::vector<cl_command_queue> * queues, std::vector<cl_mem> * buffers, int * inputOutput, int numElements)
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
}

// execute all kernels
void Average::callKernels(std::vector<cl_command_queue> * queues, std::vector<cl_kernel> * kernels, std::vector<cl_event> * events, int numBufferElements)
{
	cl_int errNum;

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

		events->push_back(event);
	}
}

//return results from GPU to CPU memory
void Average::copyDataToHost(std::vector<cl_command_queue> * queues, std::vector<cl_mem> * buffers, cl_uint numBuffers, float * outputVals)
{
	cl_int errNum;

	// Read back computed data
	clEnqueueReadBuffer(
		(*queues)[0],
		(*buffers)[buffers->size() - 1],
		CL_TRUE,
		0,
		sizeof(float)*numBuffers,
		(void*)outputVals,
		0,
		NULL,
		NULL);
}