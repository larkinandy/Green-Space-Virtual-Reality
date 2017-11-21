#include "OpenCL_SMA.h"

SMA_Analyzer::SMA_Analyzer() 
{

}

SMA_Analyzer::SMA_Analyzer(char * inputFilepath, int numObs)
{
	this->inputFilepath = inputFilepath;
	this->numObs = numObs;
}

void SMA_Analyzer::setFilepath(char * filepath) 
{
	this->inputFilepath = filepath;
}

void SMA_Analyzer::setNumobs(int numObs) 
{
	this->numObs = numObs;
}

char * SMA_Analyzer::getFilepath() 
{
	return inputFilepath;
}

int SMA_Analyzer::getNumObs() 
{
	return numObs;
}


// Function to check and handle OpenCL errors
void SMA_Analyzer::checkErr(cl_int err, const char * name)
{
	if (err != CL_SUCCESS) {
		std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}

// identify platforms and choose first platform on list
void SMA_Analyzer::setupPlatform(cl_platform_id ** platformIDs, cl_uint numPlatforms, int platform)
{
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr(
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
		"clGetPlatformIDs");

	*platformIDs = (cl_platform_id *)malloc(sizeof(cl_platform_id) * 2);
	
	if (debug) { std::cout << "Number of platforms: \t" << numPlatforms << std::endl; 
	}
	errNum = clGetPlatformIDs(numPlatforms, *platformIDs, NULL);
	checkErr(
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
		"clGetPlatformIDs");

	if (debug) { DisplayPlatformInfo( *platformIDs[platform], CL_PLATFORM_VENDOR, "CL_PLATFORM_VENDOR"); }
}

// identify devices and choose first OpenCL compatible device on list
void SMA_Analyzer::setupDevices(cl_platform_id * platformIDs, cl_device_id ** deviceIDs, int platform, cl_uint * numDevices) 
{
	cl_int errNum;
	errNum = clGetDeviceIDs(
		platformIDs[platform],
		CL_DEVICE_TYPE_ALL,
		0,
		NULL,
		numDevices);
	if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
	{
		checkErr(errNum, "clGetDeviceIDs");
	}

	*deviceIDs = (cl_device_id *)malloc(sizeof(cl_device_id) * *numDevices);
	errNum = clGetDeviceIDs(
		platformIDs[platform],
		CL_DEVICE_TYPE_ALL,
		*numDevices,
		deviceIDs[0],
		NULL);
	checkErr(errNum, "clGetDeviceIDs");


	InfoDevice<cl_device_type>::display(
		*deviceIDs[0],
		CL_DEVICE_TYPE,
		"CL_DEVICE_TYPE");

}

// create context and attach selected devices
void SMA_Analyzer::setupContext(cl_platform_id * platformIDs, cl_context * context, cl_device_id * deviceIDs, int platform, cl_uint numDevices) 
{
	cl_int errNum;

	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platformIDs[platform],
		0
	};

	*context = clCreateContext(
		contextProperties,
		numDevices,
		deviceIDs,
		NULL,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateContext");
}

// load program and .cl file and build
void SMA_Analyzer::createProgram(cl_program * program, cl_context context, cl_int numDevices, cl_device_id * deviceIDs) 
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

// print results of kernel operation
void SMA_Analyzer::printOutput(int numDevices, float * results, int numElements)
{
	std::cout << "averages of sequential 4 elements for an input array with " << numElements << " elements" << std::endl;
	// Display output in rows
	for (int index = 0;  index < numElements; index++)
	{
		std::cout << " " << results[index];
		}
		std::cout << std::endl;
}

// create buffers and sub-buffers
void SMA_Analyzer::createBuffers(int ** inputOutput, int numBuffers, cl_context context, std::vector<cl_mem> * buffers, int numElementsPerBuffer)
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

// creat input data for testing the 'getAverage' function 
void SMA_Analyzer::createInput(int ** inputOutput, int numBufferElements) 
{
	*inputOutput = new int[numBufferElements];
	for (unsigned int i = 0; i < numBufferElements ; i++)
	{
		(*inputOutput)[i] = i;
	}
}

// create command queues, one queue for each main and sub-buffer
void SMA_Analyzer::ceateCommandQueues(int numBuffers, cl_device_id * deviceIDs, cl_context context, std::vector<cl_command_queue> * queues, std::vector<cl_mem> buffers, 
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
		errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffers[buffers.size()-1]);
		errNum = clSetKernelArg(kernel, 2, sizeof(cl_int), &i);
		
		checkErr(errNum, "clSetKernelArg(rect_based_avg)");
		kernels->push_back(kernel);
	}
}

// copy data from device to the main buffer. 
void SMA_Analyzer::copyDataToBuffer(std::vector<cl_command_queue> * queues, std::vector<cl_mem> * buffers, int * inputOutput, int numElements)
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
void SMA_Analyzer::callKernels(std::vector<cl_command_queue> * queues, std::vector<cl_kernel> * kernels, std::vector<cl_event> * events, int numBufferElements)
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
void SMA_Analyzer::copyDataToHost(std::vector<cl_command_queue> * queues, std::vector<cl_mem> * buffers, cl_uint numBuffers, float * outputVals) 
{
	cl_int errNum;

	// Read back computed data
	clEnqueueReadBuffer(
		(*queues)[0],
		(*buffers)[buffers->size()-1],
		CL_TRUE,
		0,
		sizeof(float)*numBuffers,
		(void*)outputVals,
		0,
		NULL,
		NULL);
}

// main function.  For each four sequential elements in an array, compute the average (mean) value using an OpenCL kernel
void SMA_Analyzer::getAverage(int numElements, int * inputData) 
{
	// setup opencl objects and variables
	cl_int errNum;
	cl_uint numPlatforms = NULL;
	cl_uint numDevices;
	cl_platform_id * platformIDs = NULL;
	cl_device_id * deviceIDs = NULL;
	cl_context context = NULL;
	cl_program program;
	std::vector<cl_kernel> kernels;
	std::vector<cl_command_queue> queues;
	std::vector<cl_mem> buffers;
	std::vector<cl_event> events;

	// for measuring the elapsed time to completion
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::milliseconds milliseconds;
	Clock::time_point t0 = Clock::now();

	// setup other objects and variables
	const int NUM_ELEMENTS_PER_BUFFER = 4;
	const int numBuffers = numElements / NUM_ELEMENTS_PER_BUFFER;
	float * subBufferAvgs = (float *)malloc(sizeof(float) * numBuffers);

	// Setup 
	setupPlatform(&platformIDs, numPlatforms,0);
	setupDevices(platformIDs, &deviceIDs, 0, &numDevices);
	setupContext(platformIDs, &context, deviceIDs, 0, 1);
	createProgram(&program, context, 1, deviceIDs);
	createBuffers(&inputData, numBuffers, context, &buffers, NUM_ELEMENTS_PER_BUFFER);
	ceateCommandQueues(numBuffers, deviceIDs, context, &queues, buffers,program, &kernels);
	copyDataToBuffer(&queues, &buffers, inputData, numElements);
	
	// Run
	callKernels(&queues, &kernels, &events, NUM_ELEMENTS_PER_BUFFER);
	
	// Technically don't need this as we are doing a blocking read
	// with in-order queue.
	clWaitForEvents(events.size(), &events[0]);

	// Get results and print
	copyDataToHost(&queues, &buffers, numBuffers , subBufferAvgs);
	printOutput(numDevices, subBufferAvgs, numBuffers);

	// free memory
	free(platformIDs);
	free(deviceIDs);
	free(subBufferAvgs);

	//measure elapsed time
	Clock::time_point t1 = Clock::now();
	milliseconds ms = std::chrono::duration_cast<milliseconds>(t1 - t0);
	std::cout << "Time to completion: " << ms.count() << "ms\n" << std::endl;
}
