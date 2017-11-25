# include "Average.h"

Average::Average(cl_context * contextPtr, cl_device_id * deviceIDs, cl_uint numDevices, cl_uint preferredDevice): DeviceBaseClass(contextPtr, deviceIDs, numDevices, preferredDevice)
{
	
}

Average::Average() : DeviceBaseClass() 
{

}


Average::~Average()
{

}


void Average::getAverage(cl_uint numElements,cl_int * inputData, cl_float ** outputData) 
{

	// for measuring the elapsed time to completion
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::milliseconds milliseconds;
	Clock::time_point t0 = Clock::now();

	*outputData = (cl_float *)malloc(sizeof(cl_float) * 4);


	// setup other objects and variables
	const cl_int NUM_ELEMENTS_PER_BUFFER = 4;
	const cl_int numBuffers = numElements / NUM_ELEMENTS_PER_BUFFER;

	createProgram(1, deviceIDs,preferredDevice);
	createBuffers(numBuffers, NUM_ELEMENTS_PER_BUFFER);
	ceateCommandQueues(4);
	std::cout << "copying data to buffer" << std::endl;
	copyDataToBuffer(0,0, inputData, NUM_ELEMENTS_PER_BUFFER);
	copyDataToBuffer(1, 1, &inputData[4], NUM_ELEMENTS_PER_BUFFER);
	copyDataToBuffer(2, 2, &inputData[8], NUM_ELEMENTS_PER_BUFFER);
	copyDataToBuffer(3, 3, &inputData[12], NUM_ELEMENTS_PER_BUFFER);
	std::cout << "copied data to buffer" << std::endl;
	// Run

	
	callKernels();

	// Technically don't need this as we are doing a blocking read
	// with in-order queue.
	//clWaitForEvents(events.size(), &events[0]);

	// Get results and print
	copyDataToHost(0,buffers.size()-1,*outputData, numBuffers);

	//measure elapsed time
	Clock::time_point t1 = Clock::now();
	milliseconds ms = std::chrono::duration_cast<milliseconds>(t1 - t0);
	std::cout << "Time to completion: " << ms.count() << "ms\n" << std::endl;
	

}

// create buffers and sub-buffers
void Average::createBuffers(cl_uint numBuffers, cl_uint numElementsPerBuffer)
{
	cl_uint bufferIndex = 10;
	createBuffer(sizeof(cl_int), numElementsPerBuffer, CL_MEM_READ_ONLY);
	if (debug) {
		std::cout << "buffer index: " << bufferIndex << std::endl;
	}
	createBuffer(sizeof(cl_int), numElementsPerBuffer, CL_MEM_READ_ONLY);
	if (debug) {
		std::cout << "buffer index: " << bufferIndex << std::endl;
	}
	createBuffer(sizeof(cl_int), numElementsPerBuffer, CL_MEM_READ_ONLY);
	if (debug) {
		std::cout << "buffer index: " << bufferIndex << std::endl;
	}
	createBuffer(sizeof(cl_int), numElementsPerBuffer, CL_MEM_READ_ONLY);
	if (debug) {
		std::cout << "buffer index: " << bufferIndex << std::endl;
	}
	cl_mem outputBuffer;
	outputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*(numBuffers), NULL, &errNum);
	checkErr(errNum, "clCreateBuffer");
	buffers.push_back(outputBuffer);
}

void Average::callKernels()
{
	for (cl_uint i = 0; i < queues.size(); i++)
	{
		enqeueKernel(i, 1, 0);
	}
}


// create command queues, one queue for each main and sub-buffer
void Average::ceateCommandQueues(cl_uint numBuffers)
{
	// Create command queues
	for (cl_uint i = 0; i < numBuffers; i++)
	{
		createCommandQueue(0);

		cl_kernel kernel = clCreateKernel(
			program,
			"rect_based_avg",
			&errNum);
		checkErr(errNum, "clCreateKernel(rect_based_avg)");

		// set arguments for kernels.  Input buffer changes, but output buffer remains constant 
		errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffers[i]);
		checkErr(errNum, "clSetKernelArg(rect_based_avg)");
		errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffers[buffers.size() - 1]);
		checkErr(errNum, "clSetKernelArg(rect_based_avg)");
		errNum = clSetKernelArg(kernel, 2, sizeof(cl_int), &i);
		checkErr(errNum, "clSetKernelArg(rect_based_avg)");
		kernels.push_back(kernel);
	}
}



