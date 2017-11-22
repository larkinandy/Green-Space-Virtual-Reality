# include "Average.h"

Average::Average(cl_context * contextPtr, cl_device_id * deviceIDs, cl_uint numDevices): DeviceBaseClass(contextPtr, deviceIDs, numDevices)
{
	
}

Average::Average() : DeviceBaseClass() 
{

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

	createProgram(1, deviceIDs);
	createBuffers(numBuffers, NUM_ELEMENTS_PER_BUFFER);
	ceateCommandQueues(numBuffers, deviceIDs, &queues, buffers, &kernels);
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

