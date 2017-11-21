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

SMA_Analyzer::~SMA_Analyzer() 
{

}

int SMA_Analyzer::getSelectedPlatform() 
{
	return this->selectedPlatform;
}
int SMA_Analyzer::getSelectedDevice() 
{
	return this->selectedDevice;
}

void SMA_Analyzer::printDeviceInfo() 
{
	contextManager.printDeviceInfo();
}

// Function to check and handle OpenCL errors
void SMA_Analyzer::checkErr(cl_int err, const char * name)
{
	if (err != CL_SUCCESS) {
		std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}



// main function.  For each four sequential elements in an array, compute the average (mean) value using an OpenCL kernel
void SMA_Analyzer::getAverage(int numElements, int * inputData, float **outputData) 
{

	deviceIDs = contextManager.getOptimalDevice();
	context = *contextManager.getOptimalContext();

	averager = new Average(&context, deviceIDs);
	averager->getAverage(numElements, inputData,outputData);

	std::cout << (*outputData)[0] << std::endl;

}
