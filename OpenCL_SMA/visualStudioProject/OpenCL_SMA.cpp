#include "OpenCL_SMA.hpp"



SMA_Analyzer::SMA_Analyzer() 
{
	contextManager = new Context_Manager();
	contextManager->getOptimalDevices(&deviceIDs, &selectedDevice, &numDevices);
	contextManager->getOptimalContext(&context);
}

SMA_Analyzer::SMA_Analyzer(char * inputFilepath, int numObs)
{
	this->inputFilepath = inputFilepath;
	this->numObs = numObs;
	contextManager = new Context_Manager();
	contextManager->getOptimalDevices(&deviceIDs, &selectedDevice, &numDevices);
	contextManager->getOptimalContext(&context);
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
	delete contextManager;
	cout << "destroying SMA Analyzer " << endl;
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
	contextManager->printDeviceInfo();
}

// Function to check and handle OpenCL errors
void SMA_Analyzer::checkErr(cl_int err, const char * name)
{
	if (err != CL_SUCCESS) {
		std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}

cl_int SMA_Analyzer::parseCSV(char *inputFile) {

	parser = new ClParser(&context, deviceIDs, numDevices, selectedDevice);
	parser->parseFile(inputFile);
	delete parser;
	return 0;
}


// main function.  For each four sequential elements in an array, compute the average (mean) value using an OpenCL kernel
void SMA_Analyzer::getAverage(int numElements, int * inputData, float **outputData) 
{

	averager = new Average(&context, deviceIDs,numDevices,selectedDevice);
	averager->getAverage(numElements, inputData,outputData);

	std::cout << (*outputData)[0] << std::endl;

	delete averager;

}


