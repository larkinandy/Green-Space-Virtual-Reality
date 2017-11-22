#include "testDriver.hpp"
///
//	main() function
//

// print results of kernel operation
void printOutput(int numElements, float * dataset)
{
	std::cout << "averages of sequential 4 elements for an input array with " << numElements << " elements" << std::endl;
	// Display output in rows
	for (int index = 0; index < numElements; index++)
	{
		std::cout << " " << dataset[index];
	}
	std::cout << std::endl;
}

void createInput(int numElements, int **dataset) 
{
	*dataset = (int*)malloc(sizeof(int)* numElements);
	for (int index = 0; index < numElements; index++) {
		(*dataset)[index] = index;
	}
}




int main(int argc, char** argv)
{

	//_CrtSetDbgFlag(_CRTDBG_LEAK_CHECK_DF);
	
	
	SMA_Analyzer analyzer = SMA_Analyzer();
	
	
	
	int * testData1 = NULL;
	float * output1 = NULL;
	float * output2 = NULL;
	createInput(16,&testData1);
	
	
	analyzer.getAverage(16, testData1,&output1);
	int *testData2 = NULL;
	createInput(16,&testData2);
	analyzer.getAverage(16, testData2,&output2);
	


	printOutput(4,output2);
	
	free(testData1);
	free(testData2);
	free(output1);
	free(output2);

	std::cout << "Program completed successfully" << std::endl;

	
	return 0;
	
	
}
