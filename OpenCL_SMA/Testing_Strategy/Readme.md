## Testing Strategy for OpenCL_SMA <br>

Testing strategy is based on evaluating each function independently while coding, followed by integrative testing of entire [blocks at each syncpoint](https://github.com/larkinandy/Green-Space-Virtual-Reality/blob/master/OpenCL_SMA/Support%20Documents/Project%20Flowchart_Nov17_17.png).  Data used to test program functionality consists of 100 datapoints.  Intermediate outputs for the test data at each milestone (i.e. syncpoint) were calculated in R and stored in .csv files.  At each milestone, intermediate outputs will be compared to expected output to validate functionality before progressing to the next block of code.  

### Data files
1. TestingHarness.xlsx - Excel file containing the operation, test condition, and valid output conditions targeted by the testing harness, organized by test blocks.
2. TestData_OpenCL_SMA.csv - Records used for program testing.
3. block2ExpectedResults_HourIndices.csv - Expected results after executing the identify hourly tweet indices operation using the test data as proram input.
4. block2ExpectedResults_keywordIndicator.csv - Expected results after executing the identify keyword indices operation using the test data as program input.
5. block3ExpectedResults.csv - Expected results after executing all of the operations in block 3.
6. CalculateExpectedValues.R - Script used to generated expected results.  Also used to compare GPU performance to serial performance in R.
