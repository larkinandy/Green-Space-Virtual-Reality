## Custom functions 

### Kernel functions
**Identify_hour_cutoffs**(int * tweet_hours, unsigned int * hour_indicators)  - takes hour of tweet posts, already sorted, and fills array of indices where the last value for each tweet for an hour is stored.  <br>
**find_tweets_with_keyword**(string[] tweet_text, char * keyword, int * keyword_indicators) – compares  characters in tweets to keyword.  If the keyword is in the tweet, store 1, otherwise store 0.  <br>
**Calc_num_keywords**(int * hour_indicators, int * keyword_indicators, int * num_tweets) – for each hour, calculate the number of tweets that include the keyword.  Stores results in vector num_tweets. <br>
**Calc_stats_all_hours**(int * hour_indicators, int * keyword_indicators, int * num_tweets, int * attributes, int * cords, int * color) – for all hours, calculate statistics. <br>
**Calc_stats_one_hour**(int * hour_indictors, int * keyword_indicators, int num_tweets, int * attributes, int * cords, int * color) – for one hour, calculate statistics <br>

### Functions called by kernels
**Calc_mean**(int * hour_indicators, int * keyword_indicators, int num_tweets, int * attribute, int * mean) – calculate mean value for one variable of interest.  Not a kernel, but used as a function in kernels <br>
**Calc_variance**(int * hour_indicators, int keywod_indicators, int mean, int * attribute, int *variance) – calculate standard deviation for one variable of interest.  Not a kernel, but used as a function in kernels. <br>
**Convert_sentiment_to_color**(int sentiment, int * color) – convert sentiment value to color <br>
**Convert_variance_to_conf_region**(int * variance_mat, int * conf_regions) – convert variance values to confidence rgions

### Host functions 
**Todo: add host functions**



