# ActivityNet
2016 ActivityNet action recognition challenge. CNN + LSTM approach. Multi-threaded loading.

Notes:
- Nice example of training w/ and w/o multi-threaded loading
- Did not finish before challenge deadline (no test results)
- Bug in LSTM forced use of regular RNN (investigate error & fix if re-using)
- Need to find a faster way to load videos with lua (currently slower than training)
