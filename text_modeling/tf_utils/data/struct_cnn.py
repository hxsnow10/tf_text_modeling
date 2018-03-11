# encoding=utf-8

''' cnn on graph data

1. use python to generate window-sized data, which will be convolued later
this data is lokke like 1)batch*K*max-window-size:id 2)batch*K:struct_id

2. use tf to merge data with same struct_id, then make convolution on it

'''


