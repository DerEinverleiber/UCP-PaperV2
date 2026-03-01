# UCP-Paper V2
IEEE 57-system benchmarking data (incl. explanation of data format) taken from https://labs.ece.uw.edu/pstca/

### paper_results.ipynb
Contains main findings / Plots of paper and shows how to use the attached python files

### powergrid.py
Contains PowerGrid class, loss function 

### optimizer.py
Will contain implementation of Qunatum Optimizer

### convert_data.py
Only serves purpose of dealing with IEEE57 data. Currently it only works partially.

### data/bus_data_short.csv 
IEEE57 data which is used for busses

### data/ieee57_branch.csv 
IEEE57 data used for branches. csv was created using `convert_data.py`

### data/ieee57_bus.csv 
IEEE57 data for buses which is not yet correctly exported as csv, due to incomplete implementation of `convert_data.py`
