# Data-aware-FL
This repo provides the implementation for our [paper](https://ieeexplore.ieee.org/abstract/document/9499115) "_Data-Aware Device Scheduling for Federated Edge Learning_"

To install the requirements 
```
pip install -r requirements.txt
```

To run the preliminary simulations, set the desired options in the option file ( model, dataset ... ) and the number of clients to select in the federated_main.py file and run 

```
python federated_main.py
```

To run the DAS algorithm and its comparison, first create random values for the wireless environment and data distributions

```
python generate_random_values.py
```

Then run the optimization problems to select clients by running 
```
python optim_importance.py
python optim_age1.py
```

The federated process will use the generated user lists by these two scripts, the comparison can be run in 

```
python DAS_federated_main.py
```
