import h5py

xf = h5py.File("preTrained_models/model_trained_SumMe", 'r')

print(xf.keys())