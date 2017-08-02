
import glob
import pickle
import numpy as np

combine_file_name = 'v23_AD/combined.pkl'
pkl_files = glob.glob('v23_AD/*')

X_combine = []
Y_combine = []

for i in range(750):
    skip = False
    try:
        with open(pkl_files[i], 'rb') as f:
            data = pickle.load(f)
            X_train = data['X_train']
            Y_train = data['y_train']
    except Exception:
        print('Failed to load : ' + pkl_files[i])
        skip = True
        
    if not skip:
        for j in range(np.asarray(X_train).shape[0]):
            X_combine.append(X_train[j])
            Y_combine.append(Y_train[j])

# X_combine=np.asarray(X_combine) #np arrays are huge on disk, save as list
# Y_combine=np.asarray(Y_combine)



# Save Data
import pickle
data = {}
data['X_train'],data['Y_train'] = X_combine,Y_combine
with open(combine_file_name, 'wb') as f:
    pickle.dump(data,f)















# EoF