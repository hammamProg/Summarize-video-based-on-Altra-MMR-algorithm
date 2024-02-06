import pickle 

def download_object_as_pickle(file_name, data):
    """Download any object by passing the file_name which is the file that will be downloaded, and the data which is the object you want to save"""
    file = open(file_name, 'wb')
    pickle.dump(data, file)
    file.close()
    
def upload_pickle_object(file_name):
    """Restore any pickle object by passing the file_name (the path for that object in your worksapce)"""
    file = open(file_name, 'rb')
    data = pickle.load(file)
    file.close()
    return data

# save the cnn_features, and make sure to download it to your computer to save time of cnn_feature_extraction take which goes up to 30 min or more
#download_object_as_pickle('features', frames_features_vector_by_cnn)

# Upload your pickle file to restore cnn_features_object 
# features_filename = "/kaggle/input/cnn-features-extracted/features"
# frames_features_vector_by_cnn = upload_pickle_object(features_filename)