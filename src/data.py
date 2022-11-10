import re
#------------------------------------------------------------

def data_preprocess(data, type: str):
    
    """
    A function to retrieve a data from a  given url 
    and then preprocess to store cleaned label and 
    train dataset into our data directory 
    ------------------
    Parameters:
    ------------------
    """

    label = []
    features = []

    with open(data, 'r') as f:
        lines = f.readlines()
    if type == "train":
        labels = [int(l[0]) for l in lines]
        for index, item in enumerate(labels):
            if (item == 0):
                labels[index] = -1
        columns = [re.sub(r'[^\w]', ' ',l[1:]).split() for l in lines]
         
    else:
        label = []
        columns = [re.sub(r'[^\w]', ' ',l).split() for l in lines]
    for col in columns:
        lines = [0]*100001
        for idx, val in enumerate(col):
            lines[int(val)] = 1
        features.append(lines)
    return features, label
#if __name__ == '__main__':
    #train_features, label = data_preprocess(train_path, 'train')
    #test_features, label = data_preprocess(test_path, "test")