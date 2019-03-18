import json
from sklearn.model_selection import train_test_split

def create_labels(data_folder, test_size, val_size, random_state):
    """Function to split the IAM dataset in train/val/test and 
        create a json with labels 

    Parameters
    ------
    data_folder: str
        base folder with IAM images 
    test_size: float
        size of the test set
    val_size: val_size
        size of the val set
    random_state: int
        random seed for splitting
        
    Returns
    -------
    dataset_out: dict
        dictionary containing dataset['train'], ['val'], ['test']
    """

#    maxTextLen = 50

    #filenames = []
    #labels = []

    dataset = []
    
    f=open(data_folder + 'words.txt')
    chars = set()
    for line in f:
        # ignore comment line
        if not line or line[0]=='#':
            continue

        line_split = line.strip().split(' ')
        assert len(line_split) >= 9

        # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
        filename_split = line_split[0].split('-')
#        filename = data_folder + 'words/' + filename_split[0] + '/' + filename_split[0] + '-' + filename_split[1] + '/' + line_split[0] + '.png'

        filename = filename_split[0] + '/' + filename_split[0] + '-' + filename_split[1] + '/' + line_split[0] + '.png'
        
        # GT text are columns starting at 9
        #label = ' '.join(line_split[8:])[:maxTextLen]
        label = ' '.join(line_split[8:])
        chars = chars.union(set(list(label)))

        # put sample into list
        dataset.append({'filename': filename, 'label': label})

    dataset_train_val, dataset_test = train_test_split(dataset, test_size=test_size, random_state=random_state)  
    dataset_train, dataset_val = train_test_split(dataset_train_val, test_size=val_size, random_state=random_state)      

    dataset_out = {}

    dataset_out['train'] = dataset_train
    dataset_out['val'] = dataset_val
    dataset_out['test'] = dataset_test
    
    return dataset_out

if __name__ == '__main__':

    data_folder = '../datasets/'
    filename_out = 'labels.json'
    
    test_size = 0.1 
    val_size = 0.1 
    random_state = 45
    
    dataset_out = create_labels(data_folder,  test_size = test_size, val_size = val_size, random_state = random_state)
    
    with open(data_folder + filename_out, 'w') as f:
        json.dump(dataset_out, f)
    