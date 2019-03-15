from nltk.metrics.distance import edit_distance
import numpy as np

def score_prediction(y_true, y_pred):
    """Function to score prediction on IAM, using Levenshtein distance
       to calculate character error rate (CER)
    
    Parameters
    ------
    y_true: list
        list of ground truth labels
    y_pred: list
        list of predicted labels

    Returns
    -------
    CER: float
        character error rate
    WER: float
        word error rate
    """
        
    words_identified = 0
    characters_identified = 0
    char_tot = 0
#    CER = 0

#    list_accuracy_characters = []
    
    for i in range(len(y_pred)):
        #pred_row = [y_true[i], y_pred[i]] 
        
        #check if date are the same
        #if pred_row[0] == pred_row[1]:
        if y_true[i] == y_pred[i]:

            words_identified += 1
            
#        if len(pred_row[1]) < len(pred_row[0]):
#            pred_row[1] += '-' * (len(pred_row[0]) - len(pred_row[1]))    
#        elif len(pred_row[1]) > len(pred_row[1]):

#            pred_row[1] = pred_row[1][0:len(pred_row[0])]

        #check the number of characters that are the same
 #       print(y_true[i])
 #       print(y_pred[i])
        
        levenshtein_distance = edit_distance(y_true[i], y_pred[i])
        n_char = np.maximum(len(y_true[i]), len(y_pred[i]))
        
        normalized_distance = levenshtein_distance/n_char

        characters_identified += normalized_distance
#        char_tot += n_char
        
#        CER += normalized_distance
        
#        print(len(y_true[i]))
#        for k in range(len(y_true[i])):
#            print()
#            if y_true[i][k] == y_pred[1][k]:
#        characters_identified += 1
#        char_tot += 1

    # array_accuracy_characters = np.asarray(list_accuracy_characters)
    CER = float((characters_identified) / len(y_true))
    WER = (len(y_pred) - words_identified)/len(y_pred) 
        
    return CER, WER
#    return WER