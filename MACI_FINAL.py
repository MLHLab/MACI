
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


class MACI:
    # function to create the sliding window from the data and returns it in 2 arrays
    def create_dataset(self,sequence_data, class_names):
        data_final = []
        classes_final = []
        for name in class_names:
            temp = sequence_data[sequence_data['Drug Class'] == name]
            temp_seq = np.array(temp['Sequence'])
            print('Class found %s'%name)
            for seq in temp_seq:
                for i in range(100, len(seq)):
                    fragment = seq[i-100:i]
                    if len(fragment) == 100:
                        data_final.append(seq[i-100:i])
                        classes_final.append(name)
           
        return data_final, classes_final


    # function to encode the fragment absed on characters
    def encoder(self,protein): 
        temp = [] 
        for character in protein: 
            if character == 'A': 
                temp.append(1) 
            if character == 'G': 
                temp.append(0.75) 
            if character == 'C': 
                temp.append(0.25) 
            if character == 'T': 
                temp.append(0.5)
        #temp = np.array(temp)
        return temp


    # function to create the encoder required
    def create_encoder(self,class_names):
        le = LabelEncoder()
        le.fit(np.unique(np.array(class_names)))
        return le



    # function to create and train the model, we have a parameter in class_count to modify the final neuron count

    def model_create(self,X, Y_train, class_count):
        model = keras.Sequential(
        [
            layers.InputLayer(input_shape=(100,)),
            layers.Dense(100, activation='LeakyReLU'),
            layers.Dense(400, activation='LeakyReLU'),
            layers.Dense(200, activation='LeakyReLU'),
        layers.Dense(class_count, activation = 'softmax')

        ]
        )

        opt = tf.keras.optimizers.Adagrad(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.fit(X, Y_train, epochs = 30, verbose = 1)
        model.save('trained_model.h5')


    # function to calculate the final accuracy. It takes in as inputs the class names, the label encoder object, the test results and the original values
    def model_check_results(self,class_names, label_encoder_obj, test_df, model):
        final_result_class = []
        precision_values = []
        recall_values = []
        f1_values = []
        accuracy_values = []
        cross_val_scores = []
        for name in class_names:
            temp_df = test_df[test_df['Classes'] == name]
            X_test = list(temp_df['fragment'])
            X = np.zeros(shape = (len(X_test), 100))
            Y_test = label_encoder_obj.transform(temp_df['Classes'])
            final_result_class.append(name)
        for i in range(len(X_test)):
            if type(X_test[i]) == str:
                temp = X_test[i].strip('][').split(', ')
            #print(temp)
            #print(type(temp))
            for j in range(len(temp)):
                X[i][j] = temp[j]
        predictions = model.predict(X)
        Y_pred =[]
        for k in range(len(predictions)):
            temp_pred = predictions[k]
            max_value = max(temp_pred)
            result = np.where(temp_pred == max_value)[0][0]
            Y_pred.append(result)
        print(Y_test)
        print(Y_pred)
        precision_values.append(precision_score(Y_test, Y_pred, average = 'macro'))
        recall_values.append(recall_score(Y_test, Y_pred, average = 'macro'))
        f1_values.append(f1_score(Y_test, Y_pred, average = 'macro'))
        accuracy_values.append(accuracy_score(Y_test, Y_pred))
        # added cross validation
        cross_val_scores.append(cross_val_score(Y_test, Y_pred))
        #print(X_test)
        results_df = pd.DataFrame()
        results_df['Class Name'] = final_result_class
        results_df['Precision'] = precision_values
        results_df['Recall'] = recall_values
        results_df['F1 Score'] = f1_values
        results_df['Accuracy'] = accuracy_values
        results_df['Cross Val Score'] = cross_val_scores
        results_df.to_excel('RESULTS_FINAL_100_2.xlsx')