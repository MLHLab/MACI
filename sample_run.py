import pandas as pd
import numpy as np
import MACI_FINAL

starting_data = pd.read_csv('Final_Starting_data.csv')
sequence_data = starting_data['Sequence']
drug_class_data = starting_data['Drug Class']

unique_classes = np.unique(drug_class_data)

selected_classes = []

obj = MACI_FINAL.MACI()
segmented_data_set,classes_segmented = obj.create_dataset(starting_data, unique_classes)

fragmented_dataframe = pd.DataFrame()
fragmented_dataframe['fragment'] = segmented_data_set
fragmented_dataframe['Classes'] = classes_segmented



#choose classes with size over 30000
selected_classes = []
for name in unique_classes:
    temp = fragmented_dataframe[fragmented_dataframe['Classes'] == name]
    if len(temp) >= 30000:
        selected_classes.append(name)




final_df = fragmented_dataframe[fragmented_dataframe['Classes'].isin(selected_classes)]
print(final_df.head(5))
train_df = final_df.groupby('Classes').apply(lambda x: x.sample(n=5000, replace = True)).reset_index(drop = True)


#encode each fragment
sample_sequences = train_df['fragment']
sample_classes = train_df['Classes']
sample_selected_x = []
sample_selected_y = []
for (sample,sample_class) in zip(sample_sequences,sample_classes):
    enocded_fragment = obj.encoder(sample)
    #not selecting ones that may have issues
    if len(enocded_fragment) == 100:
        sample_selected_x.append(enocded_fragment)
        sample_selected_y.append(sample_class)


train_y = np.zeros(shape = (len(sample_selected_y), len(selected_classes)))
le = obj.create_encoder(selected_classes)
quantified_y = le.transform(sample_selected_y)
for i in range(len(sample_selected_y)):
    train_y[i][quantified_y[i]] = 1

#model creation
obj.model_create(np.array(sample_selected_x), train_y, len(selected_classes))


