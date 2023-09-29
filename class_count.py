import pandas as pd
import numpy as np

aro_data = pd.read_excel('ARO_selected.xlsx', names = ['Selected ARO'])
display(aro_data.head(5))

file1 = open('arg_june_23 (1).fasta', 'r')
Lines = file1.readlines()


selected_aro = np.array(aro_data['Selected ARO'])
# Strips the newline character
count = 0
aro = []
aro_number = []
sequence = []
selected_aro_flag = 0
for line in Lines:
    temp = line.strip()
    if count%2 == 0:
        parts = temp.split('|')
        #print(parts)
        temp_ARO = parts[1]
        temp_ARO_num = parts[4]
        #check if ARO is in selected ARO file if yes then flag else 0
        if temp_ARO in selected_aro:
            selected_aro_flag = 1
            aro.append(temp_ARO)
            aro_number.append(temp_ARO_num)
        else:
            selected_aro_flag = 0
    #print(temp_ARO)
    #print(selected_aro_flag)
    else:
        # if flagged then add sequence
        if selected_aro_flag == 1:
            sequence.append(temp)
    count += 1

print(len(aro))
print(len(sequence))
print(len(aro_number))

aro_class_name = pd.read_excel('ARO_all.xlsx')
display(aro_class_name.head(5))

final_sequence_data_frame = pd.DataFrame()
final_sequence_data_frame['ARO Accession'] = aro_number
final_sequence_data_frame['Sequence'] = sequence

display(final_sequence_data_frame.head(5))

df = pd.merge(final_sequence_data_frame, aro_class_name, on='ARO Accession', how='inner')

display(df.head(10))

df.to_csv('Final_Starting_data.csv')

"""### Second Part"""

df = pd.read_csv('Final_Starting_data.csv')
columns = ['ARO Accession', 'Sequence', 'Drug Class']
df = df[columns]
display(df.head(5))

classes = np.array(df['Drug Class'])
class_selected = []
class_frag_count = []
class_frag_count_with_repeat = []
count = 0
for element in classes:
    temp = df[df['Drug Class'] == element]
    class_selected.append(element)
    temp_sequences = np.array(temp['Sequence'])
    sub_strings = []
    for seq in temp_sequences:
        for i in range(100, len(seq)):
            ss = seq[i-100:i]
            sub_strings.append(ss)
    unique_frags_with_repeat = len(sub_strings)
    unique_frags = len(np.unique(np.array(sub_strings)))
    print(unique_frags)
    print('count %d'%(count))
    count += 1
    class_frag_count.append(unique_frags)
    class_frag_count_with_repeat.append(unique_frags_with_repeat)

df_final_counts = pd.DataFrame()
df_final_counts['Class Name'] = class_selected
df_final_counts['Unique Frags']  = class_frag_count

display(df_final_counts.head(10))

df_final_counts.to_csv('Final_class_based_count.csv')

df_final_counts_repeat = pd.DataFrame()
df_final_counts_repeat['Class Name'] = class_selected
df_final_counts_repeat['Unique Frags']  = class_frag_count_with_repeat

display(df_final_counts_repeat.head(10))

df_final_counts_repeat.to_csv('Final_class_based_count_repeat.csv')

