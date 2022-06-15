# Code to post-process T5 output (to something cleaner)

import pandas as pd

input_path = '/scratch/guchoadeassis/groupw/outputs/predictions_du.csv'
output_path = '/scratch/guchoadeassis/groupw/outputs/predictions_du_clean.csv'

df = pd.read_csv(input_path)
new_df = pd.DataFrame()

generated_clean, source_column, target_column = [], [], []
punctuation =  ["'",'!','(',')','-','[',']','{','}',';',':','"',',','<','>','.','/','?','@','#','$','%','^','&','*','_','~']

def post_process(source, generated):
    new_words = []
    # Add capital letters
    punc_dict = punctuate_dict(generated)
    for word in source.split(' '):
        new_word = word
        if word in punc_dict:
            new_word = punc_dict[word]
        new_words.append(new_word)   
    new_sentence = ' '.join(new_words)
    return new_sentence

def punctuate_dict(sentence):
    punc_dict = {}
    for word in sentence.split(' '):
        if word not in punc_dict:
            word_clean = ''.join(c for c in word.lower() if c not in punctuation)
            punc_dict[word_clean] = word
    return punc_dict

def isNaN(num):
    return num != num

for i, row in df.iterrows():
    source, generated, target = row['Source Text'], row['Generated Text'], row['Actual Text']
    if not isNaN(source) and not isNaN(generated):
        source_column.append(source)
        target_column.append(target)
        generated_clean.append(post_process(row['Source Text'], row['Generated Text']))

new_df['Clean Generated Text'] = generated_clean
new_df['Source Text'] = source_column
new_df['Actual Text'] = target_column
new_df.to_csv(output_path)
print("Finished Task")
