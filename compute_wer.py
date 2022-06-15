# Code to compute WER + SER

import worderrorrate
import pandas as pd

punctuation = ["'",'!','(',')','-','[',']','{','}',';',':','"',',','<','>','.','/','?','@','#','$','%','^','&','*','_','~']

input_path = '/scratch/guchoadeassis/groupw/outputs/predictions_en_clean.csv'

df = pd.read_csv(input_path)

# Remove punctuation
def remove_punctuation(sentence):
    return ''.join([c for c in sentence if c not in punctuation])

# Remove capitalization
def remove_capitalization(sentence):
    return sentence.lower()

# Word Error Rate
def compute_wer(df, ref_name, hyp_name, preprocess=lambda x: x):
    num, den = 0, 0
    for _, row in df.iterrows():
        wer = worderrorrate.WER(preprocess(row[ref_name]).split(' '), preprocess(row[hyp_name]).split(' '))
        num += wer.nerr
        den += len(wer.ref)
    return num/den

# Slot Error Rate
def compute_ser(df, ref_name, hyp_name):
    C, I, D, S = 0, 0, 0, 0
    for _, row in df.iterrows():
        ref_row = row[ref_name]
        hyp_row = row[hyp_name]
        ref = ref_row + max(0, len(hyp_row)-len(ref_row)) * ' '
        hyp = hyp_row + max(0, len(ref_row)-len(hyp_row)) * ' '
        for i in range(len(ref)):
            ref_char = ref[i]
            hyp_char = hyp[i]
            if hyp_char in punctuation:
                if ref_char == hyp_char:
                    C += 1
                elif ref_char in punctuation:
                    S += 1
                else:
                    I += 1
            elif ref_char in punctuation:
                D += 1
    print(C, I, D, S)
    return (I + D + S)/(C + D + S)

wer_generated = compute_wer(df, 'Actual Text', 'Clean Generated Text')
wer_generated_nopunc = compute_wer(df, 'Actual Text', 'Clean Generated Text', remove_punctuation)
wer_generated_nocap = compute_wer(df, 'Actual Text', 'Clean Generated Text', remove_capitalization)
wer_source = compute_wer(df, 'Actual Text', 'Source Text')
wer_source_nopunc = compute_wer(df, 'Actual Text', 'Source Text', remove_punctuation)
wer_source_nocap = compute_wer(df, 'Actual Text', 'Source Text', remove_capitalization)
ser_generated = compute_ser(df, 'Actual Text', 'Clean Generated Text')
ser_source = compute_ser(df, 'Actual Text', 'Source Text')

print(f'Word Error Rate - Generated Text = {wer_generated}')
print(f'Word Error Rate - Generated Text (punc) = {wer_generated_nocap}')
print(f'Word Error Rate - Generated Text (cap) = {wer_generated_nopunc}')
print(f'Word Error Rate - Source Text = {wer_source}')
print(f'Word Error Rate - Source Text (punc) = {wer_source_nocap}')
print(f'Word Error Rate - Source Text (cap) = {wer_source_nopunc}')
print(f'Slot Error Rate - Generated Text = {ser_generated}')
print(f'Slot Error Rate - Source Text = {ser_source}')
