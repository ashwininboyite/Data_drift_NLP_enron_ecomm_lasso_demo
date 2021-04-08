import pandas as pd
from scipy.stats import ks_2samp
import string 

#modelop.init
def begin():
    global train, numerical_features
    train = pd.read_csv('training_data.csv')
    numerical_features = train.select_dtypes(['int64', 'float64']).columns
    pass

#modelop.score
def action(datum):
    yield datum

#modelop.metrics
def metrics(data):
    clean_test_data = clean_data_preprocessing(len(data), data)
    clean_train_data = clean_data_preprocessing(len(train), train )
    ks_pvalues = ks_test(clean_test_data, clean_train_data)

    yield ks_pvalues

def text_lowercase(text): 
    return text.lower() 

def remove_punctuation(text): 
    translator = str.maketrans('', '', string.punctuation) 
    return text.translate(translator) 

def clean_data_preprocessing(data_set_length, original_data):
  original_data = original_data.dropna()
  original_data_copy = original_data.copy()
  count = 0
  for value in original_data.content:
    text_lowercase_text = text_lowercase(value)
    remove_punctuation_text  = remove_punctuation(text_lowercase_text)
    count = count+1
    original_data_copy.at[count,'content']= remove_punctuation_text
  return original_data_copy

def  ks_test(clean_test_data, clean_train_data):
    ks_tests = [ks_2samp(clean_train_data.loc[:, feat], clean_test_data.loc[:, feat]) for feat in numerical_features]
    pvalues = [x[1] for x in ks_tests]
    list_of_pval = [f"{feat}_p-value" for feat in numerical_features]
    ks_pvalues = dict(zip(list_of_pval, pvalues))
    return ks_pvalues