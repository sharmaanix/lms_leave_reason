import os


import re
import nltk
import pickle
import contractions
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem  import WordNetLemmatizer

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('brown')

# print('brfore')
filename = 'pickle_model.pkl'
with open(filename, 'rb') as file:  
    model = pickle.load(file)

# print('after')
lemmatizer = WordNetLemmatizer()
stop_word = list(set(stopwords.words('english')))
stop_word = list(filter(lambda x : x!=str('not'),stop_word))

def replace_word(array):
  '''To replace the most common roman word to english word'''
  
  text = " ".join(array)
  text = text.replace('bandha', 'strike')
  text = text.replace('banda','strike')
  text = text.replace('bandh', 'strike')
  
  return text.split()


def noun_phrase(token):
  '''To extract important grammatical token from text'''

  key = pos_tag(token)
  data=[]
  for tag in key:
    if tag[1] == 'NN':
      data.append(tag[0])
  
  if len(data) == 0:
    return token
  
  return list(data)


def text_processing(text):
    
    #removing everything except letter and space
    temp = re.sub(r"[^A-Za-z ]+",r"",text)
    #lower case
    temp = temp.lower()
    #remove unnecessary spaces
    temp = " ".join(temp.split())
    #word tokenize
    temp = word_tokenize(temp)
    # remove stop words
    temp = [ item for item in temp if item not in stop_word]
    #lemmatize
    temp = [lemmatizer.lemmatize(item) for item in temp]
    
    return temp



def find_token(text):
    '''
    This function convert text, i.e string, into token
    after doing all required pre-processing

    argument: text -> input string to find token
    returns the token

    input: text of type string
    output: strings in list
    
    '''
    preprocess_text = text_processing(text)
    noun_phrase_token = noun_phrase(preprocess_text)
    replace_roman_word = replace_word(noun_phrase_token)

    return replace_roman_word



def find_class(text):
    ''''
    This function return the classes on which the WFH reason
    belongs to 

    argument: text -> wfh reason 
    returns the classes on which wfh reason belongs

    input: text of type string
    output: text of type string
    '''

    assert isinstance(text,str)
    tokens = find_token(text)

    if len(tokens) == 0:
        return 'empty'
    
    list_group = ['unwell','home','strike','weather','travel','emergency','personal']
    similar = []

    for compare in list_group:
        compare_with_all = []
        
        for token in tokens:
            try:
                similarity = model.similarity(compare,token)
                compare_with_all.append(similarity)
            except:
                compare_with_all.append(-1)
        
        similar.append(max(compare_with_all))

    if max(similar) > 0:
        max_index = similar.index(max(similar))
        return list_group[max_index]
    
    return 'other'

print(find_class('feeling unwell today'))

