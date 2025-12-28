import nltk
import re
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer


def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


def preprocessing_function(text: str) -> str:
    preprocessed_text = remove_stopwords(text)
    
    # TO-DO 0: Other preprocessing function attemption
    # Begin your code 
    # preprocessed_text = remove_stopwords(text)
    # preprocessed_text = preprocessed_text.lower()                        #lower case
    # preprocessed_text = re.sub('<br />>','',preprocessed_text)           #html endl
    # preprocessed_text = re.sub(' +', ' ', preprocessed_text)             #consecutive blank -> a blank
    # preprocessed_text = re.sub('(?<=\w)([.])', r' \1', preprocessed_text)#special character
    # preprocessed_text = re.sub('\d', '', preprocessed_text)              #number
    return preprocessed_text







    # End your code
