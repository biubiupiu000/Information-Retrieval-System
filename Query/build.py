import argparse
import os
import datetime
import collections
import time
from datetime import datetime
import pickle
from nltk.stem.porter import PorterStemmer
from index_variable import *
import shutil
import numpy as np

def main():
    start_time=time.time()
    doc_id = collections.defaultdict(int)

    single_word_dict = collections.defaultdict(int)
    single_id=collections.defaultdict(int)

    phrase_word_dict = collections.defaultdict(int)
    phrase_id = collections.defaultdict(int)

    position_word_dict = collections.defaultdict(int)
    position_id = collections.defaultdict(int)

    stem_word_dict = collections.defaultdict(int)
    stem_id = collections.defaultdict(int)

    parser = argparse.ArgumentParser(description='index_generation')
    parser.add_argument('file', type=str, help='You should enter accurate document address')
    parser.add_argument('stop_file', type=str, help='You should enter accurate stopwords address')
    parser.add_argument('index_mode',type=str,help='It only has four modes(single, phrase, position and stemming)')
    args = parser.parse_args()
    stop_word_list=read_stop_list(args.stop_file)
    files=os.listdir(args.file)

    #generate stop_words
    # stop_word_list=read_stop_list('stops.txt')
    # files = os.listdir(r'BigSample')
    if args.index_mode=='single':
        single_index_generation(args.file,files,single_id,doc_id,stop_word_list,start_time) # generate single word_index
        doc_sta(args.file,args.index_mode)
        print(time.time() - start_time)
    elif args.index_mode=='phrase':
        phrase_index_generation(args.file,files,phrase_id, doc_id, stop_word_list,start_time)
        doc_sta(args.file,args.index_mode)
        print(time.time() - start_time)
    elif args.index_mode=='position':
        position_index_generation(args.file,files, position_id, doc_id, start_time)
        doc_sta(args.file,args.index_mode)
        print(time.time() - start_time)
    elif args.index_mode=='stem':
        stem_index_generation(args.file,files,  stem_id, doc_id, stop_word_list, start_time)
        doc_sta(args.file,args.index_mode)
        print(time.time() - start_time)
    else:
        print('You should enter correct mode')


def read_stop_list(stop_file):
    stop_word_list=[]
    with open(stop_file, 'r') as stop:
        for i in stop:
            stop_word_list.append(i.strip())
    stop.close()
    return stop_word_list

#use generator to load lines
def generate_file_lines(file):
    for line in file:
        yield line

#common line cleaning
def common_line_deal(text):
    # sub escape symbols
    text = re.sub(r'&hyph;', '-', text)
    text = re.sub(r'&blank;', ' ', text)
    text = re.sub(r'&sect;', '§', text)
    text = re.sub(r'&times', '×', text)

    # apart words
    text = re.sub(r'/', ' ', text)
    text = re.sub(r';', ' ', text)
    text = re.sub(r'_', ' ', text)
    text = re.sub(r'[=+&]', ' ', text)

    # sub title
    text = re.sub(r'</?\w+>', '', text)
    # deal with phone number
    text = re.sub(r'\(\d+\) \d+-\d+', lambda x: re.sub(r'[\(\)]', '', x.group().split()[0]) + x.group().split()[1],
                  text)
    #weird currency
    text = re.sub(sp_regex2, lambda x: ''.join(x.group().split()), text)
    return text

#email ip url
def special_check(text,regex_rule,dict,docid):

    if re.search(regex_rule,text):
        special_words=re.search(regex_rule,text)
        word=special_words.group()
        add_in_dict(word,dict,docid)



#add word in temp file
def add_in_dict(word,dict,docid):
    if not word in dict:
        dict[word] = {docid: 1}
    else:
        if docid in dict[word]:
            dict[word][docid] += 1
        else:
            dict[word][docid] = 1
def position_in_dict(word,dict,docid,pos):
    if not word in dict:
        dict[word] = {docid: [pos]}
    else:
        if docid in dict[word]:
            dict[word][docid].append(pos)
        else:
            dict[word][docid] = [pos]

#make sure date is valid
def isvaliddate(date,pattern):
    try:
        datetime.strptime(date,pattern)
        return True
    except:
        return False

#make sure two-phrase is valid
def isbi_valid(word1,word2,stop_word):
    if any([j in stop_word for j in [word1,word2]]):
        return False
    if any([re.search(r'[@#\.,^:\[\]<>()×*§&]',word1),re.search(r'[@#\.,^:\[\]<>()×*§&]',word2)]):
        return False

    return True
#make sure three-phrase is valid
def istri_valid(word1,word2,word3,stop_word):
    if any([j in stop_word for j in [word1, word2, word3]]):
        return False
    # remove()[]
    if any([re.search(r'[@#\.,^:\[\]<>()×*§&]', word1), re.search(r'[@#\.,^:\[\]<>()×*§&]', word2),
            re.search(r'[@#\.,^:\[\]<>()×*§&]', word3)]):
        return False
    return True

#deal with date
#deal with date
def date_deal(text):
    if re.search(r'\b\d{1,3}[-/]\d{1,2}[-/]\d{2,4}', text):
        date = re.findall(r'\b\d{1,3}[-/]\d{1,2}[-/]\d{2,4}', text)
        for i in date:
            alpha = i.split('-')[-1]
            if len(alpha) == 4 and isvaliddate(i, '%m-%d-%Y'):
                continue
            else:
                if isvaliddate(i, '%m-%d-%y'):
                    t=datetime.strptime(i, '%m-%d-%y').strftime('%m-%d-%Y')

                    text=re.sub(i,t,text)

    if a.search(text):
        alpha = a.finditer(text)
        for i in alpha:
            b = i.group().split()
            c = date_map[b[0]] + '-' + b[1][:-1] + '-' + b[-1]
            if isvaliddate(c, '%m-%d-%Y'):
                final = datetime.strptime(c, '%m-%d-%Y').strftime('%m-%d-%Y')
                text=re.sub(i.group(),final,text)
    return text
#deal with digit
def digit_deal(text):
    text = re.sub(sp_regex9, lambda x: ''.join(x.group().split(sep=',')), text)
    text = re.sub(r'\$?\d+\.00', lambda x: x.group().split(sep='.')[0], text)
    return text

# handle common word deals
def word_tokenize(word,stop_word):
    stop_word_list=stop_word
    word = word.casefold()
    # extra signal
    if re.match('^[`\'.,%(\[_=\-$§]+.*', word):
        word = re.sub('^[`\'.,%(\[_=\-$§]+', '', word)
    if re.match('.*[.,;\'):?\]_\-<>.]+$', word):
        word = re.sub('[.,;\'):\]?_\-<>.]+$', '', word)
    # Abbreviations
    if re.fullmatch(sp_regex1, word):
        alpha = word.split(sep='.')[-1]
        if alpha in file_ext_list:
            word = re.sub(sp_regex1, lambda x: ''.join(x.group().split(sep='.')), word)
            # add_in_dict(alpha, dict, docid)
            words=[alpha,word]
            return words
        word = re.sub(sp_regex1, lambda x: ''.join(x.group().split(sep='.')), word)

    # digit-alpha
    if re.fullmatch(sp_regex4, word):
        alpha = word.split(sep='-')[1]

        if len(alpha) >= 3 and alpha not in stop_word_list:
            word = ''.join(word.split(sep='-'))
            # add_in_dict(alpha, dict, docid)
            words=[alpha,word]

            return words
        word = ''.join(word.split(sep='-'))
    # alpha-digit
    if re.fullmatch(sp_regex3, word):
        alpha = word.split(sep='-')[0]
        if len(alpha) >= 3 and alpha not in stop_word_list:
            word = ''.join(word.split(sep='-'))
            words=[alpha,word]

            return words
        word = ''.join(word.split(sep='-'))
    # hyphen
    if re.match(sp_regex5, word):
        words = word.split(sep='-')
        Word = ''.join(words)
        words.append(Word)
        for h_word in words:
            if h_word  in stop_word_list or h_word in pre_list:
                # add_in_dict(h_word, dict, docid)
                words.remove(h_word)

        return words
    # remove()[]
    word = re.sub(r'[\[\]()<>#*^&,]', '', word)
    return word
def word_tokenize_phrase(word):
    word = word.casefold().strip()
    # extra signal
    if re.match('^[`\'.,%(\[_=\-$§]+.*', word):
        word = re.sub('^[`\'.,%(\[_=\-$§]+', '', word)
    if re.match('.*[.,;\'):?\]_\-<>.]+$', word):
        word = re.sub('[.,;\'):\]?_\-<>.]+$', '', word)
    word = re.sub(r'[\[\]()<>#*^&,]', '', word)
    return word
def word_tokenize_common(word):
    word = word.casefold()
    # extra signal
    if re.match('^[`\'.,%(\[_=\-$§]+.*', word):
        word = re.sub('^[`\'.,%(\[_=\-$§]+', '', word)
    if re.match('.*[.,;\'):?\]_\-<>.]+$', word):
        word = re.sub('[.,;\'):\]?_\-<>.]+$', '', word)
    # remove()[] etc.
    word = re.sub(r'[\[\]()<>#*^&]', '', word)
    return word



#Doc statistic
def doc_sta(file,mode):
    with open(file+'/'+mode+'/'+mode+'_lexicon.pkl', 'rb') as f:
        d = pickle.load(f)
        df=[i[2] for i in d]
        print('max df:', np.max(df))
        print('min df:', np.min(df))
        print('mean df:', np.mean(df))
        print('median df:', np.median(df))


def single_index_generation(file,files, dict2, dict3, stop_word,initial_time):
    docno = 0

     #temporary dictionary to collect info.
    #assert whether have temp file to store temporary files
    try:
        os.makedirs(file+'/single')
    except:
        print('Have output files')
    for file_name in files:
        if re.search(r'\.',file_name):
            with open(file +'/'+ file_name, 'rt') as f:
                for line in (generate_file_lines(f)):

                    #common deals
                    if not re.match(regex1, line) and not re.match(regex2, line) and not line.isspace() and not re.match(
                            '<PARENT>', line):
                        if re.match('<DOCNO>', line):
                            docno+=1
                            DocID = line.split()[1]
                            dict3[DocID]=docno #store docno

                        else:
                            line = common_line_deal(line)
                            line=date_deal(line)
                            special_check(line, sp_regex6,dict2, docno)
                            special_check(line, sp_regex7, dict2, docno)
                            special_check(line, sp_regex8, dict2 ,docno)
                            line = digit_deal(line)

                            stop_word_list=stop_word
                            word_list=line.split() #generate word tokens
                            for w in word_list:
                                words = word_tokenize(w, stop_word_list)
                                if isinstance(words, list):
                                    for word in words:
                                        if word in stop_word_list or word in r_list or word.isspace():
                                            continue
                                        add_in_dict(word, dict2, docno)
                                else:
                                    if words in stop_word_list or words in r_list or words.isspace():
                                        continue
                                    add_in_dict(words, dict2, docno)

            print('at present Docno:',docno)
            print('at present time:', time.time()-initial_time)
            f.close()
        else:
            continue
    print('the time after generating:',time.time()-initial_time)
    #generate posting list
    nums=[len(i) for i in dict2.values()]
    c=sum(nums)
    print('totally posting lists:', c)

    posting_list=collections.defaultdict(int)
    lexicon=collections.defaultdict(int)
    wordno=0
    for i,j in dict2.items():
        lexicon[i]=wordno
        posting_list[wordno]=j
        wordno+=1
    with open(file + '/single/single.pkl', 'wb') as fl:
        pickle.dump(posting_list, fl)
    fl.close()
    #generate lexicon file
    Lexicon=[(i,lexicon[i],len(dict2[i])) for i in dict2.keys()]
    print('totally lexicon lists:', len(Lexicon))
    with open(file+'/single/single_lexicon.pkl','wb') as fl:
        pickle.dump(Lexicon,fl)
    fl.close()
    with open(file + '/single/doc_map.pkl', 'wb') as fl:
        pickle.dump(dict3,fl)
    fl.close()
def phrase_index_generation(file,files,dict2,dict3,stop_word,initial_time):
    docno=0
    # assert whether have temp file to store temporary files
    try:
        os.makedirs(file+'/phrase')
    except:
        print('Have output files')

    for file_name in files:
        if re.search(r'\.',file_name):
            with open(file +'/'+ file_name, 'rt') as f:
                for line in (generate_file_lines(f)):
                    if not re.match(regex1, line) and not re.match(regex2, line) and not line.isspace() and not re.match(
                        '<PARENT>', line):
                        if re.match('<DOCNO>', line):
                            docno += 1
                            DocID = line.split()[1]
                            dict3[DocID] = docno # store docno
                        else:
                            line = common_line_deal(line)
                            line = digit_deal(line)
                            stop_word_list=stop_word
                            word_list=line.split()
                            #handle two-phrase
                            if len(word_list)>=2:
                                for index in range(len(word_list)-1):
                                    word_1,word_2=word_tokenize_phrase(word_list[index]),word_tokenize_phrase(word_list[index+1])
                                    if isbi_valid(word_1,word_2,stop_word_list):
                                        term=word_1+' '+word_2
                                        add_in_dict(term,dict2,docno)

                        #handle three-phrase
                            if len(word_list)>=3:
                                for index in range(len(word_list)-2):
                                    word_1, word_2 = word_tokenize_phrase(word_list[index]),word_tokenize_phrase(word_list[index + 1])
                                    word_3=word_tokenize_phrase(word_list[index+2])
                                    if istri_valid(word_1,word_2,word_3,stop_word_list):
                                        term=word_1 + ' ' + word_2+ ' ' + word_3
                                        add_in_dict(term,dict2, docno)

            print('at present Docno:', docno)
            print('at present time:', time.time() - initial_time)
        else:
            continue
    print('the time after generating:',time.time()-initial_time)
    posting_list=collections.defaultdict(int)
    lexicon=collections.defaultdict(int)
    wordno=0
    for i,j in dict2.items():
        lexicon[i]=wordno
        wordno+=1

    Lexicon = [(i, lexicon[i],len(dict2[i])) for i in dict2.keys() if sum(dict2[i].values())>=5 ]
    print('totally lexicon:',len(Lexicon))
    terms=[(i[0],i[1]) for i in Lexicon]
    for term in terms:
        posting_list[term[1]]=dict2[term[0]]
    nums = [len(i) for i in posting_list.values()]
    c = sum(nums)
    print('totally posting lists:', c)
    with open(file+'/phrase/phrase.pkl', 'wb') as f:
        pickle.dump(posting_list, f)
    f.close()
    with open(file+'/phrase/phrase_lexicon.pkl', 'wb') as f:
        pickle.dump(Lexicon, f)
    f.close()
    with open(file + '/phrase/doc_map.pkl', 'wb') as fl:
        pickle.dump(dict3,fl)
    fl.close()
def position_index_generation(file,files,dict2,dict3,initial_time):

    docno=0
    wordno = 0
    # assert whether have temp file to store temporary files
    try:
        os.makedirs(file+'/position')
    except:
        print('Have output files')
    for file_name in files:
        if re.search(r'\.',file_name):
            with open(file +'/'+ file_name, 'rt') as f:
                for line in (generate_file_lines(f)):
                    # determine wheter it is up to limit
                    if not re.match(regex1, line) and not re.match(regex2, line) and not line.isspace() and not re.match(
                            '<PARENT>', line):
                        if re.match('<DOCNO>', line):
                            docno+=1
                            DocID = line.split()[1]
                            dict3[DocID] = docno #store docno

                            word_pos = -1 # use word_pos to store position info.
                        else:
                            line = common_line_deal(line)
                            line = digit_deal(line)
                            word_list=line.split()
                            for w in word_list:
                                word_pos+=1
                                word=word_tokenize_common(w)
                                if word in r_list or word.isspace(): #include stopwords
                                    continue

                                position_in_dict(word, dict2, docno, word_pos)
        else:
            continue
        print('at present Docno:',docno)
        print('at present time:', time.time()-initial_time)
    print('the time after generating:',time.time()-initial_time)
    nums=[len(i) for i in dict2.values()]
    c=sum(nums)
    print('totally posting lists:', c)

    posting_list=collections.defaultdict(int)
    lexicon=collections.defaultdict(int)
    wordno=0
    for i,j in dict2.items():
        lexicon[i]=wordno
        posting_list[wordno]=j
        wordno+=1
    Lexicon = [(i, lexicon[i], len(dict2[i])) for i in dict2.keys()]
    print('totally lexicon lists:', len(Lexicon))
    with open(file+'/position/position.pkl', 'wb') as f:
        pickle.dump(posting_list, f)
    f.close()
    with open(file+'/position/position_lexicon.pkl', 'wb') as f:
        pickle.dump(Lexicon, f)
    f.close()
    with open(file + '/position/doc_map.pkl', 'wb') as fl:
        pickle.dump(dict3,fl)
    fl.close()
def stem_index_generation(file,files,dict2,dict3,stop_word,initial_time):
    porter_stemmer = PorterStemmer()
    docno = 0
    try:
        os.makedirs(file+'/stem')
    except:
        print('Have output files')
    for file_name in files:
        if re.search(r'\.',file_name):
            with open(file +'/'+ file_name, 'rt') as f:
                for line in (generate_file_lines(f)):
                    # determine wheter it is up to limit
                    # common deals
                    if not re.match(regex1, line) and not re.match(regex2, line) and not line.isspace() and not re.match(
                            '<PARENT>', line):
                        if re.match('<DOCNO>', line):
                            docno+=1
                            DocID = line.split()[1]
                            dict3[DocID] = docno #store docno
                        else:
                            line = common_line_deal(line)
                            line = digit_deal(line)

                            stop_word_list=stop_word
                            word_list=line.split()
                            for w in word_list:
                                word=word_tokenize_common(w)
                                if word in stop_word_list or word in r_list or word.isspace():
                                    continue
                                word=porter_stemmer.stem(word) #add stemmer tool

                                add_in_dict(word, dict2, docno)
            print('at present Docno:', docno)
            print('at present time:', time.time() - initial_time)
        else:
            continue
    # ensure the last temporary file

    print('the time after generating:',time.time()-initial_time)
    # generate posting list
    nums = [len(i) for i in dict2.values()]
    c = sum(nums)
    print('totally posting lists:', c)
    posting_list = collections.defaultdict(int)
    lexicon = collections.defaultdict(int)
    wordno = 0
    for i, j in dict2.items():
        lexicon[i] = wordno
        posting_list[wordno] = j
        wordno += 1

    Lexicon = [(i, lexicon[i], len(dict2[i])) for i in dict2.keys()]
    print('totally lexicon lists:', len(Lexicon))
    with open(file+'/stem/stem.pkl', 'wb') as f:
        pickle.dump(posting_list, f)
    f.close()
    with open(file+'/stem/stem_lexicon.pkl', 'wb') as f:
        pickle.dump(Lexicon, f)
    f.close()
    with open(file + '/stem/doc_map.pkl', 'wb') as fl:
        pickle.dump(dict3,fl)
    fl.close()


if __name__ == '__main__':
    main()