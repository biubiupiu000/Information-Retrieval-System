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
    parser.add_argument('memory_limit',type=int)
    args = parser.parse_args()
    stop_word_list=read_stop_list(args.stop_file)
    files=os.listdir(args.file)
    memory_limit=args.memory_limit
    #generate stop_words
    # stop_word_list=read_stop_list('stops.txt')
    # files = os.listdir(r'BigSample')
    if args.index_mode=='single':
        single_index_generation(args.file,files,single_id,doc_id,stop_word_list,start_time,memory_limit) # generate single word_index
        doc_sta(args.index_mode)
        print(time.time() - start_time)
    elif args.index_mode=='phrase':
        phrase_index_generation(args.file,files,phrase_id, doc_id, stop_word_list,start_time,memory_limit)
        doc_sta(args.index_mode)
        print(time.time() - start_time)
    elif args.index_mode=='position':
        position_index_generation(args.file,files, position_id, doc_id, start_time,memory_limit)
        doc_sta(args.index_mode)
        print(time.time() - start_time)
    elif args.index_mode=='stemming':
        stem_index_generation(args.file,files,  stem_id, doc_id, stop_word_list, start_time,memory_limit)
        doc_sta(args.index_mode)
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
    text = re.sub(r'&sect;', '??', text)
    text = re.sub(r'&times', '??', text)

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
def special_check(text,regex_rule,dict,dict2,docid,no):
    num=no
    if re.search(regex_rule,text):
        special_words=re.search(regex_rule,text)
        word=special_words.group()
        if word not in dict2:
            dict2[word] = num
            num += 1
        add_in_dict(word,dict,docid)

    return num

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
    if any([re.search(r'[@#\.,^:\[\]<>()??*??&]',word1),re.search(r'[@#\.,^:\[\]<>()??*??&]',word2)]):
        return False

    return True
#make sure three-phrase is valid
def istri_valid(word1,word2,word3,stop_word):
    if any([j in stop_word for j in [word1, word2, word3]]):
        return False
    # remove()[]
    if any([re.search(r'[@#\.,^:\[\]<>()??*??&]', word1), re.search(r'[@#\.,^:\[\]<>()??*??&]', word2),
            re.search(r'[@#\.,^:\[\]<>()??*??&]', word3)]):
        return False
    return True

#deal with date
#deal with date
def date_deal(text,dict,docid):
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
    if re.match('^[`\'.,%(\[_=\-$??]+.*', word):
        word = re.sub('^[`\'.,%(\[_=\-$??]+', '', word)
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
    if re.match('^[`\'.,%(\[_=\-$??]+.*', word):
        word = re.sub('^[`\'.,%(\[_=\-$??]+', '', word)
    if re.match('.*[.,;\'):?\]_\-<>.]+$', word):
        word = re.sub('[.,;\'):\]?_\-<>.]+$', '', word)
    word = re.sub(r'[\[\]()<>#*^&,]', '', word)
    return word
def word_tokenize_common(word):
    word = word.casefold()
    # extra signal
    if re.match('^[`\'.,%(\[_=\-$??]+.*', word):
        word = re.sub('^[`\'.,%(\[_=\-$??]+', '', word)
    if re.match('.*[.,;\'):?\]_\-<>.]+$', word):
        word = re.sub('[.,;\'):\]?_\-<>.]+$', '', word)
    # remove()[] etc.
    word = re.sub(r'[\[\]()<>#*^&]', '', word)
    return word
#merge temporary files
def merge():
    files = os.listdir(r'temp')
    length = len(files)
    count=0
    try:
        os.makedirs('merge')
    except:
        print('Have temp files')
    if len(files)>=2:
        while count < length:
            if not os.listdir('merge'):     #first case
                file1 = open('temp/' + files.pop(), 'r')
                file2 = open('temp/' + files.pop(), 'r')
                file3 = open('merge/temp2.txt', 'a')
                temp = 0 #signal for what is left
                a = file1.readline()
                b = file2.readline()
                c = a.split(sep=',')
                d = b.split(sep=',')
                # four possibilities
                if int(c[0]) > int(d[0]):
                    file3.write(b)
                    temp = {'a': a}
                elif int(c[0]) < int(d[0]):
                    file3.write(a)
                    temp = {'b': b}
                else:
                    if int(c[1]) > int(d[1]):
                        file3.write(b)
                        temp = {'a': a}
                    elif int(c[1]) < int(d[1]):
                        file3.write(a)
                        temp = {'b': b}
                #following cases
                while True:
                    if 'a' in temp:
                        b = file2.readline()
                        d = b.split(sep=',')
                        if b == '':
                            break
                        if int(c[0]) > int(d[0]):
                            file3.write(b)

                        elif int(c[0]) < int(d[0]):
                            file3.write(a)

                            temp = {'b': b}
                        else:
                            if int(c[1]) > int(d[1]):
                                file3.write(b)
                            elif int(c[1]) < int(d[1]):
                                file3.write(a)
                                temp = {'b': b}
                    else:
                        a = file1.readline()
                        if a == '':
                            break
                        c = a.split(sep=',')
                        if int(c[0]) > int(d[0]):
                            file3.write(b)
                            temp = {'a': a}

                        elif int(c[0]) < int(d[0]):
                            file3.write(a)

                        else:
                            if int(c[1]) > int(d[1]):
                                file3.write(b)
                                temp = {'a': a}
                            elif int(c[1]) < int(d[1]):
                                file3.write(a)
                if b == '':
                    file3.write(a)
                    while True:
                        a = file1.readline()
                        if a == '':
                            break
                        file3.write(a)
                elif a == '':
                    file3.write(b)
                    while True:
                        b = file1.readline()
                        if b == '':
                            break
                        file3.write(b)
                file1.close()
                file2.close()
                file3.close()
                count = count + 2
            else:
                file1 = open('temp/' + files.pop(), 'r')
                file2 = open('merge/' + 'temp' + str(count) + '.txt', 'r')
                count += 1
                file3 = open('merge/' + 'temp' + str(count) + '.txt', 'a+')
                temp = 0
                a = file1.readline()
                b = file2.readline()
                c = a.split(sep=',')
                d = b.split(sep=',')
                if int(c[0]) > int(d[0]):
                    file3.write(b)
                    temp = {'a': a}
                elif int(c[0]) < int(d[0]):
                    file3.write(a)
                    temp = {'b': b}
                else:
                    if int(c[1]) > int(d[1]):
                        file3.write(b)
                        temp = {'a': a}
                    elif int(c[1]) < int(d[1]):
                        file3.write(a)
                        temp = {'b': b}
                while True:
                    if 'a' in temp:
                        b = file2.readline()
                        d = b.split(sep=',')
                        if b == '':
                            break
                        if int(c[0]) > int(d[0]):
                            file3.write(b)

                        elif int(c[0]) < int(d[0]):
                            file3.write(a)

                            temp = {'b': b}
                        else:
                            if int(c[1]) > int(d[1]):
                                file3.write(b)
                            elif int(c[1]) < int(d[1]):
                                file3.write(a)
                                temp = {'b': b}
                    else:
                        a = file1.readline()
                        if a == '':
                            break
                        c = a.split(sep=',')
                        if int(c[0]) > int(d[0]):
                            file3.write(b)
                            temp = {'a': a}

                        elif int(c[0]) < int(d[0]):
                            file3.write(a)

                        else:
                            if int(c[1]) > int(d[1]):
                                file3.write(b)
                                temp = {'a': a}
                            elif int(c[1]) < int(d[1]):
                                file3.write(a)
                if b == '':
                    file3.write(a)
                    while True:
                        a = file1.readline()
                        if a == '':
                            break
                        file3.write(a)
                elif a == '':
                    file3.write(b)
                    while True:
                        b = file2.readline()
                        if b == '':
                            break
                        file3.write(b)
                file1.close()
                file2.close()
                file3.close()
                os.remove('merge/' + 'temp' + str(count - 1) + '.txt')
    else:
        count=1
    return count

#Doc statistic
def doc_sta(file):
    with open('Bigsample'+file+'/'+file+'_lexicon.pkl', 'rb') as f:
        d = pickle.load(f)
        df=[i[2] for i in d]
        print('max df:', np.max(df))
        print('min df:', np.min(df))
        print('mean df:', np.mean(df))
        print('median df:', np.median(df))


def single_index_generation(file,files, dict2, dict3, stop_word,initial_time,limit):
    docno = 0
    wordno = 0
    memory=False
    count=0
    temp_map=collections.defaultdict(int) #temporary dictionary to collect info.
    #assert whether have temp file to store temporary files
    try:
        os.makedirs('temp')
    except:
        print('Have temp files')

    try:
        os.makedirs(file+'/single')
    except:
        print('Have output files')
    for file_name in files:
        if re.search(r'\.',file_name):
            with open(file +'/'+ file_name, 'rt') as f:
                for line in (generate_file_lines(f)):
                    # determine wheter it is up to limit
                    if re.match('</DOC>',line):
                        if memory:
                            temp=sorted(temp_map.keys(),key=lambda x:dict2[x])
                            temp_file=collections.defaultdict(int)
                            for W in temp:
                                for doc,info in temp_map.get(W).items():
                                    temp_file[(dict2[W],dict3[doc])]=info

                            with open('temp/' + str(count) + '.txt', 'a') as fs:
                                # pickle.dump(temp_file,fs)
                                for triple in temp_file:
                                    fs.write(
                                        str(triple[0]) + ',' + str(triple[1]) + ',' + str(temp_file[triple]) + '\n')
                            fs.close()
                            temp_map = collections.defaultdict(int)
                            memory=False

                    #common deals
                    if not re.match(regex1, line) and not re.match(regex2, line) and not line.isspace() and not re.match(
                            '<PARENT>', line):
                        if re.match('<DOCNO>', line):
                            docno+=1
                            DocID = line.split()[1]
                            dict3[DocID]=docno #store docno

                        else:
                            line = common_line_deal(line)
                            line=date_deal(line, temp_map, DocID)
                            wordno=special_check(line, sp_regex6, temp_map,dict2, DocID,wordno)
                            wordno=special_check(line, sp_regex7, temp_map,dict2, DocID,wordno)
                            wordno=special_check(line, sp_regex8, temp_map,dict2 ,DocID,wordno)
                            line = digit_deal(line)

                            stop_word_list=stop_word
                            word_list=line.split() #generate word tokens
                            for w in word_list:
                                words = word_tokenize(w, stop_word_list)
                                if isinstance(words, list):
                                    for word in words:
                                        if word in stop_word_list or word in r_list or word.isspace():
                                            continue
                                        if word not in dict2:
                                            dict2[word] = wordno
                                            wordno += 1
                                        if word not in temp_map:
                                            count += 1
                                        else:
                                            if DocID not in temp_map[word]:
                                                count += 1
                                        if count % int(limit) == 0:
                                            memory = True
                                        add_in_dict(word, temp_map, DocID)
                                else:
                                    if words in stop_word_list or words in r_list or words.isspace():
                                        continue
                                    if words not in dict2:
                                        dict2[words] = wordno
                                        wordno += 1
                                    if words not in temp_map:
                                        count += 1
                                    else:
                                        if DocID not in temp_map[words]:
                                            count += 1
                                    if count % int(limit) == 0:
                                        memory = True
                                    add_in_dict(words, temp_map, DocID)

            print('at present Docno:',docno)
            print('at present time:', time.time()-initial_time)
            f.close()
        else:
            continue

    # ensure the last temporary file
    if temp_map:
        temp=sorted(temp_map.keys(),key=lambda x:dict2[x])
        temp_file=collections.defaultdict(int)
        for W in temp:
            for doc,info in temp_map.get(W).items():
                temp_file[(dict2[W],dict3[doc])]=info
        with open('temp/' + str(count) + '.txt', 'a') as fs:
            # pickle.dump(temp_file,fs)
            for triple in temp_file:
                fs.write(
                    str(triple[0]) + ',' + str(triple[1]) + ',' + str(temp_file[triple]) + '\n')
        fs.close()
    print('the time after generating:',time.time()-initial_time)
    #generate posting list
    num = merge()
    if num!=1: #disc-based
        if os.path.exists('BigSample/single/single.txt'):
            os.remove('merge/temp' + str(num) + '.txt')
        else:
            os.rename('merge/temp' + str(num) + '.txt', 'merge/single.txt')
            shutil.move('merge/single.txt', 'BigSample/single')
    else: #memory-based
        F=os.listdir('temp').pop()
        if os.path.exists('BigSample/single/single.txt'):
            os.remove('temp/'+F)
        else:
            os.rename('temp/'+F,'temp/single.txt')
            shutil.move('temp/single.txt', 'BigSample/single')
    shutil.rmtree('temp')
    f = open('BigSample/single/single.txt', 'r')
    print('the time after merging:',time.time()-initial_time)

    temp = collections.defaultdict(int)
    c=0
    while True:
        a = f.readline()
        i = a.split(sep=',')
        if a == '':
            break
        if int(i[0]) in temp:
            temp[int(i[0])].append(int(i[1]))
        else:
            temp[int(i[0])] = [int(i[1])]
        c+=1
    f.close()
    print('totally posting lists:', c)
    dict2={v:k for k,v in dict2.items()}
    # generate lexicon file
    Lexicon=[(dict2[i],i,len(temp[i])) for i in temp.keys()]
    print('totally lexicon lists:', len(Lexicon))
    with open(file+'/single/single_lexicon1.pkl','wb') as fl:
        pickle.dump(Lexicon,fl)
    fl.close()
    with open(file + '/single/doc_map1.pkl', 'wb') as fl:
        pickle.dump(dict3,fl)
    fl.close()
def phrase_index_generation(file,files,dict2,dict3,stop_word,initial_time,limit):
    docno = 0
    wordno = 0
    memory = False
    count = 0
    temp_map = collections.defaultdict(int)
    # assert whether have temp file to store temporary files
    try:
        os.makedirs('temp')
    except:
        print('Have temp files')
    try:
        os.makedirs(file+'/phrase')
    except:
        print('Have output files')

    for file_name in files:
        if re.search(r'\.',file_name):
            with open(file +'/'+ file_name, 'rt') as f:
                for line in (generate_file_lines(f)):
                # determine wheter it is up to limit
                    if re.match('</DOC>',line):
                        if memory:
                            temp = sorted(temp_map.keys(), key=lambda x: dict2[x])
                            temp_file = collections.defaultdict(int)
                            for W in temp:
                                for doc, info in temp_map.get(W).items():
                                    temp_file[(dict2[W], dict3[doc])] = info
                            with open('temp/' + str(count) + '.txt', 'a') as fs:
                                for triple in temp_file:
                                    fs.write(
                                        str(triple[0]) + ',' + str(triple[1]) + ',' + str(temp_file[triple]) + '\n')

                            temp_map = collections.defaultdict(int)
                            memory=False

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
                                        if term not in dict2:
                                            dict2[term] = wordno
                                            wordno += 1
                                        if term not in temp_map:
                                            count += 1
                                        else:
                                            if DocID not in temp_map[term]:
                                                count += 1
                                        if count % int(limit) == 0:
                                            memory = True
                                        add_in_dict(term,temp_map,DocID)

                        #handle three-phrase
                            if len(word_list)>=3:
                                for index in range(len(word_list)-2):
                                    word_1, word_2 = word_tokenize_phrase(word_list[index]),word_tokenize_phrase(word_list[index + 1])
                                    word_3=word_tokenize_phrase(word_list[index+2])
                                    if istri_valid(word_1,word_2,word_3,stop_word_list):
                                        term=word_1 + ' ' + word_2+ ' ' + word_3
                                        if term not in dict2:
                                            dict2[term] = wordno
                                            wordno += 1
                                        if term not in temp_map:
                                            count += 1
                                        else:
                                            if DocID not in temp_map[term]:
                                                count += 1
                                        if count % int(limit) == 0:
                                            memory = True
                                        add_in_dict(term, temp_map, DocID)

            print('at present Docno:', docno)
            print('at present time:', time.time() - initial_time)
        else:
            continue

    if temp_map:  # ensure the last temporary file
        temp = sorted(temp_map.keys(), key=lambda x: dict2[x])
        temp_file = collections.defaultdict(int)
        for W in temp:

            for doc, info in temp_map.get(W).items():
                temp_file[(dict2[W], dict3[doc])] = info

        with open('temp/' + str(count) + '.txt', 'a') as fs:
            for triple in temp_file:
                fs.write(
                    str(triple[0]) + ',' + str(triple[1]) + ',' + str(temp_file[triple]) + '\n')
    print('the time after generating:',time.time()-initial_time)
    num = merge()
    if num!=1: #disc based
        os.rename('merge/temp' + str(num) + '.txt', 'merge/phrase.txt')
    else:  #memory based
        F=os.listdir('temp').pop()
        print(F)
        os.rename('temp/'+F,'temp/phrase.txt')
        shutil.move('temp/phrase.txt','merge')
    shutil.rmtree('temp')
    f = open('merge/phrase.txt', 'r')
    print('the time after merging:',time.time()-initial_time)

    temp = collections.defaultdict(int)
    while True:
        a = f.readline()
        i = a.split(sep=',')
        if a == '':
            break
        if int(i[0]) in temp:
            temp[int(i[0])].append(int(i[2]))
        else:
            temp[int(i[0])] = [int(i[2])]
    dict2 = {v: k for k, v in dict2.items()}
    Lexicon = [(dict2[i], i,len(temp[i])) for i in temp.keys() if sum(temp[i])>=5 ]
    print('totally lexicon:',len(Lexicon))
    with open(file+'/phrase/phrase_lexicon.pkl', 'wb') as f:
        pickle.dump(Lexicon, f)
    f.close()
    with open(file + '/phrase/doc_map.pkl', 'wb') as fl:
        pickle.dump(dict3,fl)
    fl.close()
    terms = {i[1]:1 for i in Lexicon}

    f = open('merge/phrase.txt', 'r')
    fs = open('merge/phrase_final.txt', 'a')
    c=0

    while True:
        a = f.readline()
        i = a.split(sep=',')
        if a == '':
            break
        if int(i[0]) in terms:
            fs.write(a)
            c+=1
    f.close()
    fs.close()
    print('totally posting lists:', c)
    os.remove('merge/phrase.txt')

    if os.path.exists('BigSample/phrase/phrase_final.txt'):
        os.remove('merge/phrase_final.txt')
    else:
        shutil.move('merge/phrase_final.txt','BigSample/phrase')
def position_index_generation(file,files,dict2,dict3,initial_time,limit):

    docno=0
    wordno = 0
    memory = False
    count = 0
    temp_map = collections.defaultdict(int)
    # assert whether have temp file to store temporary files
    try:
        os.makedirs('temp')
    except:
        print('Have temp files')
    try:
        os.makedirs(file+'/position')
    except:
        print('Have output files')
    for file_name in files:
        if re.search(r'\.',file_name):
            with open(file +'/'+ file_name, 'rt') as f:
                for line in (generate_file_lines(f)):
                    # determine wheter it is up to limit
                    if re.match('</DOC>',line):
                        if memory:
                            temp=sorted(temp_map.keys(),key=lambda x:dict2[x])
                            temp_file=collections.defaultdict(int)
                            for W in temp:
                                for doc,info in temp_map.get(W).items():
                                    temp_file[(dict2[W],dict3[doc])]=info
                            with open('temp/' + str(count) + '.txt', 'a') as fs:
                                for triple in temp_file:
                                    fs.write(
                                        str(triple[0]) + ',' + str(triple[1]) + ',' + str(temp_file[triple]) + '\n')
                                    # pickle.dump((triple[0],triple[1],temp_file[triple]),fs)

                            fs.close()

                            # print(data.shape)
                            temp_map = collections.defaultdict(int)
                            memory=False

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
                                if word not in dict2:
                                    dict2[word]=wordno
                                    wordno+=1

                                # use triple counter to help determine memory limits
                                if word not in temp_map:
                                    count+=1
                                else:
                                    if DocID not in temp_map[word]:
                                        count+=1
                                if count%int(limit)==0:
                                    memory=True
                                position_in_dict(word, temp_map, DocID, word_pos)
        else:
            continue
        print('at present Docno:',docno)
        print('at present time:', time.time()-initial_time)

    if temp_map:
        temp=sorted(temp_map.keys(),key=lambda x:dict2[x])
        temp_file=collections.defaultdict(int)
        for W in temp:
            for doc,info in temp_map.get(W).items():
                temp_file[(dict2[W],dict3[doc])]=info
        with open('temp/' + str(count) + '.txt', 'a') as fs:
            for triple in temp_file:
                fs.write(str(triple[0]) + ',' + str(triple[1]) + ',' + str(temp_file[triple]) + '\n')
                # pickle.dump((triple[0],triple[1],temp_file[triple]),fs)

    fs.close()

    print('the time after generating:',time.time()-initial_time)

    num = merge()

    if num!=1: #disc-based
        if os.path.exists('BigSample/position/position.txt'):
            os.remove('merge/temp' + str(num) + '.txt')
        else:
            os.rename('merge/temp' + str(num) + '.txt', 'merge/position.txt')
            shutil.move('merge/position.txt', 'BigSample/position')
    else: #memory-based
        F=os.listdir('temp').pop()
        if os.path.exists('BigSample/position/position.txt'):
            os.remove('temp/'+F)
        else:
            os.rename('temp/'+F,'temp/position.txt')
            shutil.move('temp/position.txt', 'BigSample/position')
    shutil.rmtree('temp')
    f = open('BigSample/position/position.txt', 'r')
    print('the time after merging:', time.time() - initial_time)
    temp = collections.defaultdict(int)
    c=0
    while True:
        a = f.readline()
        i = a.split(sep=',')
        if a == '':
            break
        if int(i[0]) in temp:
            temp[int(i[0])].append(int(i[1]))
        else:
            temp[int(i[0])] = [int(i[1])]
        c+=1
    f.close()
    print('totally posting lists:', c)
    dict2 = {v: k for k, v in dict2.items()}
    Lexicon = [(dict2[i], i, len(temp[i])) for i in temp.keys()]
    print('totally lexicon lists:', len(Lexicon))
    with open(file+'/position/position_lexicon.pkl', 'wb') as f:
        pickle.dump(Lexicon, f)
    f.close()
    with open(file + '/position/doc_map.pkl', 'wb') as fl:
        pickle.dump(dict3,fl)
    fl.close()
def stem_index_generation(file,files,dict2,dict3,stop_word,initial_time,limit):
    porter_stemmer = PorterStemmer()
    docno = 0
    wordno = 0
    memory = False
    count = 0
    temp_map = collections.defaultdict(int)
    # assert whether have temp file to store temporary files
    try:
        os.makedirs('temp')
    except:
        print('Have temp files')
    try:
        os.makedirs(file+'/stemming')
    except:
        print('Have output files')
    for file_name in files:
        if re.search(r'\.',file_name):
            with open(file +'/'+ file_name, 'rt') as f:
                for line in (generate_file_lines(f)):
                    # determine wheter it is up to limit
                    if re.match('</DOC>',line):
                        if memory:
                            try:
                                temp=sorted(temp_map,key=lambda x:dict2[x])
                            except:
                                c={v: k for k, v in dict2.items()}
                                temp = sorted(temp_map, key=lambda x: c[x])
                            temp_file=collections.defaultdict(int)
                            for W in temp:
                                for doc,info in temp_map.get(W).items():
                                    temp_file[(dict2[W],dict3[doc])]=info
                            with open('temp/' + str(count) + '.txt', 'a') as fs:
                                # pickle.dump(temp_file,fs)
                                for triple in temp_file:
                                    fs.write(
                                        str(triple[0]) + ',' + str(triple[1]) + ',' + str(temp_file[triple]) + '\n')
                            fs.close()

                            temp_map = collections.defaultdict(int)
                            memory=False
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
                                if word not in dict2:
                                    dict2[word]=wordno
                                    wordno+=1
                                # use triple counter to help determine memory limits
                                if word not in temp_map:
                                    count+=1
                                else:
                                    if DocID not in temp_map[word]:
                                        count+=1
                                if count%int(limit)==0:
                                    memory=True
                                add_in_dict(word, temp_map, DocID)
            print('at present Docno:', docno)
            print('at present time:', time.time() - initial_time)
        else:
            continue
    # ensure the last temporary file
    if temp_map:
        temp = sorted(temp_map.keys(), key=lambda x: dict2[x])
        temp_file = collections.defaultdict(int)
        for W in temp:
            for doc, info in temp_map.get(W).items():
                temp_file[(dict2[W], dict3[doc])] = info
        with open('temp/' + str(count) + '.txt', 'a') as fs:
            # pickle.dump(temp_file,fs)
            for triple in temp_file:
                fs.write(
                    str(triple[0]) + ',' + str(triple[1]) + ',' + str(temp_file[triple]) + '\n')
    fs.close()

    print('the time after generating:',time.time()-initial_time)
    # generate posting list
    num = merge()
    if num!=1: #disc-based
        if os.path.exists('BigSample/stemming/stem.txt'):
            os.remove('merge/temp' + str(num) + '.txt')
        else:
            os.rename('merge/temp' + str(num) + '.txt', 'merge/stem.txt')
            shutil.move('merge/stem.txt', 'BigSample/stemming')
    else: #memory-based
        F=os.listdir('temp').pop()
        if os.path.exists('BigSample/stemming/stem.txt'):
            os.remove('temp/'+F)
        else:
            os.rename('temp/'+F,'temp/stem.txt')
            shutil.move('temp/stem.txt', 'BigSample/stemming')
    shutil.rmtree('temp')
    f = open('BigSample/stemming/stem.txt', 'r')
    print('the time after merging:',time.time()-initial_time)
    temp = collections.defaultdict(int)
    c=0
    while True:
        a = f.readline()
        i = a.split(sep=',')
        if a == '':
            break
        if int(i[0]) in temp:
            temp[int(i[0])].append(1)
        else:
            temp[int(i[0])] = [1]
        c+=1
    f.close()
    print('totally posting lists:', c)
    dict2 = {v: k for k, v in dict2.items()}
    Lexicon = [(dict2[i], i, len(temp[i])) for i in temp.keys()]
    print('totally lexicon lists:', len(Lexicon))
    with open(file+'/stemming/stemming_lexicon.pkl', 'wb') as f:
        pickle.dump(Lexicon, f)
    f.close()
    with open(file + '/stemming/doc_map.pkl', 'wb') as fl:
        pickle.dump(dict3,fl)
    fl.close()


if __name__ == '__main__':
    main()