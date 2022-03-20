from index_variable import *
import re
import datetime
import pickle
import collections
import numpy as np
import time
import os
from nltk.stem.porter import PorterStemmer
import argparse




def bm25(words,lexicon,doc,idf,doc_num):

    dl=[sum(j.values()) for i,j in doc.items() if j!=0]
    avg_dl=sum(dl)/doc_num
    b=0.75
    k1=1.2
    k2=50
    doc_score={}
    for i in range(1, doc_num+1):
        doc_score[i]=0
    for w in words:
        index=lexicon[w]
        qtf=words.count(w)
        for i in range(1, doc_num+1):
            try:
                if index not in doc[i]:
                    continue
                else:

                    dl=sum(doc[i].values())
                    doc_score[i]+=idf[index]*(doc[i][index] * (k1 + 1))\
                    /(doc[i][index] + k1 * (1 - b + b * dl / avg_dl))*((k2+1)*qtf)/(k2+qtf)
            except:
                continue
    return doc_score


def generate_file_lines(file):
    for line in file:
        yield line

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
    text=text.casefold()
    return text
def special_check(text,regex_rule):
    if re.search(regex_rule,text):
        special_words=re.search(regex_rule,text)
        word=special_words.group()

    return word

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
def digit_deal(text):
    text = re.sub(sp_regex9, lambda x: ''.join(x.group().split(sep=',')), text)
    text = re.sub(r'\$?\d+\.00', lambda x: x.group().split(sep='.')[0], text)
    return text

def read_stop_list(stop_file):
    stop_word_list=[]
    with open(stop_file, 'r') as stop:
        for i in stop:
            stop_word_list.append(i.strip())
    stop.close()
    return stop_word_list
def word_tokenize(word, stop_word,sp_list):
    stop_word_list = stop_word
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
            sp_list.append(alpha)
            return word
        word = re.sub(sp_regex1, lambda x: ''.join(x.group().split(sep='.')), word)

    # digit-alpha
    if re.fullmatch(sp_regex4, word):
        alpha = word.split(sep='-')[1]

        if len(alpha) >= 3 and alpha not in stop_word_list:
            word = ''.join(word.split(sep='-'))
            # add_in_dict(alpha, dict, docid)
            # words = [alpha, word]
            sp_list.append(alpha)

        word = ''.join(word.split(sep='-'))
    # alpha-digit
    if re.fullmatch(sp_regex3, word):
        alpha = word.split(sep='-')[0]
        if len(alpha) >= 3 and alpha not in stop_word_list:
            word = ''.join(word.split(sep='-'))
            # words = [alpha, word]
            sp_list.append(alpha)


        word = ''.join(word.split(sep='-'))
    # hyphen
    if re.match(sp_regex5, word):
        words = word.split(sep='-')
        word = ''.join(words)
        for h_word in words:
            if h_word in stop_word_list or h_word in pre_list:
                # add_in_dict(h_word, dict, docid)
                # words.remove(h_word)
                continue
            else:
                sp_list.append(h_word)
    # remove()[]
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
def min_dis(l1,l2):
    ans=float('inf')
    j, k = 0, 0
    while j < len(l1) and k < len(l2):
        ans = min(ans, abs(l1[j] - l2[k]))
        if l1[j] < l2[k]:
            j += 1
        else:
            k += 1

    return ans

def main():


    parser = argparse.ArgumentParser(description='query')
    parser.add_argument('index_file', type=str, help='You should enter accurate document address')
    parser.add_argument('query_file', type=str, help='You should enter accurate query address')
    parser.add_argument('results_file', type=str, help='You should enter accurate output address')
    args = parser.parse_args()
    if not os.path.exists(args.results_file):
        os.makedirs(args.results_file)
    stop_words_ad = input('enter stops address:')
    stop_word = read_stop_list(stop_words_ad)
    stop_word_list = stop_word
    doc=open(args.index_file+'/single/doc_map.pkl','rb')
    doc_map=pickle.load(doc)
    doc_num=len(doc_map)
    doc.close()

    print('start generate phrase dict')
    start_time=time.time()
    phrase_lexicon=collections.defaultdict(int)
    with open(args.index_file+'/phrase/phrase_lexicon.pkl','rb') as f:
        d=pickle.load(f)
        for term in d:
            phrase_lexicon[term[0]]=term[1]

    f.close()
    phrase_doc_dict=collections.defaultdict(int)
    with open(args.index_file+'/phrase/phrase.pkl','rb') as f:
        d = pickle.load(f)
        phrase_dict=d

        for term, info in d.items():
            for doc, fre in info.items():
                if doc in phrase_doc_dict:
                    phrase_doc_dict[doc][term] = fre
                else:
                    phrase_doc_dict[doc] = {term: fre}
    f.close()
    print('-'*50)

    print('start generate position dict')
    position_lexicon=collections.defaultdict(int)
    with open(args.index_file+'/position/position_lexicon.pkl','rb') as f:
        d=pickle.load(f)
        for term in d:
            position_lexicon[term[0]]=term[1]
    f.close()
    position_doc_dict=collections.defaultdict(int)
    with open(args.index_file+'/position/position.pkl','rb') as f:
        d = pickle.load(f)
        position_dict=d
        for term, info in d.items():
            for doc, pos in info.items():
                if doc in position_doc_dict:
                    position_doc_dict[doc][term] = pos
                else:
                    position_doc_dict[doc] = {term: pos}
    f.close()
    print('-'*50)

    print('start generate single dict')
    idf_lexicon=collections.defaultdict(int)
    single_lexicon=collections.defaultdict(int)
    with open(args.index_file+'/single/single_lexicon.pkl','rb') as f:
        d=pickle.load(f)
        for term in d:
            single_lexicon[term[0]]=term[1]
            #idf_lexicon[term[1]]=np.log10(doc_num/term[2]) #cosine
            idf_lexicon[term[1]]=np.log10((doc_num-term[2]+0.5)/(term[2]+0.5))   #bm25
    f.close()
    doc_dict=collections.defaultdict(int)
    with open(args.index_file+'/single/single.pkl','rb') as f:
        d = pickle.load(f)
        single_dict=d

        for term, info in d.items():
            for doc, fre in info.items():
                if doc in doc_dict:
                    doc_dict[doc][term] = fre
                else:
                    doc_dict[doc] = {term: fre}
    f.close()
    print('-'*50)

    print('start query')
    retrieve_num=100
    doc_map = {int(value): key for key, value in doc_map.items()}
    result_file=open(args.results_file+'/dynamic_results.txt','a')
    with open(args.query_file,'r') as f:
        for line in generate_file_lines(f):
            if not line.isspace():
                if re.match(r'<num>',line):
                    num=int(line.split(sep=':')[1])
                if re.match(r'<title>',line):
                    line=line.split(sep=':')[1]
                    line=common_line_deal(line)
                    line=date_deal(line)
                    line=digit_deal(line)
                    word_list = line.split()
                    words=[]
                    sp_words = []
                    phrases=[]
                    documents=[]
                    if len(word_list)>=2:
                        for index in range(len(word_list)-1):
                             word_1,word_2=word_tokenize_common(word_list[index]),word_tokenize_common(word_list[index+1])
                             if isbi_valid(word_1,word_2,stop_word_list):
                                  phrases.append(word_1+' '+word_2)
                    if len(word_list)>=3:
                        for index in range(len(word_list)-2):
                             word_1,word_2=word_tokenize_common(word_list[index]),word_tokenize_common(word_list[index+1])
                             word_3=word_tokenize_common(word_list[index+2])
                             if istri_valid(word_1,word_2,word_3,stop_word_list):
                                  phrases.append(word_1 + ' ' + word_2+ ' ' + word_3)
                    if phrases:
                        for phrase in phrases:
                            if phrase in phrase_lexicon:
                                info=phrase_dict[phrase_lexicon[phrase]]
                                if len(info)>3:
                                    for D,T in info.items():
                                        documents.append((D,T))

                    if not documents:

                        Phrases=[]
                        word_list=[word_tokenize_common(i) for i in word_list]
                        length=len(word_list)
                        f_set=set([i for i in range(doc_num+1)])
                        for w in word_list:
                            index=position_lexicon[w]
                            f_set=f_set.intersection(set(position_dict[index].keys()))

                        for D in f_set:
                            Dis = 0
                            com=True
                            for index in range(length-1):
                                index1, index2 = position_lexicon[word_list[index]], position_lexicon[word_list[index + 1]]
                                pos1,pos2=sorted(position_doc_dict[D][index1]),sorted(position_doc_dict[D][index2])
                                dis = min_dis(pos1, pos2)

                                if dis>5:
                                    com=False
                                    break
                                else:
                                    Dis+=dis
                            if com and Dis>0:
                                documents.append((D,Dis/length))
                        # if len(word_list) >= 2:
                        #     for index in range(len(word_list) - 1):
                        #         word_1, word_2 = word_tokenize_common(word_list[index]), word_tokenize_common(
                        #             word_list[index + 1])
                        #         Phrases.append(word_1 + ' ' + word_2)
                        # if len(word_list) >= 3:
                        #     for index in range(len(word_list) - 2):
                        #         word_1, word_2 = word_tokenize_common(word_list[index]), word_tokenize_common(
                        #             word_list[index + 1])
                        #         word_3 = word_tokenize_common(word_list[index + 2])
                        #         Phrases.append(word_1 + ' ' + word_2 + ' ' + word_3)
                        # for phrase in phrases:
                        #     Words = phrase.split()
                        #     if len(Words)==2 and Words[0] in position_lexicon and Words[1] in position_lexicon:
                        #         index1,index2=position_lexicon[Words[0]],position_lexicon[Words[1]]
                        #         set1,set2=set(position_dict[index1].keys()),set(position_dict[index2].keys())
                        #         final=set1.intersection(set2)
                        #         for D in final:
                        #             pos1,pos2=sorted(position_doc_dict[D][index1]),sorted(position_doc_dict[D][index2])
                        #             dis=min_dis(pos1,pos2)
                        #             if dis<5:
                        #                 documents.append((D,dis))
                        #     elif len(Words)==3 and Words[0] in position_lexicon and Words[1] in position_lexicon and Words[2] in position_lexicon:
                        #         index1, index2,index3 = position_lexicon[Words[0]], position_lexicon[Words[1]],position_lexicon[Words[2]]
                        #         set1, set2,set3= set(position_dict[index1].keys()), set(position_dict[index2].keys()), set(position_dict[index3].keys())
                        #         final=set1.intersection(set2).intersection(set3)
                        #         for D in final:
                        #             pos1,pos2,pos3=sorted(position_doc_dict[D][index1]),sorted(position_doc_dict[D][index2]),sorted(position_doc_dict[D][index3])
                        #
                        #             dis1 = min_dis(pos1, pos2)
                        #             dis2=min_dis(pos2,pos3)
                        #             if dis1<=5 and dis2<5:
                        #                 documents.append((D,(dis1+dis2)/2))

                        # temp={}
                        # for i in documents:
                        #     if i[0] in temp:
                        #         temp[i[0]]=min(i[1],temp.get(i[0]))
                        #     else:
                        #         temp[i[0]]=i[1]
                        # documents=[(i,j) for i,j in temp.items()]
                        documents=sorted(documents,key=lambda x:x[1])
                    else:
                        documents=list(set(documents))
                        documents=sorted(documents,reverse=True,key=lambda x:x[1])

                    print('Retrieved doc nums(position and phrase):',num,len(documents))

                    if len(documents)>=retrieve_num:
                        for i in range(retrieve_num):
                            result_file.write(str(num) + ' ' + '0 ' + doc_map[documents[i][0]] + ' ' + str(i+1) + ' ' \
                                          + str(documents[i][1]) + ' ' + 'dynamic\n')

                    else:
                        for w in word_list:
                            if w in stop_word_list or w in r_list or w.isspace():
                                continue
                            word = word_tokenize(w, stop_word_list, sp_words)
                            words.append(word)
                        words = words + sp_words
                        score = bm25(words, single_lexicon, doc_dict, idf_lexicon,doc_num)
                        score = sorted([(i, j) for i, j in score.items()], reverse=True, key=lambda x: x[1])
                        n = 0
                        if len(documents)==0:
                            while len(documents)<retrieve_num:
                                documents.append(score[n])
                                n+=1
                        else:
                            docs={i[0]:0 for i in documents}
                            while len(documents)<retrieve_num:
                                if score[n][0] not in docs:
                                    documents.append(score[n])
                                n+=1

                        for i in range(retrieve_num):
                            result_file.write(str(num) + ' ' + '0 ' + doc_map[documents[i][0]] + ' ' + str(i+1) + ' ' \
                                          + str(documents[i][1]) + ' ' + 'dynamic\n')
    result_file.close()
    print('time cost:', time.time() - start_time)
    file = args.results_file+'/dynamic_results.txt'
    try:
        command = 'treceval qrel.txt {}'.format(file)
        os.system(command)
    except:
        print('Please put treceval and qrel file in the same directory')

main()