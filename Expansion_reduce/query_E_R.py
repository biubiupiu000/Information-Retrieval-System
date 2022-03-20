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

def cosine(words,lexicon,dic_weights,idf,doc_num):

    doc_score=collections.defaultdict(int)
    for i in range(1,doc_num+1):
        sum1,sum2,sum3=0,0,0
        for w in words:
            index=lexicon[w]
            #weight=idf[index]*np.log10(words.count(w) + 1)
            weight=idf[index]*words.count(w)
            sum2+=np.power(weight,2)
            try:
                if index in dic_weights[i]:
                    sum1 += dic_weights[i][index] * weight
                    sum3 += np.power(dic_weights[i][index], 2)
                    # sum1+= dic_weights[i][index] * weight
                    # sum3+=np.power(dic_weights[i][index],2)
            except:
                doc_score[i]=0
                break
        if sum1!=0:

            doc_score[i]=sum1/np.power(sum2*sum3,1/2)
        else:
            doc_score[i] = 0
    return doc_score

def language_model(words,lexicon,doc,indexs,doc_num):
    dl = [sum(j.values()) for i, j in doc.items() if j!=0]
    avg_dl = sum(dl) / doc_num
    sum_dl=sum(dl)
    doc_score = {}
    for i in range(1, doc_num + 1):
        doc_score[i] = 0
    for w in words:
        index=lexicon[w]
        qft=words.count(w)
        Q=len(words)
        for i in range(1, doc_num+1):
            try:
                if index not in doc[i]:
                    continue
                else:

                    dl=sum(doc[i].values())
                    terms=sum(indexs[index].values())
                    doc_score[i]+=1/np.log10((doc[i][index]+avg_dl*terms/sum_dl)\
                                  /(dl+avg_dl))*-1

                    #doc_score[i]+=1/np.log10((doc[i][index]+avg_dl*terms/sum_dl)/(dl+avg_dl))*qft/Q*-1
            except:
                continue

    return doc_score

def bm25(words,lexicon,doc,idf,doc_num,word):

    dl=[sum(j.values()) for i,j in doc.items() if j!=0]
    avg_dl=sum(dl)/doc_num
    b=0.75
    k1=1.2
    k2=200
    doc_score={}
    for i in range(1, doc_num+1):
        doc_score[i]=0
    for w in words:
        index=lexicon[w]
        qtf=word.count(w)
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


def feedback(feedback_docs,feedback_terms,lexicon,score,doc,idf,o_terms):

    f_docs = [i[0] for i in score[:feedback_docs]]
    temp = {}
    for i in f_docs:
        for j in doc[i]:
            # if j not in temp:
            #     temp[j] = doc[i][j]
            # else:
            #     temp[j] += doc[i][j]
            if j not in temp:
                temp[j] = 1
            else:
                temp[j] += 1
    for i in temp:
        temp[i] *= idf[i]

    terms = sorted([(i, j) for i, j in temp.items()], reverse=True, key=lambda x: x[1])
    r_terms=[]
    o_terms=collections.Counter(o_terms)
    while feedback_terms>0:
        temp=list(lexicon.keys())[list(lexicon.values()).index(terms.pop(0)[0])]
        if temp not in o_terms:
            r_terms.append(temp)
            feedback_terms-=1
    return r_terms


def co_occur(num,o_terms,lexicon,dic,doc):
    r_terms=[]
    for i in o_terms:
        index=lexicon[i]
        co_terms=[]

        for j in dic[index]:
            temp=list(doc[j].keys())
            temp.remove(index)
            co_terms=co_terms+temp
        co_terms=collections.Counter(co_terms)
        co_terms=sorted([(i, j) for i, j in co_terms.items()], reverse=True, key=lambda x: x[1])
        co_terms=[list(lexicon.keys())[list(lexicon.values()).index(i[0])] for i in co_terms[:num]]
        r_terms=r_terms+co_terms
    return r_terms


def reduction(words,lexicon,dic,word,thres):
    temp = {}
    for i in words:
        index = lexicon[i]
        if index not in temp:
            #max_tf = max(dic[index].values())
            # temp[index]=lexicon[index]*max_tf
            temp[index] = words.count(i)
    temp = sorted([(i, j) for i, j in temp.items()], reverse=True, key=lambda x: x[1])
    threshold = round(len(word) * thres)

    q_terms = [i[0] for i in temp[:threshold]]
    q_terms = [list(lexicon.keys())[list(lexicon.values()).index(i)] for i in q_terms]
    return q_terms
def main():
    parser = argparse.ArgumentParser(description='query')
    parser.add_argument('index_file', type=str, help='You should enter accurate document address')
    parser.add_argument('query_file', type=str, help='You should enter accurate query address')
    parser.add_argument('index_mode', type=str, help='It only has two modes(single and stem)')
    parser.add_argument('retrieval_mode', type=str, help='It only has three models(cosine, bm25 and lr)')
    parser.add_argument('expansion_mode', type=str, help='It only has two methods')
    parser.add_argument('results_file', type=str, help='You should enter accurate output address')
    args = parser.parse_args()
    if not os.path.exists(args.results_file):
        os.makedirs(args.results_file)
    stop_words_ad=input('enter stops address:')
    stop_word=read_stop_list(stop_words_ad)
    stop_word_list=stop_word

    if os.path.exists(args.results_file + '/' + args.retrieval_mode + '_' + args.index_mode + '_results.txt'):
        os.remove(args.results_file + '/' + args.retrieval_mode + '_' + args.index_mode + '_results.txt')
    result_file = open(args.results_file + '/' + args.retrieval_mode + '_' + args.index_mode + '_results.txt', 'a')
    threshold = input('please enter the fraction of query terms(from 0-1): ')
    if args.expansion_mode == 'feedback':
        ex_doc_nums = input('please enter the num of the top n docs: ')
        ex_term_nums = input('please enter the num of the top m terms: ')
    elif args.expansion_mode == 'occurrence':
        ex_term_nums = input('please enter the num of expansion terms for each word: ')
    start_time = time.time()
    if args.index_mode=='single':
        doc=open(args.index_file+'/single/doc_map.pkl','rb')
        doc_map=pickle.load(doc)
        doc.close()

        print('generate doc dict')
        doc_dict = collections.defaultdict(int)
        with open(args.index_file+'/single/single.pkl', 'rb') as f:
            d = pickle.load(f)
            single_dict = d

            for term, info in d.items():
                for doc, fre in info.items():
                    if doc in doc_dict:
                        doc_dict[doc][term] = fre
                    else:
                        doc_dict[doc] = {term: fre}
        f.close()
        print('doc dict complete')
        doc_num = len(doc_dict)

        print('generate idf and lexicon dict')
        idf_lexicon=collections.defaultdict(int)
        single_lexicon=collections.defaultdict(int)
        with open(args.index_file+'/single/single_lexicon.pkl','rb') as f:
            d=pickle.load(f)
            for term in d:
                single_lexicon[term[0]]=term[1]
                if args.retrieval_mode=='bm25':
                    idf_lexicon[term[1]]=np.log10((doc_num-term[2]+0.5)/(term[2]+0.5))   #bm25
                elif args.retrieval_mode=='cosine' or args.retrieval_mode=='lm':
                    idf_lexicon[term[1]] = np.log10(doc_num / term[2])  # cosine
                else:
                    print('Don\'t have this model')
                    return

        f.close()
        if args.retrieval_mode=='cosine':
            dic_weights = collections.defaultdict(int)
            for i in range(1, doc_num + 1):
                try:
                    info = doc_dict[i]
                    sum = 0
                    for index, fre in info.items():
                        sum += np.power((np.log10(fre) + 1) * idf_lexicon[index], 2)
                    for index, fre in info.items():
                        if i not in dic_weights:
                            dic_weights[i] = {index: (np.log10(fre) + 1) * idf_lexicon[index] / sum}
                        else:
                            dic_weights[i][index] = (np.log10(fre) + 1) * idf_lexicon[index] / sum
                except:
                    continue
        print('idf dict complete')

        print('start query')
        doc_map = {int(value): key for key, value in doc_map.items()}
        with open(args.query_file,'r') as f:
            content = False
            for line in generate_file_lines(f):
                if not line.isspace():
                    if re.match(r'<num>', line):
                        words = []
                        num = int(line.split(sep=':')[1])
                    if content and not re.match(r'</top>', line):
                        line = common_line_deal(line)
                        line = date_deal(line)
                        line = digit_deal(line)
                        sp_words = []
                        word_list = line.split()
                        for w in word_list:
                            if w in stop_word_list or w in r_list or w.isspace():
                                continue
                            word = word_tokenize(w, stop_word_list, sp_words)
                            words.append(word)
                        words = words + sp_words

                    if re.match(r'<narr>', line):
                        content = True
                    if re.match(r'</top>', line):
                        word=list(set(words))
                        content = False
                        q_terms=reduction(words,single_lexicon,single_dict,word,float(threshold))
                        if args.retrieval_mode=='bm25':
                            score=bm25(q_terms,single_lexicon,doc_dict,idf_lexicon,doc_num,words)
                        else:
                            score=cosine(q_terms,single_lexicon,dic_weights,idf_lexicon,doc_num)
                        score=sorted([(i,j) for i,j in score.items()],reverse=True,key=lambda x:x[1])
                        if args.expansion_mode=='feedback':
                            f_terms=feedback(int(ex_doc_nums),int(ex_term_nums),single_lexicon,score,doc_dict, idf_lexicon, q_terms)
                        elif args.expansion_mode=='occurrence':
                            f_terms=co_occur(int(ex_term_nums),q_terms,single_lexicon,single_dict,doc_dict)

                        sp_words = []
                        for w in f_terms:
                            if w in stop_word_list or w in r_list or w.isspace():
                                continue

                            word = word_tokenize(w, stop_word_list, sp_words)
                            q_terms.append(word)
                            words.append(word)
                        words = words + sp_words
                        q_terms = q_terms + sp_words

                        word = list(set(q_terms))
                        if args.retrieval_mode == 'bm25':
                            score = bm25(word, single_lexicon, doc_dict, idf_lexicon, doc_num, words)
                        else:
                            score = cosine(word, single_lexicon, dic_weights, idf_lexicon, doc_num)
                        score = sorted([(i, j) for i, j in score.items()], reverse=True, key=lambda x: x[1])

                        score = [i for i in score if i[1] > 0]
                        if len(score) >= 100:
                            for i in range(0, 100):
                                result_file.write(str(num) + ' ' + '0 ' + doc_map[score[i][0]] + ' ' + str(i) + ' ' \
                                                  + str(score[i][1]) + ' ' + args.retrieval_mode + '\n')
                        else:
                            for i in range(len(score)):
                                result_file.write(str(num) + ' ' + '0 ' + doc_map[score[i][0]] + ' ' + str(i) + ' ' \
                                                  + str(score[i][1]) + ' ' + args.retrieval_mode + '\n')


        result_file.close()
        print('single results output complete')
        print('time cost:',time.time()-start_time)
        file = args.results_file + '/' + args.retrieval_mode + '_' + args.index_mode + '_results.txt'
        try:
            command = 'treceval qrel.txt {}'.format(file)
            os.system(command)
        except:
            print('Please put treceval and qrel file in the same directory')

    elif args.index_mode=='stem':
        doc = open(args.index_file + '/stem/doc_map.pkl', 'rb')
        doc_map = pickle.load(doc)
        doc.close()

        print('generate doc dict')
        doc_dict = collections.defaultdict(int)
        with open(args.index_file + '/stem/stem.pkl', 'rb') as f:
            d = pickle.load(f)
            stem_dict = d

            for term, info in d.items():
                for doc, fre in info.items():
                    if doc in doc_dict:
                        doc_dict[doc][term] = fre
                    else:
                        doc_dict[doc] = {term: fre}
        f.close()
        print('doc dict complete')
        doc_num = len(doc_dict)

        print('generate idf and lexicon dict')
        idf_lexicon = collections.defaultdict(int)
        stem_lexicon = collections.defaultdict(int)
        with open(args.index_file + '/stem/stem_lexicon.pkl', 'rb') as f:
            d = pickle.load(f)
            for term in d:
                stem_lexicon[term[0]] = term[1]
                if args.retrieval_mode == 'bm25':
                    idf_lexicon[term[1]] = np.log10((doc_num - term[2] + 0.5) / (term[2] + 0.5))  # bm25
                elif args.retrieval_mode == 'cosine' or args.retrieval_mode == 'lm':
                    idf_lexicon[term[1]] = np.log10(doc_num / term[2])  # cosine
                else:
                    print('Don\'t have this model')
                    return
        f.close()
        if args.retrieval_mode=='cosine':
            dic_weights = collections.defaultdict(int)
            for i in range(1, doc_num + 1):
                try:
                    info = doc_dict[i]
                    sum = 0
                    for index, fre in info.items():
                        sum += np.power((np.log10(fre) + 1) * idf_lexicon[index], 2)
                    for index, fre in info.items():
                        if i not in dic_weights:
                            dic_weights[i] = {index: (np.log10(fre) + 1) * idf_lexicon[index] / sum}
                        else:
                            dic_weights[i][index] = (np.log10(fre) + 1) * idf_lexicon[index] / sum
                except:
                    continue
        print('idf dict complete')

        print('start query')
        porter_stemmer = PorterStemmer()
        doc_map = {int(value): key for key, value in doc_map.items()}
        with open(args.query_file, 'r') as f:
            content = False
            for line in generate_file_lines(f):
                if not line.isspace():
                    if re.match(r'<num>', line):
                        words = []
                        num = int(line.split(sep=':')[1])
                    if content and not re.match(r'</top>', line):
                        line = common_line_deal(line)
                        line = date_deal(line)
                        line = digit_deal(line)
                        word_list = line.split()
                        for w in word_list:
                            if w in stop_word_list or w in r_list or w.isspace():
                                continue
                            word = word_tokenize_common(w)
                            words.append(porter_stemmer.stem(word))
                    if re.match(r'<narr>', line):
                        content = True
                    if re.match(r'</top>', line):
                        word = list(set(words))
                        content = False
                        q_terms = reduction(words, stem_lexicon, stem_dict, word, float(threshold))
                        if args.retrieval_mode == 'bm25':
                            score = bm25(q_terms, stem_lexicon, doc_dict, idf_lexicon, doc_num, words)
                        else:
                            score = cosine(q_terms, stem_lexicon, dic_weights, idf_lexicon, doc_num)
                        score = sorted([(i, j) for i, j in score.items()], reverse=True, key=lambda x: x[1])
                        if args.expansion_mode == 'feedback':
                            f_terms = feedback(int(ex_doc_nums), int(ex_term_nums),stem_lexicon, score, doc_dict,
                                               idf_lexicon, q_terms)
                        elif args.expansion_mode == 'occurrence':
                            f_terms = co_occur(int(ex_term_nums), q_terms, stem_lexicon, stem_dict, doc_dict)
                        for w in f_terms:
                            if w in stop_word_list or w in r_list or w.isspace():
                                continue

                            word = word_tokenize_common(w)
                            q_terms.append(word)
                            words.append(word)
                        word = list(set(q_terms))
                        if args.retrieval_mode == 'bm25':
                            score = bm25(word, stem_lexicon, doc_dict, idf_lexicon, doc_num, words)
                        else:
                            score = cosine(word, stem_lexicon, dic_weights, idf_lexicon, doc_num)
                        score = sorted([(i, j) for i, j in score.items()], reverse=True, key=lambda x: x[1])

                        score = [i for i in score if i[1] > 0]
                        if len(score) >= 100:
                            for i in range(0, 100):
                                result_file.write(str(num) + ' ' + '0 ' + doc_map[score[i][0]] + ' ' + str(i) + ' ' \
                                                  + str(score[i][1]) + ' ' + args.retrieval_mode + '\n')
                        else:
                            for i in range(len(score)):
                                result_file.write(str(num) + ' ' + '0 ' + doc_map[score[i][0]] + ' ' + str(i) + ' ' \
                                                  + str(score[i][1]) + ' ' + args.retrieval_mode + '\n')

        result_file.close()
        print('single results output complete')
        print('time cost:', time.time() - start_time)
        file = args.results_file + '/' + args.retrieval_mode + '_' + args.index_mode + '_results.txt'
        try:
            command = 'treceval qrel.txt {}'.format(file)
            os.system(command)
        except:
            print('Please put treceval and qrel file in the same directory')
main()