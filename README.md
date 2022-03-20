# Information-Retrieval-System

## Environment Require
### Language
Python 3.8+

### Libraries
argparse, os, datetime, re, collections, time, pickle, nltk, shutil, numpy

## Built Index
```
Python all_index_disc.py  --file  --stop_file  --mode  --memory_limit
```
>file-> text files folder path <br>
>stop_file-> stopwords file <br>
>Index_mode-> options(single, phrase, position and stem)<br>
>Memory_limit-> the memory constraint through the project(if you want test memory-based method, please enter any number bigger than 500000)

You can get three files for each mode in the file path ‘BigSample/<mode_name>/’. They are the pattern as following:<br>
1. ‘<mode_name>.txt’ including the triples<termid, docid, term frequency><br>
In positon.txt, the term frequency is replaced by positional information lists. In phrase folder, the name of posting list is ‘phrase_final.txt’.<br>
2. ‘<mode_name>_lexicon.pkl’ including the triples<term, termid, document frequency>.<br>
3. ‘doc_map.pkl’ including doc. Dictionary for{doc:docid}<br>

## Query
```
python  build.py  --file  --stop_file  --index_mode 
```
>file-> text files folder path<br>
stop_file-> stopwords file<br>
Index_mode-> options(single, phrase, position and stem)<br>
```
python query_static.py --index_file --query_file --index_mode --retrieval_mode  --results_file
```
>index_file-> the generated index folder path<br>
query_file-> query file<br>
index_mode-> options(single and stem)<br>
retrieval_mode-> options(cosine, bm25,lm)<br>
results_file-> the output folder path<br>

You can get the file ‘<retrieval_mode>_<index_mode>_results’ in the final output path. The pattern is based on what treceval requires.<br>
```
python query_dynamic.py --index_file --query_file  --results_file
```
>index_file-> the generated index folder path<br>
query_file-> query file<br>
results_file-> the output folder path<br>

You can get the file ‘dynamic_results’ in the final output path. The pattern is based on what treceval requires.<br>

## Expansion and Reduction
```
python query_expansion.py --index_file --query_file --index_mode --retrieval_mode --expansion_mode  --results_file
```
>index_file-> the generated index folder path<br>
query_file-> query file<br>
index_mode-> options(single and stem)<br>
retrieval_mode-> options(cosine, bm25)<br>
expansion_mode-> options(feedback and occurrence)<br>
results_file-> the output folder path<br>

You can get the file ‘<retrieval_mode>_<index_mode>_results’ in the final output path. The pattern is based on what treceval requires.<br>
```
python query_reduction.py --index_file --query_file --index_mode --retrieval_mode --sort_mode  --results_file
```
>index_file-> the generated index folder path<br>
query_file-> query file<br>
index_mode-> options(single and stem)<br>
retrieval_mode-> options(cosine and bm25)<br>
sort_mode-> options(tfidf and query)<br>
results_file-> the output folder path<br>

You can get the file ‘<retrieval_mode>_<index_mode>_results’ in the output directory. The pattern is based on what treceval requires.
```
python query_E_R.py --index_file --query_file --index_mode --retrieval_mode --expansion_mode  --results_file
```
>index_file-> the generated index folder path<br>
query_file-> query file<br>
index_mode-> options(single and stem)<br>
retrieval_mode-> options(cosine, bm25)<br>
expansion_mode-> options(feedback and occurrence)<br>
results_file-> the output folder path<br>
