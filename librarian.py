import csv
import os
import nltk
import math
import model.dictionary as dictionary
from itertools import islice
from nltk.tokenize import TweetTokenizer

punctuation = ['.', ',', '!', '?', '(', ')', '$', ':', ';', '{', '}', '[', ']', '•', '|']

def text_from_path(path):
	with open(path) as f:
		return f.read()

def get_data(path, n=100000):
	with open(path) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		rows = [row for row in islice(csv_reader,0,n)]
		return rows

def get_data_from_tab_delimited(path, n=100000, headers=None):
	with open(path, encoding = "ISO-8859-1") as f:
		next(f)
		reader=csv.reader(f,delimiter='\t')
		rows = [row for row in islice(reader,0,n)]
	if headers:
		rows.insert(0, headers)	# prepend to list
	return rows

# directory of only text files (NOTE: In future, consider scraping to CSV instead)
def get_data_from_directory(dirpath, headers=['','']):
	rows = [headers]
	for filename in os.listdir(dirpath):
		if os.path.isdir(filename):
			return "directory must only contain text files"
		if not dirpath[-1] == '/':
			dirpath += '/'
		filepath = dirpath+filename
		with open(filepath, encoding = "ISO-8859-1") as f:
			text = f.read()
			rows.append([filename, text])
	return rows


# remove any dictionaries with suspiciously similar profiles
def eliminate_duplicates(tds, threshold=.9):
	if len(tds) == 0:
		return []
	else:		
		head = tds[0]
		remainder = tds[1:]
		if list_contains_duplicate(head, remainder, threshold):
			return eliminate_duplicates(remainder, threshold)
		else:
			return [head] + eliminate_duplicates(remainder, threshold)

# given a dictionary and a list of dictionaries, returns whether any is a duplicate
def list_contains_duplicate(d, dlist, threshold=.9):
	for head in dlist:
		if dictionary.similarity(d, head) > threshold:
			return True
	return False



def all_text_from_column(rows, col_name):

	if col_name in rows[0]:
		n = rows[0].index(col_name)
		return [row[n] for row in rows[1:]]
	else:
		return ''


### TEXT PROCESSING METHODS ###

def split_into_sentences(text):
	tokenizer = TweetTokenizer()
	return tokenizer.tokenize(text)

def split_into_words(sentence):
	tokenizer = TweetTokenizer()
	return [w.lower() for w in tokenizer.tokenize(sentence) if not w in punctuation]


# a dictionary of all terms in the document of length n
def term_dict(doc, n=1):
	term_dict = {}
	words = split_into_words(doc)
	for i in range(len(words)+1-n):
		term = " ".join(words[i:i+n])
		if term in term_dict:
			term_dict[term] += 1
		else:
			term_dict[term] = 1
	return term_dict

# a list of dictionaries of terms in the document of length n
def term_dicts(corpus, n=1):
	return [term_dict(d, n) for d in corpus]

# list of integers representing term frequency across documents
def frequency_distribution(term, tds):
	freqs = []
	for td in tds:
		if term in td:
			freqs.append(td[term])
		else:
			freqs.append(0)
	return freqs

# how many times the term appears in the document
def term_frequency(term, doc):
	return term_dict(doc)[term]

# how many documents in the corpus include the term
def doc_frequency(term, all_tds):
	return len([1 for td in all_tds if term in td])

# a measure of how topical this term is for this document
def tf_idf(doc, corpus):
	pass

# list of the same length as the corpus list with top tf-idf candidates for topic words
def keywords(corpus, td_list, num_keywords):
	pass


# returns lower and upper bounds containing 95 percent of occurrence rates of the term
def tf_bounds(term, tds, n=2):
	distribution = frequency_distribution(term, tds)
	m = mean(distribution)
	sd = stdev(distribution)
	return m - n*sd, m + n*sd

# returns terms in a dictionary that occur in at least two (or n) dictionaries from a list of dictionaries
def non_unique_terms(term_dict, dict_list, n=1):
	return {k: v for k, v in term_dict.items() if doc_frequency(k, dict_list) >= n}

# returns a dictionary of document frequencies (df) for all terms in the dictionary list
# only includes terms with a df of at least n
def df_dict(term_dict, dict_list, threshold=1):
	to_return = {}
	for k, v in term_dict.items():
		df = doc_frequency(k, dict_list)
		if df >= threshold:
			to_return[k] = df
	return to_return



#### SEARCH ####

# takes a list of docs and a corresponding (equal length) list of term dicts
def docs_containing_term(term, docs, term_dicts):
	return [docs[i] for i, td in enumerate(term_dicts) if term in td]


### LOAD RESOURCES ###

def stopwords():
	with open('resources/stopwords.txt') as f:
		return set(f.read().split('\n'))



### STATISTICAL METHODS ####

# standard deviation
def stdev(values):
	N = len(values)
	mean = sum(values) / N
	sum_squared_differences = sum([(x-mean)**2 for x in values])
	return math.sqrt(sum_squared_differences / (N-1))

def mean(values):
	return sum(values) / len(values)




#### SAVING DICTIONARIES ####

def save_ngrams_from_field(docs, field, n, dirname='NO DIR', dup_threshold=.8):
	text = all_text_from_column(docs,field)
	tds = term_dicts(text,n)		# one term dictionary per document
	unique_tds = eliminate_duplicates(tds, threshold=dup_threshold)
	big_td = dictionary.union(unique_tds)	# one term dictionary for all documents

	to_save = df_dict(big_td, unique_tds, threshold=2)
	print(len(to_save))
	dirpath = 'stats/%s/' % dirname
	if not os.path.exists(dirpath):
		os.mkdir(dirpath)
	filename = '%s_%sg_df.txt' % (field.lower(), n)
	savepath = dirpath + filename
	dictionary.to_tab_delimited(to_save, savepath)

def process_directory(dirname, dup_threshold=.8):
	dirpath = 'data/%s' % dirname
	savepath = 'stats/%s/' % dirname

	if not os.path.exists(dirpath):
		os.mkdir(dirpath)
	if not os.path.exists(savepath):
		os.mkdir(savepath)

	rows = get_data_from_directory(dirpath, headers=['Filename','Songtext'])

	for n in range(1, 6):
		save_ngrams_from_field(rows, 'Songtext', n, dirname=dirname, dup_threshold=dup_threshold)




# arranges documents into groups of a given size
def group_into_documents(small_docs, group_size):

	num_docs = len(small_docs)

	new_docs = ['' for x in range(num_docs//group_size)]

	for doc_index in range(len(new_docs)):
		for i in range(group_size):
			full_index = doc_index*group_size+i
			new_docs[doc_index] += small_docs[full_index] + '\n'
	return new_docs

# returns a list of keywords for the doc in the context of the corpus
def keywords(doc_td, corpus_tds):

	total_docs = len(corpus_tds)

	tfidfs = {k: tfidf(k, doc_td, corpus_tds) for k in doc_td.keys()}

	return dictionary.sort_descending(tfidfs)[:10]

def tfidf(k, doc_td, corpus_tds):
	total_docs = len(corpus_tds)
	tf = doc_td[k]
	df = doc_frequency(k,corpus_tds)
	idf = -1 * math.log(df/total_docs)
	return tf*idf

