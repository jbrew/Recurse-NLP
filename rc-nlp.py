import librarian
import operator
import model.tagger as tagger
from nltk.tokenize import TweetTokenizer

### TEXT PROCESSING METHODS ###

def split_into_sentences(text):
	tokenizer = TweetTokenizer()
	return tokenizer.tokenize(text)

def split_into_words(sentence):
	tokenizer = TweetTokenizer()
	punctuation = ['.', ',', '!', '?', '(', ')', '$', ':', ';', '{', '}', '[', ']', '•', '|']
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


# how many times the term appears in the document
def term_frequency(term, doc):
	return term_dict(doc)[term]

# how many documents in the corpus include the term
def doc_frequency(term, term_dicts):
	return len([1 for td in term_dicts if term in td])

# list of the same length as the corpus list with top tf-idf candidates for topic words
def keywords(corpus, term_dicts, num_keywords):
	pass	# TODO


def graph(term, term_dicts):

	print('\n\tfrequency of "' + term.upper() + '"')
	for i in range(0, len(term_dicts), 10):
		term_dict_set = term_dicts[i:i+10]
		count = sum([td[term] for td in term_dict_set if term in td])
		line = str(i) + '\t' + '|'*count
		print(line)



def get_number(blog_title):
	return int(blog_title.split('-')[2])



if __name__ == '__main__':
	rows = librarian.get_data_from_directory('posts', ['filename','main_text'])

	sorted_rows = sorted(rows[1:], key=lambda row: get_number(row[0]))

	main_texts = [main_text for filename, main_text in sorted_rows]

	tds = term_dicts(main_texts)

	print(doc_frequency('the', tds))

	words = ['hacker', 'recurse']

	for word in words:
		graph(word, tds)



# list of integers representing term frequency across documents
def frequency_distribution(term, term_dicts):
	freqs = []
	for td in term_dicts:
		if term in td:
			freqs.append(td[term])
		else:
			freqs.append(0)
	return freqs




