from bs4 import BeautifulSoup
import urllib.request
import urllib.error
import re
import os
import string

def scrape():

	#url = raw_input('Enter URL with links to pages\n')
	url = 'https://www.recurse.com/blog/archive'
	page = urllib.request.urlopen(url)
	soup = BeautifulSoup(page, "html.parser")
	items = soup.findAll('a')
	print(len(items))

	for i in range(len(items)):
		print(i,items[i].get_text())
	
	range_input = input('Enter range separated by comma\n')
	num_range = [int(number.strip()) for number in range_input.split(',')]

	for i in range(num_range[0],num_range[1]):
		
		print(i)
		link = 'https://www.recurse.com' + items[i]['href']
		print(link)

		path = 'posts/%s' % (items[i]['href'].replace('/','-'))
		outfile = open(path, 'w')
		try:
			page = urllib.request.urlopen(link)
		except:
			page = None

		if page:
			soup = BeautifulSoup(page, "html.parser")
			paragraphs = soup.findAll('p')
			for p in paragraphs:
				outfile.write(p.get_text() + '\n\n')




scrape()