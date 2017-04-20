import tensorflow as tf
import re
from pathlib2 import Path

caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


def get_sentences():
	contents = Path('./data/data_sentences.txt').read_text()
	sentences = split_into_sentences(contents)	
	return sentences


def get_encoded_sentences_for_decoder(lookup_table, max_len):
	sentences = get_sentences()
	encoded_sentences = list()
	
	for sn in sentences:
		words = sn.split(' ')
		
		padding_len = max_len - len(words)
		tmp_sentences = list()
		
		tmp_sentences = [lookup_table[word] for word in words if word in lookup_table]
		
		if len(tmp_sentences) < max_len:
			tmp_sentences.extend([lookup_table["PAD"] for i in range(max_len - len(tmp_sentences))])
		else:
			tmp_sentences = tmp_sentences[:max_len]
		
		tmp_sentences = [lookup_table["GO"]] + tmp_sentences[:-1]
		encoded_sentences.append(tmp_sentences)
	
	return encoded_sentences
	
def get_encoded_sentences(lookup_table, max_len):
	sentences = get_sentences()
	encoded_sentences = list()
	
	for sn in sentences:
		words = sn.split(' ')
		
		padding_len = max_len - len(words)
		tmp_sentences = list()
		
		tmp_sentences = [lookup_table[word] for word in words if word in lookup_table]
		
		if len(tmp_sentences) < max_len:
			tmp_sentences.extend([lookup_table["PAD"] for i in range(max_len - len(tmp_sentences))])
		else:
			tmp_sentences = tmp_sentences[:max_len]
			
		encoded_sentences.append(tmp_sentences)
	
	return encoded_sentences
	
	
def split_into_sentences(text):
	text = " " + text + "  "
	text = text.replace("\n"," ")
	text = re.sub(prefixes,"\\1<prd>",text)
	text = re.sub(websites,"<prd>\\1",text)
	if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
	text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
	text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
	text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
	text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
	text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
	text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
	text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
	if "”" in text: text = text.replace(".”","”.")
	if "\"" in text: text = text.replace(".\"","\".")
	if "!" in text: text = text.replace("!\"","\"!")
	if "?" in text: text = text.replace("?\"","\"?")
	text = text.replace(".",".<stop>")
	text = text.replace("?","?<stop>")
	text = text.replace("!","!<stop>")
	text = text.replace("<prd>",".")

	text = text.replace("\"", '')
	text = text.replace("--", '')
	text = text.replace(",", '')
	text = text.replace("“", '')
	text = text.replace("”", '')
	text = text.replace("!", '')
	text = text.replace("?", '')
	text = text.replace("‘", '')
	text = text.replace("’", '')
	text = text.replace(".", '')
	text = text.lower()

	sentences = text.split("<stop>")
	sentences = sentences[:-1]
	sentences = [s.strip() for s in sentences]
	return sentences	
	
	
	
