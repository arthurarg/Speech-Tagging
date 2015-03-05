import sys
import nltk
import math

import numpy as np
from nltk.corpus import brown

#this function takes the words from the training data and returns a python list of all of the words that occur more than 5 times
#wbrown is a python list where every element is a python list of the words of a particular sentence
def calc_known(wbrown):
    knownwords = []
    
    # dic contains words counting
    dic={}
    for l in wbrown:
        for w in l:
            if w in dic:
                dic[w] += 1
            else:
                dic[w]=1
                
            # a soon as a word occurs more than 5 time, we add it to knownwords
            if dic[w]==6:
                knownwords.append(w)
    return knownwords

#this function takes a set of sentences and a set of words that should not be marked '_RARE_'
#brown is a python list where every element is a python list of the words of a particular sentence
#and outputs a version of the set of sentences with rare words marked '_RARE_'
def replace_rare(brown, knownwords):
    rare = []
    
    # go through all the words of the corpus
    for l in brown:
        temp=[]
        for w in l:
        
        	# if w not known, replaced with "_RARE_" in the new sentence temp
            if w in knownwords:
                temp.append(w)
            else:
                temp.append("_RARE_")
                
        # new sentence temp added to the new set rare
        rare.append(temp) 
    return rare

#this function takes the ouput from replace_rare and outputs it
def q3_output(rare):
    outfile = open("B3.txt", 'w')

    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()

#this function takes tags from the training data and calculates trigram probabilities
#tbrown (the list of tags) should be a python list where every element is a python list of the tags of a particular sentence
#it returns a python dictionary where the keys are tuples that represent the trigram, and the values are the log probability of that trigram
def calc_trigrams(tbrown):
	# dictionary counting trigram occurrences
    qvalues = {}
    # dictionary counting bigram occurrences
    c={}
    # go through lines
    for l in tbrown:
        for i in range(2, len(l)):
        	# count trigrams
            key=tuple((l[i-2], l[i-1], l[i],))
            if key in qvalues:
                qvalues[key] +=1
            else:
                qvalues[key] = 1
            
            # count bigrams
            key=tuple((l[i-1], l[i],))
            if key in c:
                c[key] +=1
            else:
                c[key] = 1
    
    # occurrences of start symbol = nb of sentences
    c[('*','*')]=len(tbrown)
    
    # compute trigram prob from previous counting
    for key in qvalues:
        qvalues[key] = math.log(1.0*qvalues[key]/c[key[:-1]], 2)
    
    return qvalues

#this function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(qvalues):
    #output
    outfile = open("B2.txt", "w")
    for trigram in qvalues:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(qvalues[trigram])])
        outfile.write(output + '\n')
    outfile.close()

#this function calculates emission probabilities and creates a list of possible tags
#the first return value is a python dictionary where each key is a tuple in which the first element is a word
#and the second is a tag and the value is the log probability of that word/tag pair
#and the second return value is a list of possible tags for this data set
#wbrown is a python list where each element is a python list of the words of a particular sentence
#tbrown is a python list where each element is a python list of the tags of a particular sentence
def calc_emission(wbrown, tbrown):
    evalues = {}
    taglist = []
    
    # dic with tag counting
    count = {}
    
    for i in range(0, len(wbrown)):
    
    	# count occurrences of the tuple (word, tag) over all the set
        for w in range(0, len(wbrown[i])):
            t = tuple( (wbrown[i][w], tbrown[i][w]) )
            if t in evalues:
                evalues[t] += 1
            else:
                evalues[t] = 1
            
            # count the tag
            if t[1] in count:
                count[t[1]] += 1
            else:
                    count[t[1]] = 1
                    taglist.append(t[1])
    
    # compute the proba from occurrences counting
    for key in evalues:
        evalues[key] = math.log( 1.0*evalues[key]/count[key[1]], 2)
    return evalues, taglist

#this function takes the output from calc_emissions() and outputs it
def q4_output(evalues):
    #output
    outfile = open("B4.txt", "w")
    for item in evalues:
        output = " ".join([item[0], item[1], str(evalues[item])])
        outfile.write(output + '\n')
    outfile.close()


#this function takes data to tag (brown), possible tags (taglist), a list of known words (knownwords),
#trigram probabilities (qvalues) and emission probabilities (evalues) and outputs a list where every element is a string of a
#sentence tagged in the WORD/TAG format
#brown is a list where every element is a list of words
#taglist is from the return of calc_emissions()
#knownwords is from the the return of calc_knownwords()
#qvalues is from the return of calc_trigrams
#evalues is from the return of calc_emissions()
#tagged is a list of tagged sentences in the format "WORD/TAG". Each sentence is a string with a terminal newline, not a list of tokens.
def viterbi(brown, taglist, knownwords, qvalues, evalues):
    tagged = []
    n=len(taglist)
    
    brown_rare = replace_rare(brown, knownwords)
    
    # STEP 1 - compute states proba
    
    # go through lines of brown and brown smoothed
    for line0, line in zip(brown, brown_rare):
        
        # track keep track of argmax{ p(j,k,l),l }
        track=np.zeros( (len(line), n, n), dtype=int)
        
        # b(j, k)=max{ p(j,k,l),l }
        b=-1000*np.ones((n,n))
        b[0,0]=0
        
        # go through every words of the sentence
        for i in range(2, len(line)):
        	
        	# compute temp(j, k)=max{ p(j,k,l),l } for word i from b(k,l)
            temp=np.zeros((n,n))
            
            # go through every possible state for word i
            for j in range(0, n):
            	# o = proba of word i given state j
                o=(line[i], taglist[j])
                
                # if o==0 b(j,k)=-1000 for each k
                # no need to compute all the proba
                start=-1000
                if o in evalues:
                    start = evalues[o]
                else:
                    temp[j,:]=-1000*np.ones(n)
                    continue
                
                # go through all the possible states (k,l) for previous words (i-1, i-2)
                # temp(j,k)=max{p(j,k,l), l}
                for k in range(0, n):
                    m=-1000                    
                    for l in range(0,n):
                        p=b[k,l]
                        t= (taglist[l], taglist[k], taglist[j])
                        if t in qvalues:
                            p += qvalues[t]
                        else:
                            p += -1000
                        
                        p+=start
                        
                        # track[i,j,k] = argmax{p(j,k,l), l}
                        if p>m:
                            m=p
                            track[i, j, k] = l
                    temp[j, k]=m
            # store temp as b, states proba for words (i, i-1)
            b=temp
            
            
        # STEP 2 - use back pointer (track) to find the most likely state
        
        # find the highest proba for the sentence, b(i,j)
        m=-1000
        i=0
        j=0
        for k in xrange(n):
            for l in xrange(n):
                if b[k, l] > m:
                    i=k
                    j=l
                    m=b[k, l]
        
        # find the state of w corresponding to the highest proba going backward
        w=len(line)-1
        # store tags sequence in tags
        tags = np.zeros(len(line), dtype=int)
        while w>=2:
            if w<len(line)-1:
                tags[w] = i
            
            # track[w, i, j] is the state for (w-2) that gives the highest proba to have (w, w-1) in state (i, j)
            temp=track[w, i, j]
            i=j
            j=temp
            w -= 1
        
        # compute the tagged sentence from tags sequence
        newline=""
        for i in range(2, len(line)-1):
            if i>2:
                newline += " "
            newline += line0[i]+"/"+taglist[tags[i]]
        tagged.append(newline+"\n")
        

            
    return tagged

#this function takes the output of viterbi() and outputs it
def q5_output(tagged):
    outfile = open('B5.txt', 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

#this function uses nltk to create the taggers described in question 6
#brown is the data to be tagged
#tagged is a list of lists of tokens in the WORD/TAG format.
def nltk_tagger(brown):
	
	# trains the taggers
	training = nltk.corpus.brown.tagged_sents(tagset='universal')
	default_tagger = nltk.DefaultTagger('NOUN')
	bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
	trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)
	
	# tags each line of brown, stores into tagged
	tagged = []
	for line in brown:
		tags=trigram_tagger.tag(line)
		s=[]
		for i in range(2, len(line)-1):
			s.append(tags[i][0]+"/"+tags[i][1])
		tagged.append(s)
	return tagged

def q6_output(tagged):
    outfile = open('B6.txt', 'w')
    for sentence in tagged:
        output = ' '.join(sentence) + '\n'
        outfile.write(output)
    outfile.close()

#a function that returns two lists, one of the brown data (words only) and another of the brown data (tags only)
def split_wordtags(brown_train):
	# function that split <word/tag> and return (<word>, <tag>)
	def split_dash(word):
		i=len(word)
		while word[i-1]!='/' and i>0:
			i=i-1
		return word[:(i-1)], word[i:]
	
	wbrown = []
	tbrown = []
	for l in brown_train:# go through each line
		wl=[]
		tl=[]
		
		# splits sentence in tokens <word/tag>
		tokens = l.split()
		wl += ["*",  "*"]
		tl += ["*",  "*"]
		for e in tokens:# splits the token and stores word and tag in 2 separate lists wl, wt
			s=split_dash(e)
			wl.append(s[0])
			tl.append(s[1])
		wl.append("STOP")
		tl.append("STOP")
		
		wbrown+=[wl]
		tbrown+=[tl]
	return wbrown, tbrown
	
def main():
    #open Brown training data
    infile = open("Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    #split words and tags, and add start and stop symbols (question 1)
    wbrown, tbrown = split_wordtags(brown_train)

    #calculate trigram probabilities (question 2)
    qvalues = calc_trigrams(tbrown)

    #question 2 output
    q2_output(qvalues)

    #calculate list of words with count > 5 (question 3)
    knownwords = calc_known(wbrown)

    #get a version of wbrown with rare words replace with '_RARE_' (question 3)
    wbrown_rare = replace_rare(wbrown, knownwords)

    #question 3 output
    q3_output(wbrown_rare)

    #calculate emission probabilities (question 4)
    evalues, taglist = calc_emission(wbrown_rare, tbrown)

    #question 4 output
    q4_output(evalues)

    #delete unneceessary data
    del brown_train
    del wbrown
    del tbrown
    del wbrown_rare

    #open Brown development data (question 5)
    infile = open("Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()
    
    #format Brown development data here
    # splits sentences in lists, adds start and stop symbols
    temp=[]
    for l in brown_dev:
        temp.append( ["*","*"]+l.split()+["STOP"] )
    brown_dev = temp
    
    
    #do viterbi on brown_dev (question 5)
    viterbi_tagged = viterbi(brown_dev, taglist, knownwords, qvalues, evalues)
    
    
    #question 5 output
    q5_output(viterbi_tagged)

    #do nltk tagging here
    nltk_tagged = nltk_tagger(brown_dev)
    
    #question 6 output
    q6_output(nltk_tagged)
    
if __name__ == "__main__": main()
