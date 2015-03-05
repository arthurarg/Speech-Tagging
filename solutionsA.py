import nltk
from nltk.collocations import *
import math

#a function that calculates unigram, bigram, and trigram probabilities
#brown is a python list of the sentences
#this function outputs three python dictionaries, where the key is a tuple expressing the ngram and the value is the log probability of that ngram
#make sure to return three separate lists: one for each ngram
def calc_probabilities(brown):

	# dic is a list with 3 dictionaries for uni/bi/trigram occurrences
    dic=[{}, {}, {}]
    
    # total number of tokens
    length=0
    
    # function to add/increase a new/already seen occurrence
    def dic_add(elem, dic):
        if elem in dic:
            dic[elem]+=1
        else:
            dic[elem]=1
            
            
	# count the occurrences of each tuple and store in dic
    for l in brown:
        tokens = nltk.word_tokenize("* * "+l+" STOP")
        length += len(tokens)-2

        for i in range(2, len(tokens)):
            dic_add((tokens[i],), dic[0])
            dic_add((tokens[i-1], tokens[i] ), dic[1])
            dic_add((tokens[i-2], tokens[i-1], tokens[i]), dic[2])
    
    # compute the probabilities from the occurrence counting
    # loop over the 3 dic
    for n in range(0, 3):
    
    	# loop over the tuples of the dic
        for key in dic[n]:
        
        	# computation of log-pb
            dic[n][key] = math.log(1.0*dic[n][key]/length, 2)
            
            # if bi/trigram the denominator is compensated by p(previous b/unigram)
            if n>0:
                if key[:-1]==tuple(('*',)) or key[:-1]==tuple(('*', '*')):
                    dic[n][key] +=  math.log(1.0*length/len(brown), 2)
                else:
                    dic[n][key] -=  dic[n-1][key[:-1]]
                    
            # if trigram the denominator is compensated again by p(unigram)
            if n>1:
                if key[:-1]==tuple(('*', '*')):
                    continue
                if key[:-2]==tuple(('*',)):
                    dic[n][key] += math.log(1.0*length/len(brown), 2)
                else:
                    dic[n][key] -=  dic[n-2][key[:-2]]
                
    return (e for e in dic)

#each ngram is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams):
    #output probabilities
    outfile = open('A1.txt', 'w')
    for unigram in unigrams:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')
    for bigram in bigrams:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')
    for trigram in trigrams:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')
    outfile.close()
    
#a function that calculates scores for every sentence
#ngram_p is the python dictionary of probabilities
#n is the size of the ngram
#data is the set of sentences to score
#this function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, data):
    n -= 1
    scores=[]
    
    # go through each sentence
    for l in data:
    	# extract tokens
        tokens = nltk.word_tokenize("* * "+l+" STOP")
        
        # sum the pb of each tokens of sentence l
        # tokens are of length (n+1)
        s=0
        for i in range(2, len(tokens)):
            s += ngram_p[tuple(tokens[j] for j in range(i-n, i+1))]
        
        scores+=[s]
    
    return scores


#this function outputs the score output of score()
#scores is a python list of scores, and filename is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()


#this function scores brown data with a linearly interpolated model
#each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
#like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, brown):

	# gather each model proba in a single list
    ngram_p=[unigrams, bigrams, trigrams]
    
    scores = []
    # go through the sentences
    for l in brown:
        tokens = nltk.word_tokenize("* * "+l+" STOP")
        
        # go through every words and sum the proba
        s=0
        for i in range(2, len(tokens)):
        
        	# compute the proba according to the 3 models and store in temp 
            temp=0
            for n in range(0,3):
                tpl=tuple(tokens[j] for j in range(i-n, i+1))
                if tpl in ngram_p[n]:
                    temp += math.pow( 2, ngram_p[n][tpl] )
                    
            # compute the log-proba of the sum
            if temp!=0:
                s += math.log(temp/3.0, 2)
            else:
                s += -1000
                
        scores+=[s]
    return scores

def main():
    #open data
    infile = open('Brown_train.txt', 'r')
    brown = infile.readlines()
    infile.close()

    #calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(brown)

    #question 1 output
    q1_output(unigrams, bigrams, trigrams)

    #score sentences (question 2)
    uniscores = score(unigrams, 1, brown)
    biscores = score(bigrams, 2, brown)
    triscores = score(trigrams, 3, brown)

    #question 2 output
    score_output(uniscores, 'A2.uni.txt')
    score_output(biscores, 'A2.bi.txt')
    score_output(triscores, 'A2.tri.txt')

    #linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, brown)

    #question 3 output
    score_output(linearscores, 'A3.txt')

    #open Sample1 and Sample2 (question 5)
    infile = open('Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open('Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    #score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    #question 5 output
    score_output(sample1scores, 'Sample1_scored.txt')
    score_output(sample2scores, 'Sample2_scored.txt')

if __name__ == "__main__": main()
