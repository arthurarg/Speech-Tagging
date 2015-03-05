Homework 1 - COMS4705
UNI: aa3572
Name: Arthur Argenson

Running time:
- part A: 1:30
- part B: 9:00

$ A1:
first lines of A1.txt:
UNIGRAM detractor -19.4274397095
UNIGRAM wolves -17.8424772088
UNIGRAM midweek -19.4274397095
UNIGRAM scoring -17.4274397095
UNIGRAM 54,320 -19.4274397095

$ A2:
A2 uni:1104.83292814
A2 bi:57.2215464238
A2 tri:5.89521267642

first lines of A2.uni.txt:
-178.786615001
-260.137472974
-143.533157817
-118.988741702
-148.06169626

We see that perplexity decreases dramatically from unigram to trigram. Thus we can conclude that the best of the tree model is the trigram one.

$ A3
first lines of A3.txt:
-46.6216736724
-85.799449324
-58.5689202358
-47.5324350611
-52.7601723486

A3 perplexity:13.0759217039

$ A4
We can see that the linear model perplexity is between the ones of trigram and bigram models and far from the one of the unigram model.
It is likely that only the probabilities of tri/bigrams have an impact. Unigram proba might be too small.


$ A5
Perplexity results:
A5 sample1: 11.6492786046
A5 sample2: 6.87993973157e+170

Sample1 perplexity is even smaller than on the training-set. We can reasonably conclude that the sample might be from Brown dataset.
However, Sample2 perplexity is huge compared to the one over the training-set. It is very likely that it does not belong to the brown dataset.

$ B2
First lines of B2.txt:
TRIGRAM * X VERB -3.90689059561
TRIGRAM NOUN VERB ADV -2.84564255455
TRIGRAM PRT VERB DET -1.84386447329
TRIGRAM * ADP VERB -4.93221475197
TRIGRAM PRT CONJ . -7.92481250361

$ B3
First lines of B3.txt:
At that time highway engineers traveled rough and dirty roads to accomplish their duties .
_RARE_ _RARE_ vehicles was a personal _RARE_ for such employees , and the matter of providing state transportation was felt perfectly _RARE_ .

$ B4
First lines of B4.txt:
tubes NOUN -12.711977598
hundred NUM -6.32713325402
red ADJ -9.15426433692
fire VERB -13.8689189884
brick NOUN -14.0745476774

$ B5
First lines of B5.txt:
He/PRON had/VERB obtained/VERB and/CONJ provisioned/VERB a/DET veteran/ADJ ship/NOUN called/VERB the/DET Discovery/NOUN and/CONJ had/VERB recruited/VERB a/DET crew/NOUN of/ADP twenty-one/NOUN ,/. the/DET largest/ADJ he/PRON had/VERB ever/ADV commanded/VERB ./.
The/DET purpose/NOUN of/ADP this/DET fourth/ADJ voyage/NOUN was/VERB clear/ADJ ./.

Percent correct tags: 92.9657787246

$ B6
First lines of B6.txt:
He/PRON had/VERB obtained/VERB and/CONJ provisioned/VERB a/DET veteran/NOUN ship/NOUN called/VERB the/DET Discovery/NOUN and/CONJ had/VERB recruited/VERB a/DET crew/NOUN of/ADP twenty-one/NUM ,/. the/DET largest/ADJ he/PRON had/VERB ever/ADV commanded/VERB ./.
The/DET purpose/NOUN of/ADP this/DET fourth/ADJ voyage/NOUN was/VERB clear/ADJ ./.

Percent correct tags: 95.3911700901

As expected the backed-off tagger performs better on the test-set. The trigram-tagger is the best of the 3 (uni/bi/trigrams taggers) however it is more likely to fail on a given sentence.
Using the 2 other as bak-off, allows to considerably reduce the failure rate, while keeping the best tagger to tag most sentences.