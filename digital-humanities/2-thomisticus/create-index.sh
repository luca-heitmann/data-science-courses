#With -n also the line numbers where the patters are found can be returned
grep -n servus OpenLibrary-CorpusThomisticus.txt
#This includes a regular expression which we will master in a later exercise!
grep -Eon ".{1,20}servus{1,20}" OpenLibrary-CorpusThomisticus.txt
