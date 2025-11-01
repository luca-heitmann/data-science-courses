library(quanteda)
# First, read all the lines with
text <- readLines("/Users/luca/Projects/ds/digital-humanities/2-thomisticus/OpenLibrary-CorpusThomisticus.txt")
# single token matching
toks <- tokens(text)
# Just some renaming of the document names
docnames(toks) <- 1:length(text)
# The kwic command creates a concordance view based on the entered pattern
res <- kwic(toks, pattern = "servu*", valuetype = "glob", window = 3)
print(res)
