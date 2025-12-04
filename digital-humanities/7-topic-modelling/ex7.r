options(stringsAsFactors = FALSE)
library(quanteda)
require(topicmodels)

textdata <- read.csv("../data/sotu_new.csv", sep = ",",
encoding = "UTF-8")
sotu_corpus <- corpus(textdata$Paragraph, docnames = textdata$X)
# Build a dictionary of lemmas
lemma_data <- read.csv("../data/resources/baseform_en.tsv",
encoding = "UTF-8")
# extended stopword list
stopwords_extended <- readLines("../data/resources/stopwords_en.txt",
encoding = "UTF-8")
# Create a DTM (may take a while)
corpus_tokens <- sotu_corpus %>%
    tokens(remove_punct = TRUE, remove_numbers = TRUE, remove_symbols = TRUE) %>%
    tokens_tolower() %>%
    tokens_replace(lemma_data$inflected_form,
    lemma_data$lemma,
    valuetype = "fixed") %>%
    tokens_remove(pattern = stopwords_extended, padding = T)

sotu_collocations <- quanteda.textstats::textstat_collocations(
    corpus_tokens,
    min_count = 25)
sotu_collocations <- sotu_collocations[1:250, ]
corpus_tokens <- tokens_compound(corpus_tokens, sotu_collocations)

