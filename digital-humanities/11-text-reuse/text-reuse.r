library(quanteda)
library(textreuse)
library(tidyverse)

source("text_reuse_text_source.R")
text <- paste(
  "How does it feel, how does it feel?",
  "To be without a home",
  "Like a complete unknown, like a rolling stone"
)
text.corpus <- corpus(text)
text.corpus <- corpus_reshape(text.corpus, to = "sentences")
text.corpus
