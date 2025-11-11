options(stringsAsFactors = FALSE)
library(quanteda)
# read the SOTU corpus data
textdata <- read.csv("./data/sotu_new.csv", sep = ",", encoding = "UTF-8")
sotu_corpus <- corpus(textdata$Paragraph, docnames = textdata$X)
# Build a dictionary of lemmas
lemma_data <- read.csv("./data/resources/baseform_en.tsv", encoding = "UTF-8")
# read an extended stop word list
stopwords_extended <- readLines("./data/resources/stopwords_en.txt", encoding = "UTF-8")
# Preprocessing of the corpus
corpus_tokens <- sotu_corpus %>%
    tokens(remove_punct = TRUE,
           remove_numbers = TRUE,
           remove_symbols = TRUE) %>%
    tokens_tolower() %>%
    tokens_replace(lemma_data$inflected_form,
                   lemma_data$lemma,
                   valuetype = "fixed") %>%
    tokens_remove(pattern = stopwords_extended, padding = T)

# calculate multi-word unit candidates
sotu_collocations <- quanteda.textstats::textstat_collocations(corpus_tokens, min_count = 25)
# check top collocations
print(head(sotu_collocations, 25))
# check bottom collocations
print(tail(sotu_collocations, 25))

# We will treat the top 250 collocations as MWU
sotu_collocations <- sotu_collocations[1:250, ]
# compound collocations
corpus_tokens <- tokens_compound(corpus_tokens, sotu_collocations)
# Create DTM (also remove padding empty term)
DTM <- corpus_tokens %>%
    tokens_remove("") %>%
    dfm()

print(DTM)

# Compute IDF: log(N / n_i)
number_of_docs <- nrow(DTM)
term_in_docs <- colSums(DTM > 0)
idf <- log(number_of_docs / term_in_docs)
# Compute TF
year <- substr(textdata$Date,1,4)
first_obama_speech <- which(textdata$President == "Barack Obama" & year == "2009")
tf <- as.vector(colSums(DTM[first_obama_speech, ]))
# Compute TF-IDF
tf_idf <- tf * idf
names(tf_idf) <- colnames(DTM)

print(sort(tf_idf, decreasing = T)[1:20])

targetDTM <- DTM
termCountsTarget <- as.vector(colSums(targetDTM[first_obama_speech, ]))
names(termCountsTarget) <- colnames(targetDTM)
# Just keep counts greater than zero
termCountsTarget <- termCountsTarget[termCountsTarget > 0]

lines <- readLines("./data/resources/eng_news_2020_300K-sentences.txt", encoding = "UTF-8")
corpus_compare <- corpus(lines)

# Create a DTM (may take a while)
corpus_compare_tokens <- corpus_compare %>%
tokens(remove_punct = TRUE, remove_numbers = TRUE, remove_symbols = TRUE) %>%
tokens_tolower() %>%
tokens_replace(lemma_data$inflected_form,
lemma_data$lemma,valuetype = "fixed") %>%
tokens_remove(pattern = stopwords_extended, padding = T)
# Create DTM
comparisonDTM <- corpus_compare_tokens %>%
tokens_compound(sotu_collocations) %>%
tokens_remove("") %>%
dfm()
termCountsComparison <- colSums(comparisonDTM)

# Loglikelihood for a single term
term <- "health_care"
# Determine variables
a <- termCountsTarget[term]
b <- termCountsComparison[term]
c <- sum(termCountsTarget)
d <- sum(termCountsComparison)
# Compute log likelihood test
Expected1 = c * (a+b) / (c+d)
Expected2 = d * (a+b) / (c+d)
t1 <- a * log((a/Expected1))
t2 <- b * log((b/Expected2))
logLikelihood <- 2 * (t1 + t2)
print(logLikelihood)
# use set operation to get terms only occurring in target document
uniqueTerms <- setdiff(names(termCountsTarget), names(termCountsComparison))
# Have a look into a random selection of terms unique in the target corpus
sample(uniqueTerms, 5)

# Create vector of zeros to append to comparison counts
zeroCounts <- rep(0, length(uniqueTerms))
names(zeroCounts) <- uniqueTerms
termCountsComparison <- c(termCountsComparison, zeroCounts)
# Get list of terms to compare from intersection
# of target and comparison vocabulary
termsToCompare <- intersect(names(termCountsTarget),
names(termCountsComparison))
# Calculate statistics (same as above, but now with vectors!)
a <- termCountsTarget[termsToCompare]
b <- termCountsComparison[termsToCompare]
c <- sum(termCountsTarget)
d <- sum(termCountsComparison)
Expected1 = c * (a+b) / (c+d)
Expected2 = d * (a+b) / (c+d)
t1 <- a * log((a/Expected1) + (a == 0))
t2 <- b * log((b/Expected2) + (b == 0))
logLikelihood <- 2 * (t1 + t2)
# Compare relative frequencies to indicate over/underuse
relA <- a / c
relB <- b / d
# underused terms are multiplied by -1
logLikelihood[relA < relB] <- logLikelihood[relA < relB] * -1

# top terms (overuse in targetCorpus compared to comparisonCorpus)
print(sort(logLikelihood, decreasing=TRUE)[1:50])
# bottom terms (underuse in targetCorpus compared to comparisonCorpus)
sort(logLikelihood, decreasing=FALSE)[1:25]
llTop100 <- sort(logLikelihood, decreasing=TRUE)[1:100]
frqTop100 <- termCountsTarget[names(llTop100)]
frqLLcomparison <- data.frame(llTop100, frqTop100)
#View(frqLLcomparison)
# Number of signficantly overused terms (p < 0.01)
print(sum(logLikelihood > 6.63))

require(wordcloud2)
require(wordcloud)
top50 <- sort(logLikelihood, decreasing = TRUE)[1:50]
top50_df <- data.frame(word = names(top50), count = top50, row.names = NULL)
wordcloud2(top50_df, shuffle = F, size = 0.2)

source("./5-keyword-extraction/calculateLogLikelihood.r")
presidents <- unique(textdata$President)

for (president in presidents) {
    cat("Extracting terms for president", president, "\n")
    selector_logical_idx <- textdata$President == president
    presidentDTM <- targetDTM[selector_logical_idx, ]
    termCountsTarget <- colSums(presidentDTM)
    otherDTM <- targetDTM[!selector_logical_idx, ]
    termCountsComparison <- colSums(otherDTM)
    loglik_terms <- calculateLogLikelihood(termCountsTarget, termCountsComparison)
    top50 <- sort(loglik_terms, decreasing = TRUE)[1:50]
    fileName <- paste0("./5-keyword-extraction/wordclouds/", president, ".pdf")
    pdf(fileName, width = 9, height = 7)
    wordcloud::wordcloud(names(top50),
                         top50,
                         max.words = 50,
                         scale = c(3, .9),
                         colors = RColorBrewer::brewer.pal(8, "Dark2"),
                         random.order = F)
    dev.off()
}