options(stringsAsFactors = FALSE)
require(quanteda)
library(dplyr)
library(ggplot2)
library(reshape2)

textdata <- read.csv("./4-frequency-analysis/sotu_new.csv", sep = ",", encoding = "UTF-8")
# we add some more metadata columns to the data frame
textdata$year <- substr(textdata$Date, 0, 4)
textdata$decade <- paste0(substr(textdata$Date, 0, 3), "0")

sotu_corpus <- corpus(textdata$Paragraph, docnames = textdata$doc_id)
# Build a dictionary of lemmas
lemma_data <- read.csv("./4-frequency-analysis/baseform_en.tsv", encoding = "UTF-8")
# Create a DTM
corpus_tokens <- sotu_corpus %>%
    tokens(remove_punct = TRUE, remove_numbers = TRUE, remove_symbols = TRUE) %>%
    tokens_tolower() %>%
    tokens_replace(lemma_data$inflected_form,
                   lemma_data$lemma,
                   valuetype = "fixed") %>%
    tokens_remove(pattern = stopwords())

print(paste0("1: ", substr(paste(corpus_tokens[1],collapse = " "), 0, 400), '...'))

DTM <- corpus_tokens %>%
    dfm()

terms_to_observe <- c("nation", "war", "god", "terror", "security")
DTM_reduced <- as.matrix(DTM[, terms_to_observe])

counts_per_decade <- aggregate(DTM_reduced,
by = list(decade = textdata$decade), sum)

# give x and y values beautiful names
decades <- counts_per_decade$decade
frequencies <- counts_per_decade[, terms_to_observe]
# plot multiple frequencies
matplot(decades, frequencies, type = "l")
# add legend to the plot
l <- length(terms_to_observe)
legend('topleft', legend = terms_to_observe, col=1:l, text.col = 1:l, lty = 1:l)

# English Opinion Word Lexicon by Hu et al. 2004
positive_terms_all <- readLines("./4-frequency-analysis/positive-words.txt")
negative_terms_all <- readLines("./4-frequency-analysis/negative-words.txt")
# AFINN sentiment lexicon by Nielsen 2011
afinn_terms <- read.csv("./4-frequency-analysis/AFINN-111.txt", header = F, sep = "\t")
positive_terms_all <- afinn_terms$V1[afinn_terms$V2 > 0]
negative_terms_all <- afinn_terms$V1[afinn_terms$V2 < 0]

positive_terms_in_suto <- intersect(colnames(DTM), positive_terms_all)
counts_positive <- rowSums(DTM[, positive_terms_in_suto])
negative_terms_in_suto <- intersect(colnames(DTM), negative_terms_all)
counts_negative <- rowSums(DTM[, negative_terms_in_suto])

counts_all_terms <- rowSums(DTM)
relative_sentiment_frequencies <- data.frame(
    positive = counts_positive / counts_all_terms,
    negative = counts_negative / counts_all_terms
)

sentiments_per_president <- relative_sentiment_frequencies %>%
    mutate(president = textdata$President, year = textdata$year) %>%
    group_by(textdata$President) %>%
    summarise(mean_positive = mean(positive, na.rm = TRUE),
              mean_negative = mean(negative, na.rm = TRUE))

head(sentiments_per_president)
tail(sentiments_per_president)

require(tidyr)
df <- sentiments_per_president %>% pivot_longer(!president)
require(ggplot2)
ggplot(data = df, aes(x = president, y = value, fill = name)) +
geom_bar(stat="identity", position=position_dodge()) + coord_flip()

ggplot(data = df, aes(x = reorder(president, value, head, 1), y = value, fill = name)) +
geom_bar(stat="identity", position=position_dodge()) + coord_flip()

ggplot(data = df, aes(x = reorder(president, value, tail, 1), y = value, fill = name)) +
geom_bar(stat="identity", position=position_dodge()) + coord_flip()

terms_to_observe <- c("war", "peace", "health", "terror", "islam",
                      "threat", "security", "conflict", "job",
                      "economy", "indian", "afghanistan", "muslim",
                      "god", "world", "territory", "frontier", "north",
                      "south", "black", "racism", "slavery", "iran")

DTM_reduced <- as.matrix(DTM[, terms_to_observe])

counts_per_decade <- aggregate(DTM_reduced,
    by = list(decade = textdata$decade),
    sum)

#rownames(DTM_reduced) <- ifelse(as.integer(textdata$year) %% 2 == 0, textdata$year, "")
ts_events_dendro <- as.dendrogram(hclust(d = dist(x = counts_per_decade[,-1]),method = "single"))
ts_events_order <- order.dendrogram(ts_events_dendro)

ts_events <- counts_per_decade |> as_tibble() |> pivot_longer(-decade)
ts_events$name <- factor(x = ts_events$name,
                         levels = (colnames(counts_per_decade)[-1])[ts_events_order],
                         ordered = TRUE)

heatmap_plot <- ggplot(data = ts_events, aes(x = decade, y = name)) +
    geom_tile(aes(fill = value)) +
    scale_fill_distiller(palette = "BuGn", direction = 1) +
    theme(,
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

# Preview the heatmap
print(heatmap_plot)
