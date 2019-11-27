library(ggthemes)
library(ggplot2)
library(dplyr)

args = commandArgs(trailingOnly=TRUE)
input <- "results1500.data"

df <- read.csv(input, colClasses = character(), sep = ",");
df <- df  %>% group_by(imp, op) %>% summarise(tm = median(t))
g <- ggplot(df, aes(imp, y=tm)) +
  theme_wsj() + scale_fill_wsj() +
  geom_bar(stat="identity", aes(fill=df$op))

g$labels$fill = "GETRF: m = n = 1500"
g
