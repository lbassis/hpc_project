library(ggthemes)
library(ggplot2)
library(dplyr)

args = commandArgs(trailingOnly=TRUE)
input <- "error.txt"#"results1500.data"

#device.off();
df <- read.csv(input, colClasses = character(), sep = ",");
#df <- df  %>% group_by(imp, op) %>% summarise(tm = median(t))
g <- ggplot(df, aes(imp, y=us)) +
  theme_wsj() + scale_fill_wsj() +
  geom_bar(stat="identity", aes(fill=df$op))

g$labels$fill = ""#"PGETRF: m = n = 300, 4 processes"
ggsave("output.pdf", plot=g)
