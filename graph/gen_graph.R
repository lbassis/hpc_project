library(ggplot2)

args = commandArgs(trailingOnly=TRUE)

input <- "perf_ddot_extra.data"
input <- "perf_no_version.data"
x_var <- "n"
y_var <- "Mflops"
title <- "Default title"
output <- "output.pdf"

if (length(args) < 5) {
  stop("Insert input file, graph title, x axis variable and y axis variable", call.=FALSE)
} else {
  input <- args[1]
  title <- args[2]
  x_var <- args[3]
  y_var <- args[4]
  output <- args[5]
}

df <- read.csv(input, colClasses = character());

if ("version" %in% colnames(df)) {
  plot <- ggplot(df, aes(x = get(x_var), y = get(y_var), col = df$version)) +
        labs(title = title, x = x_var, y = y_var, color="Version") +
        geom_line() + geom_point()
} else {
  plot <- ggplot(df, aes(x = get(x_var), y = get(y_var))) +
    labs(title = title, x = x_var, y = y_var) +
    geom_line() + geom_point()
}
ggsave(output, plot=plot)
