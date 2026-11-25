# Run in a fresh process. Only fixed program text is executed.
args <- commandArgs(trailingOnly = TRUE)
if (!requireNamespace("CORElearn", quietly = TRUE)) {
    message("Install the R package CORElearn to use core-* methods")
    quit(status = 42)
}
root <- args[1]
method <- args[2]
classification <- args[3] == "classification"
n <- as.integer(args[4])
p <- as.integer(args[5])
if (args[6] != "None") set.seed(as.integer(args[6]))
read_buffer <- function(name, size) {
    readBin(file.path(root, name), numeric(), n = size, size = 8, endian = "little")
}
features <- as.data.frame(matrix(read_buffer("x.bin", n * p), nrow = n, byrow = TRUE))
names(features) <- paste0("f", seq_len(p))
for (i in as.integer(strsplit(args[7], ",", fixed = TRUE)[[1]]) + 1L)
    features[[i]] <- factor(features[[i]])
target <- read_buffer("y.bin", n)
data <- data.frame(target = if (classification) factor(target) else target, features)
call <- list(formula = target ~ ., data = data, estimator = method)
if (file.exists(file.path(root, "cost.bin"))) {
    classes <- nlevels(data$target)
    call$costMatrix <- matrix(read_buffer("cost.bin", classes^2), nrow = classes, byrow = TRUE)
}
if (length(args) > 7L) {
    for (i in seq(8L, length(args), by = 2L)) call[[args[i]]] <- args[i + 1L]
}
weights <- do.call(CORElearn::attrEval, call)
writeBin(as.double(weights[names(features)]), file.path(root, "scores.bin"), endian = "little")
