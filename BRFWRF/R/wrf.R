#' @title Weighted Random Forest (WRF)
#' @description Train a Weighted Random Forest classifier on a (possibly) imbalanced dataset. Trains a Weighted Random Forest (WRF) model on a dataset. This WRF implements cost-sensitive learning through class weighting to handle imbalanced classification problems.
#' @param x A data frame or matrix of predictor variables.
#' @param y A factor vector of class labels.
#' @param class.weights A named numeric vector specifying weights for each class. If NULL, defaults to inverse frequency weights.
#' @param ntree Integer, the number of trees to grow in the forest.
#' @param mtry Integer, number of variables randomly sampled at each split. Defaults to \code{floor(sqrt(ncol(x)))}.
#' @param nodesize Minimum size of terminal nodes.
#' @param sampsize Number or vector of sample sizes to draw for each tree. Defaults to \code{floor(0.632 * nrow(x))}.
#' @details
#' This function implements a Weighted Random Forest algorithm. Class weights are integrated into:
#' \itemize{
#'   \item The Gini splitting criterion, making it cost-sensitive.
#'   \item The leaf node predictions, which use a weighted majority vote.
#' }
#' Final predictions are obtained by aggregating weighted probabilities across all trees.
#'
#' The OOB (out-of-bag) error is computed during training by predicting samples not used in constructing each tree.
#'
#' @return An object of class \code{"wrf"} containing:
#' \item{class.weights}{The class weights used.}
#' \item{oob.error}{The OOB error estimate.}
#' \item{oob.votes}{A matrix of accumulated OOB votes for each sample.}
#'
#' @export
#'
#' @examples
#' \dontrun{
#' data(iris)
#' setosa_vs_others <- iris$Species
#' levels(setosa_vs_others) <- c("minority", "majority", "majority")
#'
#' model <- wrf(
#'   x = iris[, -5],
#'   y = setosa_vs_others,
#'   ntree = 50,
#'   class.weights = c("minority" = 5, "majority" = 1)
#' )
#'
#' print(model$oob.error)
#' }
wrf <- function(x, y,
                class.weights = NULL,
                ntree = 100,
                mtry = NULL,
                nodesize = 1,
                sampsize = NULL) {
  if (!is.factor(y)) stop("y must be a factor")

  # Set default class weights if needed
  class.weights <- check_class_weights(y, class.weights)

  # Set default mtry
  if (is.null(mtry)) {
    mtry <- floor(sqrt(ncol(x)))
  }

  # Set default sampsize
  n <- nrow(x)
  if (is.null(sampsize)) {
    sampsize <- floor(0.632 * n)
  }

  # Prepare storage
  n <- nrow(x)
  forest <- vector("list", ntree)

  # Track OOB predictions
  # We'll store votes from OOB samples for each tree and aggregate at the end
  oob_votes <- matrix(0, nrow = n, ncol = length(levels(y)))
  colnames(oob_votes) <- levels(y)
  oob_count <- integer(n)

  # Grow each tree
  for (i in seq_len(ntree)) {
    # Bootstrap sample
    samp_idx <- sample(seq_len(n), size = sampsize, replace = TRUE)
    oob_idx <- setdiff(seq_len(n), samp_idx)

    # Grow the tree
    tree <- grow_tree(x = x[samp_idx, , drop = FALSE],
                      y = y[samp_idx],
                      class.weights = class.weights,
                      mtry = mtry,
                      nodesize = nodesize)

    forest[[i]] <- tree

    # Predict OOB samples with this tree
    if (length(oob_idx) > 0) {
      preds_proba <- predict_tree(tree, x[oob_idx, , drop = FALSE], type = "prob")
      # preds_proba is a matrix of probabilities for each class
      # Convert to weighted votes. Actually, we already have them as a proportion of weighted votes.
      # Here we just accumulate weighted votes directly:
      # Since predict_tree "prob" is normalized weights, multiply by sampsize to get "votes"
      # or we can consider them as weights directly. For consistency, we can treat them as votes:
      for (c in levels(y)) {
        oob_votes[oob_idx, c] <- oob_votes[oob_idx, c] + preds_proba[, c]
      }
      oob_count[oob_idx] <- oob_count[oob_idx] + 1
    }
  }

  # Compute OOB predictions and error
  # Normalize OOB votes by count
  valid_idx <- oob_count > 0
  oob_proba <- oob_votes
  for (i in which(valid_idx)) {
    oob_proba[i, ] <- oob_votes[i, ] / sum(oob_votes[i, ])
  }
  oob_preds <- levels(y)[max.col(oob_proba)]
  oob_preds[!valid_idx] <- NA
  oob_preds <- factor(oob_preds, levels = levels(y))

  oob_error <- mean(oob_preds[valid_idx] != y[valid_idx], na.rm = TRUE)

  model <- list(class.weights = class.weights,
                oob.error = oob_error,
                oob.votes = oob_votes,
                forest = forest,
                y = y)
  class(model) <- "wrf"
  model
}
