#-----------------------
# Utility Functions
#-----------------------

# Validate and set default class weights
#' @title Check and Set Class Weights
#' @description Internal utility function to validate and set default class weights.
#' @param y The factor of class labels.
#' @param class.weights A named numeric vector of class weights or NULL.
#' @return A named numeric vector of class weights.
#' @keywords internal
check_class_weights <- function(y, class.weights) {
  levs <- levels(y)
  if (is.null(class.weights)) {
    # Default: Inverse frequency
    tbl <- table(y)
    cw <- 1 / tbl
    cw <- cw / sum(cw) * length(cw)
    names(cw) <- names(tbl)
    class.weights <- cw
  } else {
    if (!all(levs %in% names(class.weights))) {
      stop("All class weights must be provided and named by class.")
    }
    if (any(class.weights <= 0)) {
      stop("class.weights must be positive")
    }
    class.weights <- class.weights[levs]
  }
  class.weights
}


# Calculate weighted Gini impurity
#' @title Weighted Gini Calculation
#' @description Internal function to compute the weighted Gini impurity.
#' @param y A factor of class labels.
#' @param class.weights Named numeric vector of class weights.
#' @return A numeric value of weighted Gini impurity.
#' @keywords internal
weighted_gini <- function(y, class.weights) {
  counts <- table(y)
  total <- sum(counts)
  p <- counts / total
  # Weighted gini = 1 - sum(w_c * p_c^2)
  1 - sum(class.weights * (p^2))
}


# Create a leaf node: store prediction, counts, and weights
#' @title Create a Leaf Node
#' @description Internal function to create a leaf node with weighted majority vote prediction.
#' @param y A factor of class labels.
#' @param class.weights Named numeric vector of class weights.
#' @return A list representing a leaf node.
#' @keywords internal
make_leaf <- function(y, class.weights) {
  counts <- table(y)
  w_counts <- counts * class.weights[names(counts)]
  prediction <- names(which.max(w_counts))
  leaf <- list(
    prediction = prediction,
    counts = counts,
    class.weights = class.weights,
    leaf = TRUE
  )
  leaf
}


# Grow a single decision tree with weighted Gini splits
#' @title Grow a Single Decision Tree
#' @description Internal function that grows a single decision tree using weighted Gini splits.
#' @param x A matrix or data frame of predictors.
#' @param y A factor vector of class labels.
#' @param class.weights Named numeric vector of class weights.
#' @param mtry Number of features to sample for splits.
#' @param nodesize Minimum node size.
#' @return A nested list representing the decision tree.
#' @keywords internal
grow_tree <- function(x, y, class.weights, mtry, nodesize) {

  build_node <- function(x, y) {
    # Stopping conditions
    if (length(y) <= nodesize || length(unique(y)) == 1) {
      return(make_leaf(y, class.weights))
    }

    parent_gini <- weighted_gini(y, class.weights)
    features <- sample(seq_len(ncol(x)), min(mtry, ncol(x)))
    best_gain <- 0
    best_split <- NULL

    for (f in features) {
      x_col <- x[, f]
      split_candidates <- sort(unique(x_col))
      if (length(split_candidates) == 1) next

      # Consider midpoints
      for (s_i in seq_len(length(split_candidates)-1)) {
        split_point <- (split_candidates[s_i] + split_candidates[s_i+1]) / 2
        left_idx <- x_col <= split_point
        right_idx <- !left_idx
        if (!any(left_idx) || !any(right_idx)) next

        left_gini <- weighted_gini(y[left_idx], class.weights)
        right_gini <- weighted_gini(y[right_idx], class.weights)

        nl <- sum(left_idx)
        nr <- sum(right_idx)
        n <- nl + nr
        weighted_child_gini <- (nl/n)*left_gini + (nr/n)*right_gini
        gain <- parent_gini - weighted_child_gini
        if (gain > best_gain) {
          best_gain <- gain
          best_split <- list(var = f, point = split_point)
        }
      }
    }

    if (is.null(best_split)) {
      return(make_leaf(y, class.weights))
    }

    left_idx <- x[, best_split$var] <= best_split$point
    right_idx <- !left_idx

    left_child <- build_node(x[left_idx, , drop = FALSE], y[left_idx])
    right_child <- build_node(x[right_idx, , drop = FALSE], y[right_idx])

    list(
      split_var = best_split$var,
      split_point = best_split$point,
      left_child = left_child,
      right_child = right_child,
      leaf = FALSE
    )
  }

  build_node(x, y)
}


# Predict from a single tree
#' @title Predict with a Single Tree
#' @description Internal function to predict classes or probabilities from a single tree.
#' @param tree A tree object as returned by \code{grow_tree}.
#' @param newdata Data frame or matrix of observations.
#' @param type "class" or "prob".
#' @return If \code{type = "class"}, a factor vector of predictions. If \code{type = "prob"}, a matrix of class probabilities.
#' @keywords internal
predict_tree <- function(tree, newdata, type = c("class","prob")) {
  type <- match.arg(type)

  traverse <- function(obs, node) {
    if (isTRUE(node$leaf)) {
      return(node)
    }
    if (obs[node$split_var] <= node$split_point) {
      traverse(obs, node$left_child)
    } else {
      traverse(obs, node$right_child)
    }
  }

  nodes <- lapply(seq_len(nrow(newdata)), function(i) traverse(newdata[i,], tree))

  y_levels <- names(nodes[[1]]$counts)
  if (type == "class") {
    preds <- sapply(nodes, `[[`, "prediction")
    factor(preds, levels = y_levels)
  } else {
    # Compute probabilities based on weighted counts at each leaf
    proba <- matrix(0, nrow = length(nodes), ncol = length(y_levels))
    colnames(proba) <- y_levels
    for (i in seq_along(nodes)) {
      leaf_counts <- nodes[[i]]$counts
      cw <- nodes[[i]]$class.weights
      w_counts <- leaf_counts * cw[names(leaf_counts)]
      p <- w_counts / sum(w_counts)
      proba[i, names(p)] <- p
    }
    proba
  }
}
