#' @title Predict Method for Weighted Random Forest / Balanced Random Forest
#' @description Generates predictions (classes or probabilities) for new data using a fitted \code{wrf} / \code{brf} model.
#' @param object A \code{wrf} or a \code{brf} model object.
#' @param newdata A data frame or matrix of new observations.
#' @param type Character, "class" for predicted classes or "prob" for class probabilities.
#'
#' @return If \code{type = "class"}, a factor vector of predicted classes.
#' If \code{type = "prob"}, a matrix of class probabilities.
#' @export
#'
#' @examples
#' \dontrun{
#' # Using the wrf model from the previous example:
#' preds <- predict(model, newdata = iris[, -5], type = "class")
#' head(preds)
#' }
predict.BRFWRF <- function(object, newdata, type = c("class", "prob")) {
  type <- match.arg(type)

  ntree <- length(object$forest)
  y_levels <- levels(object$y)

  # Accumulate weighted votes across trees
  votes <- matrix(0, nrow = nrow(newdata), ncol = length(y_levels))
  colnames(votes) <- y_levels

  for (tree in object$forest) {
    tree_proba <- predict_tree(tree, newdata, type = "prob")
    # Accumulate probabilities (which represent normalized weighted votes)
    votes <- votes + tree_proba
  }
  # Average over trees
  votes <- votes / ntree

  if (type == "class") {
    pred_class <- y_levels[max.col(votes)]
    factor(pred_class, levels = y_levels)
  } else {
    votes
  }
}
