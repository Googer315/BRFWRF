#' Balanced Random Forest (BRF) Implementation
#'
#' Trains a Balanced Random Forest model on the provided dataset.
#'
#' @param data A data frame containing features and the target variable.
#' @param target A string specifying the name of the target variable (class column).
#' @param ntree An integer specifying the number of trees in the forest. Defaults to 100.
#' @param mtry An integer specifying the number of variables to consider at each split. Defaults to sqrt(number of predictors).
#' @param nodesize An integer specifying the minimum number of samples required to split a node. Defaults to 1.
#' @param seed An integer for reproducibility. Defaults to 123.
#'
#' @return An object of class \code{brf} containing the trained forest and related information.
#' @export
#'
#' @examples
#' \dontrun{
#' data <- data.frame(
#'   feature1 = rnorm(1000),
#'   feature2 = rnorm(1000),
#'   class = c(rep("Majority", 900), rep("Minority", 100))
#' )
#' model_brf <- balanced_random_forest(data, target = "class", ntree = 10, mtry = 1, nodesize = 2, seed = 49)
#' }
balanced_random_forest <- function(data, target, ntree = 100, mtry = NULL,
                                   nodesize = 1, seed = 123) {
  # Load necessary library
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop("Package 'dplyr' is required but not installed.")
  }

  # Set seed for reproducibility
  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Ensure target is a factor
  data[[target]] <- as.factor(data[[target]])
  y_full <- data[[target]]

  # Identify class distribution
  class_counts <- table(y_full)
  minority_class <- names(which.min(class_counts))
  majority_class <- names(which.max(class_counts))

  minority_count <- class_counts[[minority_class]]
  majority_count <- class_counts[[majority_class]]

  cat("Original Class Distribution:\n")
  print(class_counts)

  cat("\nMinority Class:", minority_class, "with", minority_count, "instances\n")
  cat("Majority Class:", majority_class, "with", majority_count, "instances\n")

  # Separate minority and majority classes
  data_minority <- data[y_full == minority_class, , drop = FALSE]
  data_majority <- data[y_full != minority_class, , drop = FALSE]

  # Set uniform class weights (no weighting as per BRF)
  cw <- rep(1, length(class_counts))
  names(cw) <- names(class_counts)

  # Determine default mtry if not provided
  if (is.null(mtry)) {
    mtry <- floor(sqrt(ncol(data) - 1))
  }

  # Initialize a list to store each tree
  forest <- vector("list", ntree)

  # Initialize a matrix to store predictions from each tree
  predictions_matrix <- matrix(NA, nrow = nrow(data), ncol = ntree)

  for (i in 1:ntree) {
    cat("\nTraining tree", i, "of", ntree, "\n")

    # Step 1: Create a balanced subset for this tree
    balanced_subset <- create_balanced_subset(
      data = data,
      target = target,
      majority_multiplier = 1,          # Ensure equal sampling
      majority_sample_size = NULL,      # Not used; defaults to multiplier
      replace_majority = TRUE,          # Sample with replacement
      seed = seed + i                    # Different seed for each tree for variability
    )

    # Extract predictors and target from the balanced subset
    x_balanced <- balanced_subset[, setdiff(names(balanced_subset), target), drop = FALSE]
    y_balanced <- balanced_subset[[target]]

    # Step 2: Train a single decision tree on the balanced subset
    tree <- grow_tree(
      x = x_balanced,
      y = y_balanced,
      class.weights = cw,
      mtry = mtry,
      nodesize = nodesize
    )

    # Store the trained tree in 'forest'
    forest[[i]] <- tree

    # Step 3: Make predictions on the entire dataset using the trained tree
    predictions_matrix[, i] <- predict_tree(tree, newdata = data, type = "class")
  }

  # Step 4: Aggregate predictions by majority voting
  final_predictions <- apply(predictions_matrix, 1, function(x) {
    # Handle ties by randomly selecting one of the top classes
    tab <- sort(table(x), decreasing = TRUE)
    top_count <- tab[1]
    top_classes <- names(tab)[tab == top_count]
    if(length(top_classes) == 1){
      return(top_classes)
    } else {
      return(sample(top_classes, 1))
    }
  })

  # Add final predictions to the dataset
  data$Final_Prediction <- factor(final_predictions, levels = levels(y_full))

  # Create BRF model object
  brf_model <- list(
    forest = forest,            # List of trees
    y = y_full,                 # Factor vector of target variable
    classes = levels(y_full)    # Class levels
  )

  class(brf_model) <- "brf"

  return(brf_model)
}
