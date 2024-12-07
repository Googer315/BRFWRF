#' Create a Balanced Subset for BRF
#'
#' Creates a balanced subset of the data by sampling equally from the minority and majority classes.
#'
#' @param data A data frame containing features and the target variable.
#' @param target A string specifying the name of the target variable (class column).
#' @param majority_multiplier A numeric value indicating how many times to sample the majority class relative to the minority class. Defaults to 1.
#' @param majority_sample_size An integer specifying the exact number of majority class samples to draw. If provided, overrides `majority_multiplier`.
#' @param replace_majority Logical indicating whether to sample the majority class with replacement. Defaults to TRUE.
#' @param seed An integer for reproducibility. Defaults to NULL.
#'
#' @return A data frame containing a balanced subset of the original data.
#' @export
#'
#' @examples
#' \dontrun{
#' data <- data.frame(
#'   feature1 = rnorm(1000),
#'   feature2 = rnorm(1000),
#'   class = c(rep("Majority", 900), rep("Minority", 100))
#' )
#' balanced_subset <- create_balanced_subset(data, target = "class")
#' }
create_balanced_subset <- function(data, target, majority_multiplier = 1,
                                   majority_sample_size = NULL,
                                   replace_majority = TRUE, seed = NULL) {
  # Load necessary library
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop("Package 'dplyr' is required but not installed.")
  }

  # Set seed for reproducibility if provided
  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Ensure the target variable exists
  if (!(target %in% names(data))) {
    stop("Target variable not found in the dataset.")
  }

  # Ensure the target variable is a factor
  data[[target]] <- as.factor(data[[target]])

  # Calculate class distribution
  class_counts <- table(data[[target]])
  cat("Original Class Distribution:\n")
  print(class_counts)

  # Identify minority and majority classes
  minority_class <- names(which.min(class_counts))
  majority_class <- names(which.max(class_counts))

  minority_count <- class_counts[[minority_class]]
  majority_count <- class_counts[[majority_class]]

  cat("\nMinority Class:", minority_class, "with", minority_count, "instances\n")
  cat("Majority Class:", majority_class, "with", majority_count, "instances\n")

  if (!is.null(majority_sample_size)) {
    # Use the user-specified absolute number
    majority_sample_size_final <- majority_sample_size
  } else {
    # Calculate based on the multiplier
    majority_sample_size_final <- round(majority_multiplier * minority_count)
  }

  cat("\nSampling Majority Class:\n")
  cat("Number of Majority Class Samples to Draw:", majority_sample_size_final, "\n")

  # Validate sampling size
  if (majority_sample_size_final > minority_count && !replace_majority) {
    stop("majority_sample_size_final exceeds the number of available majority class samples and replace_majority is FALSE.")
  }

  # Draw samples from the minority and majority class
  minority_samples <- data %>%
    dplyr::filter((!!sym(target)) == minority_class) %>%
    dplyr::sample_n(size = minority_count, replace = TRUE)  # Bootstrap sample from minority

  majority_samples <- data %>%
    dplyr::filter((!!sym(target)) == majority_class) %>%
    dplyr::sample_n(size = majority_sample_size_final, replace = replace_majority)  # Sample majority

  # Combine the samples to create a balanced subset
  balanced_subset <- dplyr::bind_rows(minority_samples, majority_samples)

  # Shuffle the rows
  balanced_subset <- balanced_subset %>%
    dplyr::sample_frac(1) %>%
    dplyr::mutate(row_id = row_number()) %>%
    dplyr::arrange(row_id) %>%
    dplyr::select(-row_id)

  # Display the new class distribution
  # cat("\nBalanced Subset Class Distribution:\n")
  # print(table(balanced_subset[[target]]))

  return(balanced_subset)
}
