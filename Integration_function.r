# Load the 'average_toxicity.csv' and 'data_score.csv' files into two data frames.
# Assume that the 'average_toxicity.csv' file contains toxicity information
# and the 'data_score.csv' file contains enrichment scores.
Mean_toxicity <- read.csv("~/paper/ML/DEL/average_toxicity.csv", row.names=1)
enrichment <- read.csv("~/paper/ML/DEL/data_score.csv", row.names=1)

# Assign the 'SMILES' column from 'Mean_toxicity' to a new column 'smiles' in the same data frame.
Mean_toxicity$smiles <- Mean_toxicity$SMILES

# Merge the 'enrichment' and 'Mean_toxicity' data frames based on the 'smiles' column,
# keeping only the rows with matching 'smiles' values in both data frames.
# The result is stored in the new data frame 'df'.
df <- merge(enrichment, Mean_toxicity, all=FALSE)

# Load the 'desiR' library, which contains functions for calculating distance-based similarity indices.
library(desiR)

# Calculate d1 using the 'd.high' function based on 'enrichment_score' from 'df'.
# The cut1, cut2, des.min, des.max, and scale arguments are used as parameters for the function.
df$d1 <- d.high(df$enrichment_score, cut1=0.0037, cut2=73.37, des.min=0, des.max=1, scale=1)

# Calculate d2 using the 'd.low' function based on 'Average' from 'df'.
# The cut1, cut2, des.min, des.max, and scale arguments are used as parameters for the function.
df$d2 <- d.low(df$Average, cut1=4.51, cut2=7.59, des.min=0, des.max=1, scale=1)

# Calculate the overall similarity index 'D' using the 'd.overall' function
# with weights 1 for 'd1' and 'd2'.
df$D <- d.overall(df$d1, df$d2, weights=c(1, 1))

# Create a line plot of the 'D' values sorted in descending order.
# This will show the similarity indices in decreasing order.
plot(rev(sort(df$D)), type="l")

# Print the sorted 'D' values in descending order.
print(rev(sort(df$D)))

# Select the top ten rows with the highest 'D' values and store them in 'top_ten_rows'.
top_ten_rows <- df[order(df$D, decreasing=TRUE), ][1:10, ]

# Extract the 'smiles' column from 'top_ten_rows' and store it in the 'smiles' variable.
smiles <- top_ten_rows$smiles

# Print the 'smiles' values of the top ten rows.
print(smiles)
