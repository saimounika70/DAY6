# DAY6

# 🤖 Task 6 – K-Nearest Neighbors (KNN) Classification

```bash
📌 TASK SUMMARY:

This task involves using the K-Nearest Neighbors (KNN) classification algorithm on the Iris dataset. We'll implement it using Python and Scikit-learn. The key steps include normalizing features, training the model, testing with various K values, evaluating with accuracy and a confusion matrix, and visualizing decision boundaries.

────────────────────────────────────────────────────────────────────────────

🧠 STEP-BY-STEP IMPLEMENTATION:

1. Load the Iris dataset using sklearn.datasets.
2. Convert it into a pandas DataFrame for easy exploration.
3. Normalize all features using StandardScaler — essential since KNN relies on distance.
4. Split the data into training and test sets.
5. Loop through K values from 1 to 10 and store accuracy scores.
6. Select the best K value based on highest accuracy.
7. Evaluate the final model using accuracy and a confusion matrix.
8. Visualize:
   - Pairplots of features
   - Confusion matrix
   - Decision boundaries using only 2 features for 2D visualization

────────────────────────────────────────────────────────────────────────────

💬 INTERVIEW QUESTIONS WITH ANSWERS:

How does KNN work?
→ KNN stores all training samples. To classify a new sample, it calculates distance to every training point, selects K nearest ones, and picks the majority class among them.

How do you choose the right K?
→ Try several values and use the one with highest validation accuracy. Odd K values are preferred to avoid ties.

Why is normalization important?
→ Because KNN is distance-based. If features are on different scales, distance calculations become biased toward higher-range features.

What’s the time complexity of KNN?
→ Training: O(1) (lazy learning). Prediction: O(n * d) where n = number of training samples and d = number of features.

Pros and cons of KNN?
→ Pros: Simple, intuitive, no training time. Cons: Slow predictions, sensitive to noise, performance degrades with high-dimensional data.

Is KNN sensitive to noise?
→ Yes. Outliers can mislead classification, especially with small K. Using a larger K or preprocessing data helps.

How does KNN handle multi-class problems?
→ Directly. It counts votes from neighbors of all classes and picks the majority class. No need for one-vs-rest.

What’s the role of distance metrics in KNN?
→ They define what "nearest" means. Common ones are Euclidean, Manhattan, and Minkowski. The choice affects classification quality.

────────────────────────────────────────────────────────────────────────────

🤔 MY DOUBTS — CLARIFIED:

What is target?
→ In machine learning, "target" means the label we are trying to predict. In Iris, it's the flower species. 0 = setosa, 1 = versicolor, 2 = virginica.

What is pairplot and why use it?
→ Pairplot is a grid of scatter plots between each pair of features, colored by class. Helps us visually detect separation between classes.

What does suptitle with y=1.02 do?
→ suptitle adds a main title to all subplots. y=1.02 moves it slightly above the top plot to prevent overlap.

How do I extract insights from a pairplot?
→ Look for clusters and separability. If classes are clearly grouped, those features help in classification. If they overlap, they may be less useful.

How exactly does KNN work?
→ Store training data. When predicting, compute distances to all training points, find the K closest, and vote for the most common class among them.

Why is K tested only from 1 to 10?
→ It’s a manageable range that shows the trend clearly. Larger K can be used but for Iris, small K is usually enough. We use odd values to prevent tie votes.

What is max accuracy and why argmax, not max?
→ max() gives highest accuracy value. argmax() gives its index, which we use to find the corresponding K value.

What does plt.grid(True) do?
→ Adds horizontal and vertical gridlines to the plot, making it easier to read values and compare data points visually.

What is a confusion matrix and how to understand it?
→ A table showing correct vs incorrect predictions. Diagonal = correct classifications. Off-diagonal = errors. Helps evaluate class-wise performance.

What does visualizing decision boundaries mean?
→ It means showing the regions in feature space where the classifier changes its prediction from one class to another. For KNN, these boundaries are shaped by the training data and chosen K. Usually shown in 2D using only two features.

────────────────────────────────────────────────────────────────────────────

📈 RESULTS SUMMARY:

- Data normalized ✔️
- K tested from 1 to 10 ✔️
- Best K identified using accuracy ✔️
- Accuracy achieved: ~96–98% ✔️
- Confusion matrix plotted ✔️
- Decision boundary visualized in 2D ✔️
- Interview Q&A included ✔️
- Personal doubts documented ✔️
