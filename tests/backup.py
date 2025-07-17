
# Select features and target
numerical = [
    'systolic_bp', 'diastolic_bp', 'cholesterol_level', 'glucose_level',
    'weight_kg', 'age_years', 'physical_activity'
]

X = dfmodel[numerical]
y = dfmodel['heart_disease']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing: Scale numeric features
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical)
])

# Define classifiers
classifiers = [
    (LogisticRegression(max_iter=1000), 'Logistic Regression'),
    (RandomForestClassifier(random_state=42), 'Random Forest'),
    (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), 'XGBoost'),
    (DecisionTreeClassifier(random_state=42), 'Decision Tree')

]

# Prepare results container
results = {
    'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [],
    'F1 Score': [], 'ROC AUC': []
}

# Evaluation loop
for model, name in classifiers:
    pipe = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('classifier', model)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

    results['Model'].append(name)
    results['Accuracy'].append(round(accuracy_score(y_test, y_pred), 4))
    results['Precision'].append(round(precision_score(y_test, y_pred), 4))
    results['Recall'].append(round(recall_score(y_test, y_pred), 4))
    results['F1 Score'].append(round(f1_score(y_test, y_pred), 4))
    results['ROC AUC'].append(round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else 'N/A')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# Display results
results_df = pd.DataFrame(results)
print(results_df)
