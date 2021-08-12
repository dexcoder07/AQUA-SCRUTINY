def model(params):
    # Importing Libraries and Data Sets
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    df1 = pd.read_csv('water_potability.csv')
    l1 = []
    l2 = []

    ### Analysing the Dataset informations
    df1.info()
    df1.describe(include='all')
    *All
    the
    columns in the
    dataset
    are
    continuous
    variable.

    ### Checking for NULL values in column
    df1.isnull().sum()
    plt.figure(figsize=(12, 7))
    sns.heatmap(df1.isnull(), yticklabels=False, cbar=False, cmap='viridis')

    ### There are some null values in ph, Sulfate, and Trihalomethanes Column
    ### Sulfate has more number of missing values then ph and trihalomethanes
    ## Removing the NULL values
    df1.apply(lambda x: x.fillna(x.median(), inplace=True))

    # Data Analysis and Data Cleaning
    ## Analysing the Ph column
    df1.ph.mean()
    df1.ph.isnull().sum()
    ### There are 491 Missing values in ph column
    plt.figure(figsize=(12, 7))
    sns.kdeplot(df1.ph, fill=True, color='r')
    plt.figure(figsize=(12, 7))
    sns.boxplot(df1.ph)
    ##### Removing outliers
    df1.ph.describe()
    # finding the Q1(25 percentile) and Q3(75 percentile)
    IQR = df1["ph"].quantile(0.75) - df1["ph"].quantile(0.25)
    # defining max and min limits
    max_limit = df1["ph"].quantile(0.75) + (1.5 * IQR)
    min_limit = df1["ph"].quantile(0.25) - (1.5 * IQR)
    df1 = df1[(df1['ph'] > min_limit) & (df1['ph'] < max_limit)]
    plt.figure(figsize=(12, 7))
    sns.kdeplot(df1.ph, fill=True, color='r')
    plt.figure(figsize=(12, 7))
    sns.boxplot(df1.ph)
    plt.figure(figsize=(12, 7))
    sns.violinplot(df1.ph)

    ## Analysing Hardness Column
    df1.Hardness.mean()
    df1.isnull().sum()
    plt.figure(figsize=(12, 7))
    sns.histplot(df1.Hardness, kde=True)
    df1.Hardness.describe()
    hard_iqr = (df1['Hardness'].quantile(0.75) - df1['Hardness'].quantile(0.25))
    print(hard_iqr)
    df1 = df1[(df1['Hardness'] > (df1['Hardness'].quantile(0.25) - 1.5 * hard_iqr)) &
              (df1['Hardness'] < (df1['Hardness'].quantile(0.75) + 1.5 * hard_iqr))]
    plt.figure(figsize=(12, 7))
    sns.histplot(df1.Hardness, kde=True)
    plt.figure(figsize=(12, 7))
    sns.violinplot(df1.Hardness)
    df1.isnull().sum()

    # Analysing Solids
    df1.Solids.mean()
    df1.info()
    df1.Solids.describe()
    df1.isnull().sum()
    plt.figure(figsize=(12, 7))
    sns.histplot(df1.Solids, kde=True)
    sns.boxplot(df1.Solids)
    solid_iqr = (df1['Solids'].quantile(0.75) - df1['Solids'].quantile(0.25))
    print(solid_iqr)
    df1 = df1[(df1['Solids'] > (df1['Solids'].quantile(0.25) - 1.5 * solid_iqr)) &
              (df1['Solids'] < (df1['Solids'].quantile(0.75) + 1.5 * solid_iqr))]
    plt.figure(figsize=(12, 7))
    sns.boxplot(df1.Solids)
    plt.figure(figsize=(12, 7))
    sns.histplot(df1.Solids, kde=True)
    plt.figure(figsize=(12, 7))
    sns.violinplot(df1.Solids)

    ## Analysing Chloramines
    df1.Chloramines.mean()
    df1.info()
    df1.isnull().sum()
    plt.figure(figsize=(12, 7))
    sns.histplot(df1.Chloramines, kde=True, fill=True)
    plt.figure(figsize=(12, 7))
    sns.boxplot(df1.Chloramines)
    df1['Chloramines'].describe()
    Chloramines_iqr = (df1['Chloramines'].quantile(0.75) - df1['Chloramines'].quantile(0.25))
    print(Chloramines_iqr)

    df1 = df1[(df1['Chloramines'] > (df1['Chloramines'].quantile(0.25) - 1.5 * Chloramines_iqr)) &
              (df1['Chloramines'] < (df1['Chloramines'].quantile(0.75) + 1.5 * Chloramines_iqr))]
    df1.info()
    plt.figure(figsize=(12, 7))
    sns.boxplot(df1.Chloramines)
    plt.figure(figsize=(12, 7))
    sns.violinplot(df1.Chloramines)

    # Analysing Sulphate column
    df1.Sulfate.mean()
    df1.info()
    df1.Sulfate.describe()
    (df1.Sulfate.isnull().sum() / 2308) * 100
    df1.Sulfate.fillna(np.mean(df1.Sulfate), inplace=True)
    df1.Sulfate.mean()
    (df1.Sulfate.isnull().sum() / 2308) * 100
    plt.figure(figsize=(12, 7))
    sns.kdeplot(df1.Sulfate, fill=True)
    sulfate_iqr = (df1['Sulfate'].quantile(.75) - df1['Sulfate'].quantile(.25))
    print(sulfate_iqr)

    df1 = df1[(df1['Sulfate'] > (df1['Sulfate'].quantile(0.25) - 1.5 * sulfate_iqr)) &
              (df1['Sulfate'] < (df1['Sulfate'].quantile(0.75) + 1.5 * sulfate_iqr))]
    plt.figure(figsize=(12, 7))
    sns.histplot(df1.Sulfate, kde=True)
    p = np.log(df1.Sulfate)
    plt.figure(figsize=(12, 7))
    sns.kdeplot(p, fill=True, color='black')

    # Analysing Conductivity
    df1.Conductivity.mean()
    df1.Conductivity.describe()
    df1.Conductivity.isnull().sum()
    plt.figure(figsize=(12, 7))
    sns.histplot(df1.Conductivity, kde=True)
    sns.boxplot(df1.Conductivity)
    conduct_iqr = (df1['Conductivity'].quantile(0.75) - df1['Conductivity'].quantile(0.25))
    print(conduct_iqr)

    df1 = df1[(df1['Conductivity'] > (df1['Conductivity'].quantile(0.25) - 1.5 * conduct_iqr)) &
              (df1['Conductivity'] < (df1['Conductivity'].quantile(0.75) + 1.5 * conduct_iqr))]
    plt.figure(figsize=(12, 7))
    sns.boxplot(df1.Conductivity)

    # Analysing Organic carbon
    df1.Organic_carbon.mean()
    df1.Organic_carbon.describe()
    plt.figure(figsize=(12, 7))
    sns.boxplot(df1.Organic_carbon)
    Organic_carbon_iqr = (df1['Organic_carbon'].quantile(0.75) - df1['Organic_carbon'].quantile(0.25))
    print(Organic_carbon_iqr)

    df1 = df1[(df1['Organic_carbon'] > (df1['Organic_carbon'].quantile(0.25) - 1.5 * Organic_carbon_iqr)) &
              (df1['Organic_carbon'] < (df1['Organic_carbon'].quantile(0.75) + 1.5 * Organic_carbon_iqr))]
    plt.figure(figsize=(12, 7))
    sns.boxplot(df1.Organic_carbon)

    # Analysing Trihalomethanes
    df1.Trihalomethanes.mean()
    df1.Trihalomethanes.describe()
    df1.Trihalomethanes.isnull().sum()
    plt.figure(figsize=(12, 7))
    sns.histplot(df1.Trihalomethanes, kde=True)
    lower_limit_t = (df1.Trihalomethanes.mean() - (2 * df1.Trihalomethanes.std()))
    upper_limit_t = (df1.Trihalomethanes.mean() + (2 * df1.Trihalomethanes.std()))
    df1 = df1[(df1.Trihalomethanes > lower_limit_t) & (df1.Trihalomethanes < upper_limit_t)]
    plt.figure(figsize=(12, 7))
    sns.boxplot(df1.Trihalomethanes)

    # Analysing Turbidity
    df1.Turbidity.mean()
    df1.Turbidity.describe()
    df1.Turbidity.isnull().sum()
    plt.figure(figsize=(12, 7))
    sns.boxplot(df1.Turbidity)
    plt.figure(figsize=(12, 7))
    sns.histplot(df1.Turbidity, kde=True, fill=True)

    # Analysing Potability
    plt.figure(figsize=(12, 7))
    sns.countplot(df1.Potability)
    plt.figure(figsize=(12, 7))
    sns.heatmap(df1.corr(), annot=True)
    plt.figure(figsize=(10, 7))
    sns.pairplot(df1, hue='Potability')

    # Train Test Split
    df1.info()
    X = df1.iloc[:, :-1]
    Y = df1.iloc[:, -1]
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    # Scaling the Features
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(x_train)
    sc.transform(x_train)
    sc.transform(x_test)

    # Random Forest
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=100, max_depth=90)
    rfc.fit(x_train, y_train)
    rfc_ypred = rfc.predict(params)

    return rfc_ypred
    ### Random Forest Gives the best result as compared to other algorithms with the accuracy of 64.59%
