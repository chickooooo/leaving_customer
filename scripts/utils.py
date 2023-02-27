import joblib
import pandas as pd


class Setup:
    """Loads the model and required dependencies. Prepares the model for making predictions
    """

    # columns of original data
    __original_columns = [
        'RowNumber',
        'CustomerId',
        'Surname',
        'CreditScore',
        'Geography',
        'Gender',
        'Age',
        'Tenure',
        'Balance',
        'NumOfProducts',
        'HasCrCard',
        'IsActiveMember',
        'EstimatedSalary'
    ]

    # numerical features
    __numerical_features = [
        "CreditScore",
        "Age",
        "Balance",
        "EstimatedSalary",
    ]

    # ordinal features
    __ordinal_features = [
        "Tenure",
        "NumOfProducts",
    ]

    # categorical features
    __categorical_features = [
        "Geography",
    ]

    # binary features
    __binary_features = [
        "Gender",
        "HasCrCard",
        "IsActiveMember",
    ]

    # order of columns on which model was trained
    __column_order = [
        'CreditScore',
        'Age',
        'Balance',
        'EstimatedSalary',
        'Tenure',
        'NumOfProducts',
        'Geography_France',
        'Geography_Germany',
        'Geography_Spain',
        'Gender',
        'HasCrCard',
        'IsActiveMember'
    ]


    def __init__(self) -> None:
        # load model
        self.__model = joblib.load("./models/decision_tree.pkl")

        # load scalers
        self.__robust_scaler = joblib.load("./models/robust_scaler.pkl")
        self.__min_max_scaler = joblib.load("./models/min_max_scaler.pkl")


    def __prepare_data(self, X: pd.DataFrame) -> pd.DataFrame:
        # scaling numerical data
        numerical_data = X[self.__numerical_features].copy()
        numerical_scaled = self.__robust_scaler.transform(numerical_data)
        numerical_scaled = pd.DataFrame(numerical_scaled, index=numerical_data.index, columns=numerical_data.columns)
        # scaling ordinal data
        oridnal_data = X[self.__ordinal_features].copy()
        ordinal_scaled = self.__min_max_scaler.transform(oridnal_data)
        ordinal_scaled = pd.DataFrame(ordinal_scaled, index=oridnal_data.index, columns=oridnal_data.columns)
        # encoding categorical data
        categorical_data = X[self.__categorical_features].copy()
        categorical_encoded = pd.get_dummies(categorical_data)
        # encoding binary data
        binary_encoded = X[self.__binary_features].copy()
        binary_encoded['Gender'] = binary_encoded.Gender.apply(lambda x: 1 if x == 'Male' else 0)

        # combining
        testing_data = pd.concat([numerical_scaled, ordinal_scaled, categorical_encoded, binary_encoded], axis=1)

        # verifying all columns are present
        for column in self.__column_order:
            if column not in testing_data.keys():
                testing_data[column] = 0

        # specifying column order
        testing_data = testing_data[self.__column_order]

        return testing_data


    def predict(self, X: list[list]) -> list[str]:
        # convert to type DataFrame
        x_df = pd.DataFrame(data=X, columns=self.__original_columns)

        # prepare data before feeding to model
        prepared_data = self.__prepare_data(x_df)

        # make predictions
        predictions = self.__model.predict(prepared_data)
        
        # return labeled predictions
        return ['will exit' if value == 1 else 'will not exit' for value in predictions]
