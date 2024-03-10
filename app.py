import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from io import BytesIO

def preprocess_data(input_file):
    # Read the input file
    if input_file.name.endswith('.xlsx'):
        df = pd.read_excel(input_file)
    elif input_file.name.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        raise ValueError("Input file format not supported. Only .xlsx and .csv files are supported.")

    # Retrieve column names and their original formats
    columns = df.columns
    dtypes = df.dtypes

    # Define preprocessing steps for numeric columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        
        ('scaler', StandardScaler())
    ])

    # Define preprocessing for all columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, df.select_dtypes(include='number').columns)
        ])

    # Fit the preprocessor to the data and transform the input file
    transformed_data = preprocessor.fit_transform(df)

    # Concatenate the preprocessed data with the original columns
    preprocessed_df = pd.DataFrame(transformed_data, columns=df.select_dtypes(include='number').columns)
    preprocessed_df = pd.concat([preprocessed_df, df.select_dtypes(exclude='number')], axis=1)

    # Save the preprocessed data to a bytes buffer
    output_buffer = BytesIO()
    preprocessed_df.to_csv(output_buffer, index=False)
    output_buffer.seek(0)

    return output_buffer

def main():
    st.title('File Upload and Preprocessing')

    # File upload section
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "csv"])
    if uploaded_file is not None:
        st.write('File uploaded successfully!')

        # Preprocess the uploaded file
        preprocessed_data = preprocess_data(uploaded_file)

        # Download the preprocessed data
        st.download_button(label='Download Preprocessed Data', data=preprocessed_data, file_name='preprocessed_data.csv', mime='text/csv')

if __name__ == "__main__":
    main()
