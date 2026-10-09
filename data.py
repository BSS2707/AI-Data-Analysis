import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ollama

st.title("AI Data Analysis Assistant")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Dataset Information")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric("Duplicate Rows", df.duplicated().sum())

    st.subheader("Summary Statistics")
    st.dataframe(
        df.describe(include="all"),
        use_container_width=True
    )

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(exclude="number").columns.tolist()

    st.subheader("Histogram")

    if numeric_columns:

        histogram_column = st.selectbox(
            "Select column",
            numeric_columns,
            key="histogram"
        )

        fig, ax = plt.subplots()

        ax.hist(
            df[histogram_column].dropna(),
            bins=20,
            edgecolor="black"
        )

        ax.set_title("Histogram")
        ax.set_xlabel(histogram_column)
        ax.set_ylabel("Frequency")

        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("No numeric columns available.")

    st.subheader("Bar Chart")

    if numeric_columns and categorical_columns:

        category_column = st.selectbox(
            "Select category column",
            categorical_columns,
            key="bar_category"
        )

        numeric_column = st.selectbox(
            "Select numeric column",
            numeric_columns,
            key="bar_numeric"
        )

        chart_data = (
            df.groupby(category_column)[numeric_column]
            .sum()
            .sort_values(ascending=False)
            .head(20)
        )

        st.bar_chart(chart_data)

    else:
        st.info("Category and numeric columns are required.")

    st.subheader("Pie Chart")

    if categorical_columns:

        pie_column = st.selectbox(
            "Select category column",
            categorical_columns,
            key="pie"
        )

        pie_data = (
            df[pie_column]
            .fillna("Missing")
            .value_counts()
            .head(10)
        )

        fig, ax = plt.subplots()

        ax.pie(
            pie_data.values,
            labels=pie_data.index,
            autopct="%1.1f%%"
        )

        ax.set_title("Category Distribution")

        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("No categorical columns available.")

    st.subheader("Correlation Heatmap")

    if len(numeric_columns) >= 2:

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.heatmap(
            df[numeric_columns].corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("At least two numeric columns required.")

    st.subheader("Boxplot")

    if numeric_columns:

        box_column = st.selectbox(
            "Select column",
            numeric_columns,
            key="boxplot"
        )

        fig, ax = plt.subplots()

        sns.boxplot(
            x=df[box_column],
            ax=ax
        )

        ax.set_title("Boxplot")

        st.pyplot(fig)
        plt.close(fig)

        values = df[box_column].dropna()

        st.write("Minimum:", values.min())
        st.write("Q1:", values.quantile(0.25))
        st.write("Median:", values.median())
        st.write("Q3:", values.quantile(0.75))
        st.write("Maximum:", values.max())

    else:
        st.info("No numeric columns available.")

    st.subheader("Scatter Plot")

    if len(numeric_columns) >= 2:

        x_column = st.selectbox(
            "Select X Axis",
            numeric_columns,
            key="scatter_x"
        )

        y_column = st.selectbox(
            "Select Y Axis",
            numeric_columns,
            key="scatter_y"
        )

        fig, ax = plt.subplots()

        sns.scatterplot(
            data=df,
            x=x_column,
            y=y_column,
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)

        correlation = df[[x_column, y_column]].corr().iloc[0, 1]

        st.metric(
            "Correlation",
            round(correlation, 3)
        )

    else:
        st.info("At least two numeric columns required.")

    st.subheader("Column Information")

    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum(),
        "Unique Values": df.nunique()
    })

    st.dataframe(
        column_info,
        use_container_width=True
    )

    st.subheader("Ask AI About Dataset")

    question = st.text_input(
        "Enter your question"
    )

    if st.button("Ask AI") and question:

        dataset_info = f"""
Columns:
{list(df.columns)}

Summary:
{df.describe(include='all').to_string()}

Sample Data:
{df.head(10).to_string()}
"""

        prompt = f"""
You are an AI Data Analysis Assistant.

Dataset Information:
{dataset_info}

Question:
{question}

Answer in simple language.
Use only the dataset information.
Do not make up values.
"""

        try:

            with st.spinner("Generating answer..."):

                response = ollama.chat(
                    model="qwen2.5:1.5b",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

            st.subheader("AI Response")
            st.write(
                response["message"]["content"]
            )

        except Exception as e:
            st.error(f"Ollama Error: {e}")

else:
    st.info("Upload a CSV file to begin analysis.")
