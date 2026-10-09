import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ollama

st.title("AI Data Analysis Assistant")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
df = pd.read_csv(uploaded_file)

```
st.subheader("Data Preview")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Dataset Information")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", df.isnull().sum().sum())
col4.metric("Duplicate Rows", df.duplicated().sum())

st.subheader("Summary Statistics")
st.dataframe(df.describe(include="all"), use_container_width=True)

numeric_columns = df.select_dtypes(include="number").columns.tolist()
categorical_columns = df.select_dtypes(exclude="number").columns.tolist()

st.subheader("Histogram")

if numeric_columns:
    column = st.selectbox("Select a column", numeric_columns)

    fig, ax = plt.subplots()
    ax.hist(df[column].dropna(), bins=20, edgecolor="black")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram")
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("No numeric columns available.")

st.subheader("Bar Chart")

if numeric_columns and categorical_columns:
    category = st.selectbox("Select category", categorical_columns)
    number = st.selectbox("Select numeric column", numeric_columns)

    chart_data = df.groupby(category)[number].sum().nlargest(20)

    st.bar_chart(chart_data)
else:
    st.info("A category column and a numeric column are required.")

st.subheader("Pie Chart")

if categorical_columns:
    pie_column = st.selectbox("Select a column for pie chart", categorical_columns)
    pie_data = df[pie_column].fillna("Missing").value_counts().head(10)

    fig, ax = plt.subplots()
    ax.pie(pie_data.values, labels=pie_data.index, autopct="%1.1f%%")
    ax.set_title("Category Distribution")
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        pd.DataFrame({
            "Category": pie_data.index,
            "Count": pie_data.values,
            "Percentage": (pie_data.values / pie_data.sum() * 100).round(2)
        }),
        use_container_width=True
    )
else:
    st.info("No category columns available.")

st.subheader("Correlation Heatmap")

if len(numeric_columns) >= 2:
    fig, ax = plt.subplots()
    sns.heatmap(df[numeric_columns].corr(), annot=True, fmt=".2f", ax=ax)
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("At least two numeric columns are required.")

st.subheader("Boxplot")

if numeric_columns:
    box_column = st.selectbox("Select a column for boxplot", numeric_columns)

    fig, ax = plt.subplots()
    sns.boxplot(x=df[box_column], ax=ax)
    ax.set_title("Boxplot")
    st.pyplot(fig)
    plt.close(fig)

    values = df[box_column].dropna()

    st.write("Minimum:", values.min())
    st.write("First Quartile:", values.quantile(0.25))
    st.write("Median:", values.median())
    st.write("Third Quartile:", values.quantile(0.75))
    st.write("Maximum:", values.max())
else:
    st.info("No numeric columns available.")

st.subheader("Scatter Plot")

if len(numeric_columns) >= 2:
    x_column = st.selectbox("Select X-axis", numeric_columns)
    y_column = st.selectbox("Select Y-axis", numeric_columns)

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_column, y=y_column, ax=ax)
    ax.set_title("Scatter Plot")
    st.pyplot(fig)
    plt.close(fig)

    correlation = df[[x_column, y_column]].corr().iloc[0, 1]
    st.metric("Correlation", f"{correlation:.3f}")
else:
    st.info("At least two numeric columns are required.")

st.subheader("Column Information")

column_info = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Missing Values": df.isnull().sum().values,
    "Unique Values": df.nunique().values
})

st.dataframe(column_info, use_container_width=True)

st.subheader("Ask Questions About Your Dataset")

question = st.text_input(
    "Enter your question",
    placeholder="Which column has the highest average?"
)

if st.button("Ask AI") and question:
    dataset_info = f"""
    Columns: {list(df.columns)}

    Summary:
    {df.describe(include="all").to_string()}

    Sample Data:
    {df.head(10).to_string()}
    """

    prompt = f"""
    You are an AI data analysis assistant.

    Use this dataset information to answer the question.

    {dataset_info}

    Question: {question}

    Explain the answer in simple language.
    Do not invent values.
    If the answer cannot be determined, say so.
    """

    try:
        with st.spinner("AI is thinking..."):
            response = ollama.chat(
                model="qwen2.5:1.5b",
                messages=[{"role": "user", "content": prompt}]
            )

        st.subheader("AI Answer")
        st.write(response["message"]["content"])

    except Exception as error:
        st.error(f"Could not connect to Ollama: {error}")
```

else:
st.info("Upload a CSV file to start analyzing your data.")
