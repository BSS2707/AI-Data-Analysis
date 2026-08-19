import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ollama  # Ollama client

# Function to query Ollama with gemma:2b
def query_ollama(prompt, model="gemma:2b"):
    response = ollama.chat(model=model, messages=[
        {"role": "user", "content": prompt}
    ])
    return response['message']['content']

# Streamlit UI
st.title("📊 AI Data Analysis Assistant (gemma:2b)")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Data Preview")
    st.dataframe(df.head())

    st.write("### 📈 Summary Statistics")
    st.write(df.describe())

    # Graph 1: Histogram
    st.write("### Histogram")
    column = st.selectbox("Choose a column for histogram", df.columns)
    fig, ax = plt.subplots()
    df[column].hist(ax=ax, bins=20, color="skyblue", edgecolor="black")
    ax.set_title(f"Distribution of {column}")
    st.pyplot(fig)

    # Graph 2: Correlation Heatmap (numeric only)
    st.write("### Correlation Heatmap")
    numeric_df = df.select_dtypes(include='number')  # keep only numeric columns
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for correlation heatmap.")

    # Graph 3: Boxplot
    st.write("### Boxplot")
    column_box = st.selectbox("Choose a column for boxplot", df.columns)
    fig, ax = plt.subplots()
    sns.boxplot(x=df[column_box], ax=ax, color="lightgreen")
    ax.set_title(f"Boxplot of {column_box}")
    st.pyplot(fig)

    # Graph 4: Scatter Plot
    st.write("### Scatter Plot")
    x_col = st.selectbox("X-axis", df.columns)
    y_col = st.selectbox("Y-axis", df.columns)
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax, color="orange")
    ax.set_title(f"{x_col} vs {y_col}")
    st.pyplot(fig)

    # AI Insights
    if st.button("🤖 Generate AI Insights"):
        prompt = f"""
        Analyze this dataset summary:\n{df.describe().to_string()}
        Correlation matrix:\n{numeric_df.corr().to_string()}
        Provide insights about distributions, correlations, and anomalies.
        """
        with st.spinner("gemma:2b is analyzing..."):
            insights = query_ollama(prompt)
        st.success("AI Insights:")
        st.write(insights)
