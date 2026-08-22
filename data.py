import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ollama

# ==========================================================
# FUNCTION TO QUERY OLLAMA
# ==========================================================

def query_ollama(prompt, model="qwen2.5:1.5b"):
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


# ==========================================================
# STREAMLIT UI
# ==========================================================

st.title("📊 AI Data Analysis Assistant")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)


# ==========================================================
# IF FILE UPLOADED
# ==========================================================

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # ======================================================
    # DATA PREVIEW
    # ======================================================

    st.write("### 🔍 Data Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    # ======================================================
    # DATA INFORMATION
    # ======================================================

    st.write("### 📋 Dataset Information")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        df.shape[0]
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    col3.metric(
        "Missing Values",
        df.isnull().sum().sum()
    )

    col4.metric(
        "Duplicate Rows",
        df.duplicated().sum()
    )


    # ======================================================
    # SUMMARY STATISTICS
    # ======================================================

    st.write("### 📈 Summary Statistics")

    st.dataframe(
        df.describe(include="all"),
        use_container_width=True
    )


    # ======================================================
    # COLUMN TYPES
    # ======================================================

    numeric_df = df.select_dtypes(
        include="number"
    )

    numeric_columns = numeric_df.columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude="number"
    ).columns.tolist()


    # ======================================================
    # 1. HISTOGRAM
    # ======================================================

    st.write("## 📊 Histogram")

    if len(numeric_columns) > 0:

        column = st.selectbox(
            "Choose a column for histogram",
            numeric_columns,
            key="histogram_column"
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        values = df[column].dropna()

        counts, bins, patches = ax.hist(
            values,
            bins=20,
            color="skyblue",
            edgecolor="black"
        )

        ax.set_title(
            f"Distribution of {column}"
        )

        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

        # Show numbers on histogram
        for count, patch in zip(
            counts,
            patches
        ):

            if count > 0:

                ax.text(
                    patch.get_x()
                    + patch.get_width() / 2,
                    count,
                    str(int(count)),
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.warning(
            "No numeric columns available."
        )


    # ======================================================
    # 2. BAR CHART
    # ======================================================

    st.write("## 📊 Bar Chart")

    if (
        len(categorical_columns) > 0
        and len(numeric_columns) > 0
    ):

        col1, col2 = st.columns(2)

        with col1:

            bar_x = st.selectbox(
                "Select category",
                categorical_columns,
                key="bar_x"
            )

        with col2:

            bar_y = st.selectbox(
                "Select numeric column",
                numeric_columns,
                key="bar_y"
            )

        # Group data
        bar_data = (
            df.groupby(bar_x)[bar_y]
            .sum()
            .sort_values(
                ascending=False
            )
            .head(20)
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        bars = ax.bar(
            bar_data.index.astype(str),
            bar_data.values,
            color="steelblue"
        )

        ax.set_title(
            f"{bar_y} by {bar_x}"
        )

        ax.set_xlabel(bar_x)
        ax.set_ylabel(bar_y)

        plt.xticks(
            rotation=45,
            ha="right"
        )

        # Show value on bar
        for bar in bars:

            height = bar.get_height()

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                height,
                f"{height:,.2f}",
                ha="center",
                va="bottom"
            )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.warning(
            "Bar chart requires one categorical "
            "and one numeric column."
        )


    # ======================================================
    # 3. PIE CHART
    # ======================================================

    st.write("## 🥧 Pie Chart")

    if len(categorical_columns) > 0:

        pie_column = st.selectbox(
            "Choose a column for pie chart",
            categorical_columns,
            key="pie_column"
        )

        pie_data = (
            df[pie_column]
            .fillna("Missing")
            .value_counts()
            .head(10)
        )

        fig, ax = plt.subplots(
            figsize=(8, 8)
        )

        ax.pie(
            pie_data.values,
            labels=pie_data.index,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title(
            f"Distribution of {pie_column}"
        )

        st.pyplot(fig)

        plt.close(fig)

        # Show pie values
        st.write("### 📋 Pie Chart Values")

        pie_table = pd.DataFrame({
            "Category": pie_data.index,
            "Count": pie_data.values
        })

        pie_table["Percentage"] = (
            pie_table["Count"]
            / pie_table["Count"].sum()
            * 100
        ).round(2)

        st.dataframe(
            pie_table,
            use_container_width=True
        )

    else:

        st.warning(
            "No categorical columns available."
        )


    # ======================================================
    # 4. CORRELATION HEATMAP
    # ======================================================

    st.write("## 🔥 Correlation Heatmap")

    if len(numeric_columns) >= 2:

        correlation = numeric_df.corr()

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax
        )

        ax.set_title(
            "Correlation Heatmap"
        )

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.warning(
            "At least two numeric columns "
            "are required for heatmap."
        )


    # ======================================================
    # 5. BOXPLOT
    # ======================================================

    st.write("## 📦 Boxplot")

    if len(numeric_columns) > 0:

        column_box = st.selectbox(
            "Choose a column for boxplot",
            numeric_columns,
            key="boxplot_column"
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.boxplot(
            x=df[column_box],
            ax=ax,
            color="lightgreen"
        )

        ax.set_title(
            f"Boxplot of {column_box}"
        )

        ax.set_xlabel(column_box)

        st.pyplot(fig)

        plt.close(fig)

        # Show statistics
        data = df[column_box].dropna()

        st.write("### 📋 Boxplot Statistics")

        stats = pd.DataFrame({
            "Statistic": [
                "Minimum",
                "Q1",
                "Median",
                "Q3",
                "Maximum"
            ],
            "Value": [
                data.min(),
                data.quantile(0.25),
                data.median(),
                data.quantile(0.75),
                data.max()
            ]
        })

        st.dataframe(
            stats,
            use_container_width=True
        )

    else:

        st.warning(
            "No numeric columns available."
        )


    # ======================================================
    # 6. SCATTER PLOT
    # ======================================================

    st.write("## 🔵 Scatter Plot")

    if len(numeric_columns) >= 2:

        col1, col2 = st.columns(2)

        with col1:

            x_col = st.selectbox(
                "X-axis",
                numeric_columns,
                key="scatter_x"
            )

        with col2:

            y_col = st.selectbox(
                "Y-axis",
                numeric_columns,
                key="scatter_y"
            )

        show_values = st.checkbox(
            "Show values on points",
            value=False
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.scatterplot(
            x=df[x_col],
            y=df[y_col],
            ax=ax,
            color="orange",
            s=70
        )

        ax.set_title(
            f"{x_col} vs {y_col}"
        )

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

        # Show values on points
        if show_values:

            data = df[
                [x_col, y_col]
            ].dropna().head(100)

            for _, row in data.iterrows():

                ax.annotate(
                    f"({row[x_col]:.1f}, "
                    f"{row[y_col]:.1f})",
                    (
                        row[x_col],
                        row[y_col]
                    ),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=7
                )

        st.pyplot(fig)

        plt.close(fig)

        # Correlation
        correlation_value = df[
            [x_col, y_col]
        ].corr().iloc[0, 1]

        st.metric(
            "Correlation",
            f"{correlation_value:.3f}"
        )

    else:

        st.warning(
            "At least two numeric columns "
            "are required for scatter plot."
        )


    # ======================================================
    # 7. DATA TYPES
    # ======================================================

    st.write("## 🧾 Column Information")

    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing": df.isnull().sum(),
        "Unique Values": df.nunique()
    })

    st.dataframe(
        column_info,
        use_container_width=True
    )


    # ======================================================
    # 8. ASK OLLAMA
    # ======================================================

    st.write("## 🤖 Ask Questions About Your Dataset")

    question = st.text_input(
        "Ask a question about the uploaded CSV:",
        placeholder=(
            "Example: Which column has the highest average?"
        )
    )

    if st.button(
        "🤖 Ask Ollama"
    ) and question:

        # Create dataset context
        dataset_info = f"""
Dataset shape:
{df.shape}

Columns:
{list(df.columns)}

Data types:
{df.dtypes.to_string()}

Summary statistics:
{df.describe(include="all").to_string()}

Sample data:
{df.head(10).to_string()}

Correlation matrix:
{
    numeric_df.corr().to_string()
    if not numeric_df.empty
    else "No numeric columns"
}
"""

        prompt = f"""
You are an AI data analysis assistant.

Answer the user's question using
the dataset information below.

Dataset information:
{dataset_info}

User question:
{question}

Instructions:
- Answer clearly and concisely.
- Use the dataset information.
- Do not invent values.
- If the information is not available,
  say it cannot be determined.
"""

        with st.spinner(
            "Ollama is thinking..."
        ):

            answer = query_ollama(
                prompt
            )

        st.write("### 💬 Answer")

        st.write(answer)
