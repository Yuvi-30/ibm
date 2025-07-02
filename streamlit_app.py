import streamlit as st
import pandas as pd
import google.generativeai as genai

# Get API key from secrets
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")  # or gemini-2.5-flash if you're enabled for it

st.title("🚗 Car Rental Customer Feedback Analyzer")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV with a 'review' column", type=["csv"])

# Process and analyze reviews
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "review" not in df.columns:
        st.error("CSV must contain a column named 'review'")
    else:
        st.info("Analyzing reviews using Gemini...")

        results = []
        for review in df["review"]:
            with st.spinner(f"Analyzing: {review[:60]}..."):
                prompt = f"""
                Analyze this car rental customer review:
                "{review}"

                1. Classify the sentiment as Positive, Neutral, or Negative.
                2. List any complaints or issues mentioned (like late delivery, bad condition, etc.).
                3. Give a short summary of the customer's overall experience.
                """
                try:
                    response = model.generate_content(prompt)
                    results.append({
                        "Review": review,
                        "AI Analysis": response.text.strip()
                    })
                except Exception as e:
                    results.append({
                        "Review": review,
                        "AI Analysis": f"Error: {str(e)}"
                    })

        result_df = pd.DataFrame(results)
        st.success("Analysis complete ✅")
        st.dataframe(result_df)

        # Download button
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results", csv, "feedback_analysis.csv", "text/csv")
