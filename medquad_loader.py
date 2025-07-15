import pandas as pd

def load_medquad(filepath):
    '''
    Loads the MedQuAD CSV dataset using pandas and returns a dataframe.
    Expected columns: ['title', 'question', 'answer',...]
    '''
    try:
        # Load the CSV file
        df = pd.read_csv(filepath)
        print("MedQuAD data loaded successfully!")
        print(df.head()) # show first 5 rows
        return df

    except FileNotFoundError:
        print("File not found. Check your filepath")
        return None
    except Exception as e:
        print("Error loading MedQuAD:", e)
        return None

# Retrieves top_n QA pairs relevant to query and symptoms
def retrieve_relevant_answers(df, query, symptoms=None, top_n=3):
    try:
        query_lower = query.lower()

        if symptoms:
            symptom_query = " ".join(symptoms).lower()
            combined_query = query_lower + " " + symptom_query
        else:
            combined_query = query_lower

        # Filter rows where the question or answer contain relevant information
        matched_rows = df[
            df['question'].str.lower().str.contains(combined_query) |
            df['answer'].str.lower().str.contains(combined_query)]

        if matched_rows.empty:
            return []

        top_results = matched_rows.head(top_n)

        qa_pairs = []
        for _, row in top_results.iterrows():
            qa_pairs.append(f"Q: {row['question']} A: {row['answer']}")

        return qa_pairs

    except Exception as e:
        print("Error in retrieve_relevant_answers", e)
        return []


if __name__ == "__main__":
    filepath = "Kaggle - MedQuAD/medquad.csv"
    medquad_df = load_medquad(filepath)