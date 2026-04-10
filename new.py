import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
def preprocess_text(text):
    doc = nlp(str(text).lower())
    tokens = []

    for token in doc:
        if not token.is_stop and not token.is_punct:
            tokens.append(token.lemma_)

    return " ".join(tokens)



def rank_resumes(job_description):

    if job_description.strip() == "":
        print("❌ Job description cannot be empty.")
        return

    try:
        
        df = pd.read_csv(
            "/Users/swadha/Downloads/resume.csv",
            encoding="utf-8"
        ).drop_duplicates(subset=["Resume"]).head(300)

    except:
        print("❌ CSV file not found. Check file path.")
        return

    resume_column = "Resume"

    if resume_column not in df.columns:
        print("❌ 'Resume' column not found.")
        print("Available columns:", df.columns)
        return

    print(f"\n📂 Total Resumes Loaded: {len(df)}")

    
    df["Processed_Resume"] = df[resume_column].apply(preprocess_text)

    
    processed_jd = preprocess_text(job_description)

   
    documents = [processed_jd] + df["Processed_Resume"].tolist()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(documents)

    
    similarity_scores = cosine_similarity(
        tfidf_matrix[0:1], tfidf_matrix[1:]
    ).flatten()

    df["Similarity (%)"] = similarity_scores * 100

   
    df_sorted = df.sort_values(by="Similarity (%)", ascending=False)

    print("\n" + "=" * 70)
    print("AI Resume Ranking Results (Top 5 Candidates)")
    print("=" * 70)

    for rank, (index, row) in enumerate(df_sorted.head(5).iterrows(), start=1):
        print(f"\nRank #{rank}")
        print(f"Resume ID       : {index}")
        print(f"Category        : {row['Category']}")
        print(f"Matching Score  : {row['Similarity (%)']:.2f}%")
        print(f"Resume Preview  : {row[resume_column][:120]}...")
        print("-" * 70)

    print("\n✅ Resume Ranking Completed Successfully!")



if __name__ == "__main__":

    print("\n=================================================")
    print("        AI-Based Resume Screening System        ")
    print("=================================================")

    job_description = input("\nEnter Job Description:\n>> ")

    rank_resumes(job_description)
