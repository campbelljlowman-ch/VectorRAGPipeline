from data_loaders.s3_data_loader import S3BatchDownloader


S3_BUCKET = "pinpointmigration2"

def run_pipeline():
    print("Running the Vector RAG Pipeline...")
    data_loader = S3BatchDownloader(bucket=S3_BUCKET, dest_dir="data/", prefix=None, chunk_size=5, state_path=".s3_progress.json")
    file_paths = data_loader.get_next_chunk()
    print(f"Downloaded {file_paths} files from S3 bucket '{S3_BUCKET}' to 'data/' directory.")
    
def asdf():
    # pip install openai
    from openai import OpenAI
    client = OpenAI()

    # 1) Upload your PDF
    pdf = client.files.create(
        file=open("report.pdf", "rb"),
        purpose="assistants"   # files for Responses/Assistants tools
    )

    # 2) Ask the model to use that file
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text",
                "text": "Summarize the attached PDF. Give bullet points and a 2-sentence abstract."},
                {"type": "input_file", "file_id": pdf.id}
            ]
        }]
    )

    print(resp.output_text)

if __name__ == "__main__":
    run_pipeline()

